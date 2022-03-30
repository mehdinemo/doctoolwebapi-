from flask import Flask, render_template, request, url_for, jsonify, Response
from werkzeug.wsgi import FileWrapper
from werkzeug.datastructures import Headers
import os
import json
import logging
from io import BytesIO
import pandas as pd
from doctools import app as docapp

app = Flask(__name__)

logging.basicConfig(filename='log.log', level=logging.DEBUG)


@app.route('/')
def first_page():
    return render_template('first-page.html')


@app.route('/message-classification/', methods=['POST', 'GET'])
def message_classification():
    tagged_columns = ['id', 'text', 'tag']
    untagged_columns = ['id', 'text']

    if request.method == 'GET':
        url = url_for('message_classification')
        return render_template('message-classification.html', url=url)

    parameters = request.form

    if 'messages_train' not in request.files or 'messages_test' not in request.files:
        return jsonify('No file part')
    file_train = request.files['messages_train']
    file_test = request.files['messages_test']

    try:
        tagged_messages = _file_worker(file_train)
    except Exception as ex:
        return jsonify(str(ex))
    is_valid, validation_message = _reset_id(tagged_messages, tagged_columns)
    if not is_valid:
        return jsonify(validation_message)

    try:
        untagged_messages = _file_worker(file_test)
    except Exception as ex:
        return jsonify(str(ex))
    is_valid, validation_message = _reset_id(untagged_messages, untagged_columns)
    if not is_valid:
        return jsonify(validation_message)

    try:
        messages = docapp.document_classification(tagged_messages=tagged_messages, untagged_messages=untagged_messages,
                                                  method=parameters['method'],
                                                  sub_method=parameters['sub_method'],
                                                  label_method=parameters['label_method'],
                                                  n_head_score=float(parameters['n_head_score']))
    except Exception as ex:
        return jsonify(str(ex))

    messages.drop(['id'], axis=1, inplace=True)
    messages.rename(columns={'origin_id': 'id'}, inplace=True)
    messages['id'] = messages['id'].astype('int64').astype(str)

    io = _io_builder(messages, parameters['format'])
    try:
        w = FileWrapper(io)
        d = Headers()
        d.add('Content-Disposition', 'attachment', filename='results.' + parameters['format'])
        return Response(w, mimetype="application/x-binary", direct_passthrough=True, headers=d)
    except Exception as ex:
        return jsonify(f'Can not send file, Error: {ex}')


@app.route('/message-clustering/', methods=['POST', 'GET'])
def message_clustering():
    if request.method == 'GET':
        url = url_for('message_clustering')
        return render_template('message-clustering.html', url=url)

    parameters = request.form
    noise_deletion = False if 'noise_deletion' not in parameters else parameters['noise_deletion']

    if 'messages_file' not in request.files:
        return jsonify('No file part')
    file = request.files['messages_file']

    try:
        messages_df = _file_worker(file)
    except Exception as ex:
        return jsonify(str(ex))

    messages_columns = ['id', 'text']
    is_valid, validation_message = _reset_id(messages_df, messages_columns)
    if not is_valid:
        return jsonify(validation_message)

    try:
        messages_df = docapp.document_clustering(messages_df=messages_df,
                                                 prune_thresh=float(parameters['prune_thresh']),
                                                 noise_deletion=noise_deletion, eps=float(parameters['eps']),
                                                 min_samples=int(parameters['min_samples']))
    except Exception as ex:
        return jsonify(str(ex))

    messages_df.drop(['id'], axis=1, inplace=True)
    messages_df.rename(columns={'origin_id': 'id'}, inplace=True)
    messages_df['id'] = messages_df['id'].astype('int64').astype(str)

    io = _io_builder(messages_df, parameters['format'])
    try:
        w = FileWrapper(io)
        d = Headers()
        d.add('Content-Disposition', 'attachment', filename='results.' + parameters['format'])
        return Response(w, mimetype="application/x-binary", direct_passthrough=True, headers=d)
    except Exception as ex:
        return jsonify(f'Can not send file, Error: {ex}')


@app.route('/similar-messages/', methods=['POST', 'GET'])
def similar_messages():
    if request.method == 'GET':
        url = url_for('similar_messages')
        return render_template('similar-messages.html', url=url)

    parameters = request.form
    if 'messages_file' not in request.files:
        return jsonify('No file part')
    file = request.files['messages_file']

    try:
        messages_df = _file_worker(file)
    except Exception as ex:
        return jsonify(str(ex))

    messages_columns = ['id', 'text']
    is_valid, validation_message = _reset_id(messages_df, messages_columns)
    if not is_valid:
        return jsonify(validation_message)

    try:
        messages_df = docapp.similar_messages(messages_df=messages_df, sim_thresh=float(parameters['sim_thresh']))
    except Exception as ex:
        return jsonify(str(ex))

    messages_df.drop(['id'], axis=1, inplace=True)
    messages_df.rename(columns={'origin_id': 'id'}, inplace=True)
    messages_df['id'] = messages_df['id'].astype('int64').astype(str)

    io = _io_builder(messages_df, parameters['format'])

    try:
        w = FileWrapper(io)
        d = Headers()
        d.add('Content-Disposition', 'attachment', filename='results.' + parameters['format'])
        return Response(w, mimetype="application/x-binary", direct_passthrough=True, headers=d)
    except Exception as ex:
        return jsonify(f'Can not send file, Error: {ex}')


def _file_worker(file):
    filename, file_extension = os.path.splitext(file.filename)
    if file_extension not in ['.json', '.xlsx', '.csv']:
        return {'error': 'File extension not supported!'}

    try:
        file.seek(0)
        req_body = file.read()
    except Exception as ex:
        return {'error': f'Could not Read file {format(file.name)}! Error: {ex}'}

    if file_extension == '.xlsx':
        toread = BytesIO()
        toread.write(req_body)
        toread.seek(0)
        df = pd.read_excel(toread)
    elif file_extension == '.csv':
        toread = BytesIO()
        toread.write(req_body)
        toread.seek(0)
        df = pd.read_csv(toread)
    elif file_extension == '.json':
        req_body = json.loads(req_body)
        df = pd.DataFrame(req_body['messages'])

    return df


def _io_builder(messages, p_format):
    io = BytesIO()
    if p_format == 'xlsx':
        writer = pd.ExcelWriter(io, engine='xlsxwriter')
        messages.to_excel(writer, sheet_name='messages')
        writer.save()
        io.seek(0)
    elif p_format == 'csv':
        io.write(messages.to_csv(index=False, escapechar='\n').encode('utf-8'))
        io.seek(0)
    elif p_format == 'json':
        messages_dic = {'messages': messages.to_dict("records")}
        result_str = json.dumps(messages_dic, ensure_ascii=False, indent=4)
        io.write(result_str.encode('utf-8'))
        io.seek(0)

    return io


def _reset_id(messages_df, messages_columns):
    for col in messages_columns:
        if not col in messages_df.columns:
            return False, f'{col} does not exist in data!'
    if messages_df.drop(messages_df.columns.difference(messages_columns), axis=1).isnull().values.any():
        return False, f'some messages dont have require data!'

    messages_df['id'] = messages_df['id'].astype('int64')
    messages_df.reset_index(inplace=True)
    messages_df.rename(columns={'id': 'origin_id', 'index': 'id'}, inplace=True)
    messages_df['id'] = messages_df['id'].astype(int)

    return True, messages_df


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
