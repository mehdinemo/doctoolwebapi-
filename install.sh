#!/usr/bin/env bash

APP=doctools_web_api
USER=meti
GIT=http://gitlab.datamining.io
APP_SRC=${GIT}/${USER}/${APP}/-/archive/master/${APP}-master.tar
INS_SRC=${GIT}/cleanism/installer/-/archive/master/installer-master.tar

if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root"
   exit 1
fi

cd /tmp
curl -sO ${APP_SRC}
curl -sO ${INS_SRC}
tar xf ${APP}-master.tar
tar xf installer-master.tar
cp -r installer-master/installer/ ${APP}-master/
cd ${APP}-master
python3 install.py
