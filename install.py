#!/usr/bin/env python3

from installer import Installer

installer = Installer(root=True)
app = 'doctools_web_api'
dst = '/srv/' + app


def main():
    installer.run(
        'Update APT',
        'apt-get update')
    installer.run(
        'Upgrade packages',
        'DEBIAN_FRONTEND=noninteractive && apt-get -y --force-yes upgrade')
    installer.run(
        'Install prerequisites',
        'apt-get install -y python3.7-venv python3.7-dev gcc nginx')
    installer.run(
        'Remove old source codes',
        f'rm -r {dst}',
        die=False)
    installer.run(
        'Copy files',
        f'cp -r src/ {dst}')
    installer.run(
        'Create Virtual Environment',
        f'python3.7 -m venv {dst}/venv')
    installer.run(
        'Install Python Packages',
        f'source {dst}/venv/bin/activate && pip install -r requirements.txt')
    installer.run(
        'Copy uWSGI File',
        f'cp configs/{app}.ini {dst}')
    installer.run(
        'Copy systemd Service',
        f'cp configs/{app}.service /etc/systemd/system/')
    installer.run(
        'Copy Nginx File',
        f'cp configs/{app}.nginx /etc/nginx/sites-available/{app}')
    installer.run(
        'Enable Nginx Config File',
        f'ln -s /etc/nginx/sites-available/{app} /etc/nginx/sites-enabled/',
        die=False)
    installer.run(
        'Add new user',
        'useradd -m -s /bin/bash servant',
        die=False)
    installer.run(
        'Change Owner of Files',
        f'chown -R servant:www-data {dst}')
    installer.run(
        'Restart Service',
        f'systemctl daemon-reload && systemctl restart {app}')
    installer.run(
        'Enable Service',
        f'systemctl enable {app}')
    installer.run(
        'Restart Nginx',
        'systemctl restart nginx')


if __name__ == '__main__':
    main()
