from flask import render_template, redirect, url_for, abort, flash, request, current_app, make_response, send_from_directory
from . import main
import os
import subprocess

@main.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join('static'), 'asm-logo-full.62f3fd.png', mimetype='image/vnd.microsoft.icon')

@main.route('/')
def home():
    return render_template('home.html')

@main.route('/test')
def test():
    return render_template('test.html')
