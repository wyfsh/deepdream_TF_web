#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from flask import Flask, request, redirect, url_for, send_from_directory
from werkzeug import secure_filename
from time import strftime

import PIL.Image
import numpy as np
import tensorflow as tf
from dream import *

UPLOAD_FOLDER = 'upload/'
RESULT_FOLDER = 'result/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filename = strftime('%Y%m%d%H%M%S') + '.' + filename.rsplit('.', 1)[1]
            fullpath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(fullpath)

            filename = 'd' + filename
            dream(fullpath, filename)

            return redirect(url_for('download', filename=filename))

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''


def dream(fullpath, filename):
    img0 = PIL.Image.open(fullpath)
    img0 = np.float32(img0)
    layer = 'mixed3a'  # 'mixed4c'
    render_lap_deepdream(tf.square(T(layer)), img0, filename)


@app.route('/dream/<filename>')
def download(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
