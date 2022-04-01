# import tensorflow as tf
import base64
import json

import cv2
import os
import plotext as plt
# import matplotlib.pyplot as plt
import numpy as np

img_array = cv2.imread("mht-data/train/0/Training_143373.jpg")
# print(img_array)
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route('/image', methods=['POST'])
def img():
    data = request.data
    data = json.loads(data)
    data = data['base64']
    data = data.split(',')[1]

    bdata = base64.b64decode(data)
    filename = 'some_image.jpg'  # I assume you have a way of picking unique filenames
    with open(filename, 'wb') as f:
        f.write(bdata)
    # print(data)
    return 'Success'


if __name__ == '__main__':
   app.run(debug = True)

