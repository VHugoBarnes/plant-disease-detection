import base64
import json
from io import BytesIO

import numpy as np
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

@app.route('/classify/', methods=['POST'])
def predict():
    filepath = os.path.dirname(os.path.abspath(__file__))

    img_path = filepath + '/test-4.jpg'
    model_path = filepath + '/models/cassavacnn'

    img = image.load_img(img_path, target_size=(1024,1024))
    numpy_img = np.array(img)
    numpy_img = np.expand_dims(numpy_img, axis=0)

    # Carga y crea el modelo exacto, incluyendo los pesos y el optimizador
    model = tf.keras.models.load_model(model_path)

    # Predice la clase de la imagen de entrada al modelo cargado
    predicted = model.predict(numpy_img)

    # CBB
    # CBSD
    # CGM
    # CMD
    # HEALTHY

    condicion = 'Null'

    if(([[1., 0., 0., 0., 0.]] == predicted).all()):
        condicion = 'CBB'
    elif(([[0., 1., 0., 0., 0.]] == predicted).all()):
        condicion = 'CBSD'
    elif(([[0., 0., 1., 0., 0.]] == predicted).all()):
        condicion = 'CGM'
    elif(([[0., 0., 0., 1., 0.]] == predicted).all()):
        condicion = 'CMD'
    elif(([[0., 0., 0., 0., 1.]] == predicted).all()):
        condicion = 'HEALTHY'
    
    return jsonify(condicion);

if __name__ == '__main__':
    
    app.run(port=3000, debug=True)