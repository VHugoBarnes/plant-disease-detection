import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import cv2
import os

filepath = os.path.dirname(os.path.abspath(__file__))

img_path = filepath + '/test-4.jpg'
model_path = filepath + '/models/cassavacnn'

img = image.load_img(img_path, target_size=(1024,1024))
numpy_img = np.array(img)
numpy_img = np.expand_dims(numpy_img, axis=0)

# print(type(numpy_img))
# print(numpy_img.shape)

# Carga y crea el modelo exacto, incluyendo los pesos y el optimizador
model = tf.keras.models.load_model(model_path)

# Predice la clase de la imagen de entrada al modelo cargado
predicted = model.predict(numpy_img)
print('Predicted', predicted)

# label = ''
# comparison = [[0.,1.]] == predicted

# if (comparison.all()):
#     label = 'Con neumonía'
# else:
#     label = 'Sin neoumonía'

# print(label)
# plt.imshow(img)
# plt.xlabel(label)
# plt.show()
