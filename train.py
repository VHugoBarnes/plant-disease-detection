import numpy as np
import pathlib
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from tensorflow.python.ops.gen_batch_ops import batch

filepath = os.path.dirname(os.path.abspath(__file__))

training_img_dir = filepath + '/cassava/train'
test_img_dir = filepath + '/cassava/test'

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_it = datagen.flow_from_directory(training_img_dir, batch_size=8, target_size=(1024,1024))
test_it = datagen.flow_from_directory(test_img_dir, batch_size=8, target_size=(1024, 1024))

train_images, train_labels = train_it.next()
test_images, test_labels = test_it.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (train_images.shape,train_images.min(), train_images.max()))

def build_cnn():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', strides=(2,2),
                input_shape=(1024, 1024, 3)))
    model.add(tf.keras.layers.MaxPooling2D((2,2)))
    model.add(tf.keras.layers.Conv2D(64, (3,3), strides=(2,2), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2,2)))
    model.add(tf.keras.layers.Conv2D(128, (3,3), strides=(2,2), activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))

    return model

model = build_cnn()
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit_generator(train_it,
                              epochs=10,
                              steps_per_epoch=16,
                              validation_data=test_it,
                              validation_steps=8)

model_path = filepath + '/models/cassavacnn'
model.save(filepath=model_path)

print(history.history.keys())
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Metrics')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(test_acc)
