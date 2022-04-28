#!/usr/bin/env python
# coding: utf-8

import os
from random import shuffle 
import numpy as np
import splitfolders
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

#Splitting the dataset
splitfolders.ratio('flowers', output="output", seed=101, ratio=(.8, 0.1, 0.1))      # (train:val:test)

#Setting up the parameter

#For the parameter in the flow from directory
#I rescale it to 128x128 and use 32 batch size you can try other size 
batch_size = 32
img_width = 128
img_height = 128

#The augmetnation parameter
rotation_range = 0.4
width_shift_range = 0.2
height_shift_range = 0.3
shear_range = 0.2
zoom_range = 0.2

#Making the Image Data Generator

#The path for the data
path_train = "./output/train"
path_test = "./output/test"
path_val = "./output/val"

#We use ImageDataGenerator to help us augment the image
train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range = rotation_range,
                                   width_shift_range = width_shift_range,
                                   height_shift_range = height_shift_range,
                                   shear_range = shear_range,
                                   zoom_range = zoom_range,
                                   horizontal_flip = True,
                                   fill_mode = 'nearest'
                                  )

train_generator = train_datagen.flow_from_directory(directory = path_train,
                                                    batch_size = batch_size,
                                                    class_mode = "categorical",
                                                    target_size = (img_width, img_height)
                                                    )

test_datagen = ImageDataGenerator(rescale = 1./255.)

test_generator = test_datagen.flow_from_directory(directory = path_test,
                                                  batch_size = batch_size,
                                                  class_mode = "categorical",
                                                  target_size = (img_width, img_height)
                                                  )

valid_datagen = ImageDataGenerator(rescale = 1./255.)

valid_generator = test_datagen.flow_from_directory(directory = path_val,
                                                  batch_size = batch_size,
                                                  class_mode = "categorical",
                                                  target_size = (img_width, img_height)
                                                  )


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten

model = Sequential()
model.add(Conv2D(16, (3,3), activation='relu', input_shape=(128,128,3), padding='same'))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(2, 2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(13, activation='softmax'))

print(model.summary())

#Compile the model
from tensorflow.keras.optimizers import RMSprop, Adagrad, SGD, Adam
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics=['accuracy'])


# For the training. I try to use 100 epoch before this and i got 85% validation accuracy and 87% training accuracy.
# But to make it shorter i use 50% instead and it should be around 82% - 84% for the validation accuracy.

#Training the model

epoch_size = 70
history = model.fit(train_generator, validation_data = valid_generator,
                    epochs = epoch_size,
                    verbose = 1,
                    shuffle = True)

model.evaluate(test_generator)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epoch_size)

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

_, acc = model.evaluate(test_generator, steps=len(test_generator), verbose=0)
print('Accuracy: %.3f' % (acc * 100.0))

from tensorflow.keras.models import load_model
model.save('Simple_model.h5')