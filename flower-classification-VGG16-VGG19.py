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

from tensorflow.keras.optimizers import RMSprop, Adagrad, SGD, Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.efficientnet import decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras import Model


#Splitting the dataset
splitfolders.ratio('flowers', output="output", seed=101, ratio=(0.8, 0.1, 0.1))      # (train:val:test)

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

train_datagen = ImageDataGenerator(rescale = 1.0/255.0,
                                   rotation_range = 0.5,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.1,
                                   horizontal_flip = True,
                                   fill_mode = 'nearest'
                                  )

test_val_datagen = ImageDataGenerator(rescale = 1.0/255.0)

#For the parameter in the flow from directory
#I rescale it to 128x128 and use 32 batch size you can try other size 
batch_size = 64
img_width = 128
img_height = 128

train_generator = train_datagen.flow_from_directory(directory = path_train,
                                                    batch_size = batch_size,
                                                    class_mode = "categorical",
                                                    target_size = (img_width, img_height)
                                                    )

val_generator = test_val_datagen.flow_from_directory(directory = path_val,
                                                    batch_size = batch_size,
                                                    class_mode = "categorical",
                                                    target_size = (img_width, img_height)
                                                    )

test_generator = test_val_datagen.flow_from_directory(directory = path_test,
                                                    batch_size = batch_size,
                                                    class_mode = "categorical",
                                                    target_size = (img_width, img_height)
                                                    )


base_model = VGG19(weights='imagenet', include_top=False, input_shape=(128, 128,3))

# freeze extraction layers
base_model.trainable = False

# add custom top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Flatten()(base_model.output)
predictions = Dense(13, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
print(model.summary())

# confirm unfrozen layers
for layer in model.layers:
    if layer.trainable==True:
        print(layer)

from tensorflow.keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + 
    K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

#Compile the model
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics=['accuracy', recall_m, precision_m, f1_m])


# For the training. I try to use 100 epoch before this and i got 85% validation accuracy and 87% training accuracy.
# But to make it shorter i use 50% instead and it should be around 82% - 84% for the validation accuracy.

#Training the model
epoch_size = 20
history = model.fit(train_generator, validation_data = val_generator,
                    epochs = epoch_size,
                    verbose = 1,
                    shuffle = True)

model.evaluate(test_generator)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

f1 = history.history['f1_m']
val_f1 = history.history['val_f1_m']

precision = history.history['precision_m']
val_precision = history.history['val_precision_m']

recall = history.history['recall_m']
val_recall = history.history['val_recall_m']

epochs_range = range(epoch_size)

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy (VGG19)')

plt.subplot(2, 1, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss (VGG19)')
plt.show()

plt.figure(figsize=(8, 8))
plt.subplot(1, 1, 1)
plt.plot(epochs_range, loss, label='Training F1')
plt.plot(epochs_range, val_loss, label='Validation F1')
plt.legend(loc='lower right')
plt.title('Training and Validation F1 (VGG19)')
plt.show()

plt.figure(figsize=(8, 8))
plt.subplot(1, 1, 1)
plt.plot(epochs_range, loss, label='Training precision')
plt.plot(epochs_range, val_loss, label='Validation precision')
plt.legend(loc='lower right')
plt.title('Training and Validation precision (VGG19)')
plt.show()

plt.figure(figsize=(8, 8))
plt.subplot(1, 1, 1)
plt.plot(epochs_range, loss, label='Training recall')
plt.plot(epochs_range, val_loss, label='Validation recall')
plt.legend(loc='lower right')
plt.title('Training and Validation recall (VGG19)')
plt.show()

from tensorflow.keras.models import load_model
model.save('VGG19_model.h5')