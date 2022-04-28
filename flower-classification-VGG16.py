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
from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.efficientnet import decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras import Model

# 資料前處理
# 把資料分成 train, validation, test : 8:1:1
splitfolders.ratio('flowers', output="output", seed=101, ratio=(0.8, 0.1, 0.1))      # (train:val:test)

# 參數設定
rotation_range = 0.4
width_shift_range = 0.2
height_shift_range = 0.3
shear_range = 0.2
zoom_range = 0.2

# Making the Image Data Generator
# 資料夾路徑
path_train = "./output/train"
path_test = "./output/test"
path_val = "./output/val"

# We use ImageDataGenerator to help us augment the image
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

# 呼叫 VGG16 模型
# include_top=False  -> 只利用VGG16萃取特徵，後面的分類處理，都要自己設計。
# weights='imagenet' -> 即使用ImageNet的預先訓練的資料，約100萬張圖片，判斷1000類別的日常事物，例如動物、交通工具...等，我們通常選這一項。
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128,3))

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



# 編譯model
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics=['accuracy'])


# For the training. I try to use 100 epoch before this and i got 85% validation accuracy and 87% training accuracy.
# But to make it shorter i use 50% instead and it should be around 82% - 84% for the validation accuracy.

#Training the model

history = model.fit(train_generator, validation_data = val_generator,
                    epochs = 50,
                    verbose = 1,
                    shuffle = True)

model.evaluate(test_generator)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(50)

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
model.save('VGG16_model.h5')