#!/usr/bin/env python
# coding: utf-8

# In[25]:


import tensorflow.keras
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adagrad


# In[26]:


# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
keras_model = tensorflow.keras.models.load_model('VGG16_model.h5', compile=False)
keras_model._name = 'model1'
keras_model2 = tensorflow.keras.models.load_model('SimpleCNN_model.h5', compile=False)
keras_model2._name = 'model2'
keras_model3 = tensorflow.keras.models.load_model('VGG19_model.h5', compile=False)
keras_model3._name = 'model3'
models = [keras_model, keras_model2, keras_model3]
model_input = tf.keras.Input(shape=(128, 128, 3))
# model_input = tf.keras.Input(shape=(224, 224, 3))
model_outputs = [model(model_input) for model in models]
ensemble_output = tf.keras.layers.Average()(model_outputs)
ensemble_model = tf.keras.Model(inputs=model_input, outputs=ensemble_output)  

ensemble_model.compile( loss = 'categorical_crossentropy',
                        optimizer = 'Adagrad',
                        metrics=['accuracy'])

ensemble_model.save('ensmeble.h5')


# In[27]:


class config:
    epoch = 30
    model_input = tf.keras.Input(shape=(128, 128, 3))

    # For the parameter in the flow from directory
    # 把每樣不同大小的資料伸縮變成 128x128，batch_size = 32，可以嘗試其他看看size
    batch_size = 32
    img_width = 128
    img_height = 128

    # The augmetnation parameter (伸縮參數)
    rotation_range = 0.4
    width_shift_range = 0.2
    height_shift_range = 0.3
    shear_range = 0.2
    zoom_range = 0.2

    # 產生Image Data Generator
    # 資料路徑
    path_train = "./output/train"
    path_test = "./output/test"
    path_val = "./output/val"


# In[28]:


# 用 ImageDataGenerator 來調整照片
train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range = config.rotation_range,
                                   width_shift_range = config.width_shift_range,
                                   height_shift_range = config.height_shift_range,
                                   shear_range = config.shear_range,
                                   zoom_range = config.zoom_range,
                                   horizontal_flip = True,
                                   fill_mode = 'nearest'
                                  )

train_generator = train_datagen.flow_from_directory(directory = config.path_train,
                                                    batch_size = config.batch_size,
                                                    class_mode = "categorical",
                                                    target_size = (config.img_width, config.img_height)
                                                    )

test_datagen = ImageDataGenerator(rescale = 1./255.)

test_generator = test_datagen.flow_from_directory(directory = config.path_test,
                                                  batch_size = config.batch_size,
                                                  class_mode = "categorical",
                                                  target_size = (config.img_width, config.img_height)
                                                  )

valid_datagen = ImageDataGenerator(rescale = 1./255.)

valid_generator = test_datagen.flow_from_directory(directory = config.path_val,
                                                  batch_size = config.batch_size,
                                                  class_mode = "categorical",
                                                  target_size = (config.img_width, config.img_height)
                                                  )


# In[29]:


from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, Dense

base_model = load_model('ensmeble.h5')

print(base_model.summary())

base_model.trainable = False

history = base_model.fit(   train_generator,
                            validation_data = test_generator,
                            epochs = config.epoch, 
                            steps_per_epoch = config.batch_size,
                            verbose = 1,
                            shuffle = True)


# In[30]:


import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(config.epoch)

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


# In[31]:


_, acc = base_model.evaluate(test_generator, steps=len(test_generator), verbose=0)
print('Accuracy: %.3f' % (acc * 100.0))

