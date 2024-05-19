import pandas as pd
import numpy as np

import tensorflow as tf
import os

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

base_model = ResNet50(include_top=False,
                      weights=None,
                      input_shape=(224,224,3))

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1, activation='sigmoid', kernel_regularizer=l2(0.001))(x)

model = Model(inputs=base_model.input, outputs=x)

learning_rate = 0.0001
model. compile(optimizer=Adam(learning_rate=learning_rate),
              loss='binary_crossentropy',  # sparse_categotical_crossentropy
              metrics=['accuracy'])

#model.summary()

# base path
base_dir = 'D:/archive/Humans/'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
train_face_dir = os.path.join(train_dir, 'face')
validation_face_dir = os.path.join(validation_dir, 'face')

train_datagen = ImageDataGenerator(rescale= 1./255,
                                   rotation_range=15,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale= 1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=32,
                                                    color_mode='rgb',
                                                    class_mode='binary', #categorical, sparse
                                                    target_size=(224,224))
validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                              batch_size=32,
                                                              color_mode='rgb',
                                                              class_mode='binary', #categorical, sparse
                                                              target_size=(224,224))


model.fit(train_generator,
          validation_data = validation_generator,
          epochs=10)

model.save('face_detection.h5')