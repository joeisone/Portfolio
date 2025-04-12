# Practise
import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

# load data pipeline

data=tf.keras.utils.image_dataset_from_directory(r'C:\Users\joseph.boateng\PycharmProjects\pythonProject5-DL\Capstone\Data')
# converting data into numpy array

data_iterator=data.as_numpy_iterator()

# get another batch from iterator
batch=data_iterator.next()
print(batch)

print(batch[0].shape)
print(batch[0].max())
print(batch[0].min())
print(batch[1].shape)

# scaling data
scaled=batch[0]/255
print(scaled.max())
print(scaled.min())
# x represents the independent variable (images)
# y represents the dependent variable (label)

# scaling data through the pipeline
data=data.map(lambda x,y: (x/255,y))
scaled_iterator=data.as_numpy_iterator()
batch=scaled_iterator.next()
print(batch[0].min())
print(batch[0].max())

print(len(data))

# splitting data

train_size=int(len(data)*0.7)
val_size=int(len(data)*0.2)+1
test_size=int(len(data)*0.1)+1


# Establishment of training
# validation partitions
train=data.take(train_size)
val=data.take(val_size)
test=data.take(test_size)

print(len(train))
print(len(test))


# Build  model

Input=keras.layers.Input(shape=(256,256,3))
# Convolution layer
x=keras.layers.Conv2D(32,(3,3),activation='relu')(Input)
x=keras.layers.MaxPooling2D((2,2))(x)
x=keras.layers.Conv2D(64,(3,3),activation='relu')(x)
x=keras.layers.MaxPooling2D((2,2))(x)
x=keras.layers.Conv2D(128,(3,3),activation='relu')(x)
x=keras.layers.MaxPooling2D((2,2))(x)

# Flatten and Dense
x=keras.layers.Flatten()(x)
x=keras.layers.Dense(256,activation='relu')(x)
output=keras.layers.Dense(4,activation='softmax')(x)

# model

model=keras.Model(inputs=Input,outputs=output)

model.compile(
    optimizer='Adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

CNN=model.fit(train,epochs=3,validation_data=val)

model.save(os.path.join('models','DocClassifier.h5'))

# Data preprocessing

# Test model
new_model=keras.models.load_model(os.path.join('MODEL','DocClassifier.h5'))




# Functional model
# convolution neural network

# input layer shape

# maxpooling2d,conv2D,Flatten,Dense

# model

# train model

# evaluate model

# output with if statement



