#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 11:42:50 2021

@author: https://github.com/AlexandrParkhomenko
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

lang = 'en'

inputs = keras.Input(shape=(784,), name='digits')
x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
x = layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = layers.Dense(10, activation='softmax', name='predictions')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
# Load the tutorial dataset for this example
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocessing data (these are Numpy arrays)
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

# Reserve 10,000 examples for validation
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

# We indicate the training configuration (optimizer, loss function, metrics)
model.compile(optimizer=keras.optimizers.RMSprop(),  # Optimizer
              # Loss function to be minimized
              loss=keras.losses.SparseCategoricalCrossentropy(),
              # List of metrics to monitor
              metrics=[keras.metrics.SparseCategoricalAccuracy()])

# Let's train the model by splitting the data into "packages"
# of size "batch_size", and sequentially iterating the entire dataset 
# for a specified number of "epochs"
if lang== 'ru': print('# Обучаем модель на тестовых данных')
if lang== 'en': print('# Train the model on test data')
history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=3,
                    # We transfer validation data to monitor losses 
                    # and metrics on this data at the end of each epoch
                    validation_data=(x_val, y_val))

# The returned "history" object contains records of losses 
# and metrics during training
print('\nhistory dict:', history.history)

# Оценим модель на тестовых данных, используя "evaluate"
if lang== 'ru': print('\n# Оцениваем на тестовых данных')
if lang== 'en': print('\n# Evaluating on test data')
results = model.evaluate(x_test, y_test, batch_size=128)
print('test loss, test acc:', results)

# Let's generate predictions (probabilities - the output of the last layer) 
# on the new data using "predict"
if lang== 'ru': print('\n# Генерируем прогнозы для 3 образцов')
if lang== 'en': print('\n# Generating forecasts for 3 samples')
predictions = model.predict(x_test[:3])
if lang== 'ru': print('размерности прогнозов:', predictions.shape)
if lang== 'en': print('dimensions of predictions:', predictions.shape)
