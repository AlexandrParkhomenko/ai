#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 12:18:06 2021 by Parkhomenko Alexandr
If you are not careful, there are many interesting things awaiting you.

@author: Kaelbling Leslie
"""

# 4B) What is (approximately) the expected loss of the network on 1024×1 images if the convolutional layer is an averaging filter and second layer is the sum function (without a bias term)? (Note that you can answer the question theoretically or through coding, depending on your preference.) 

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Conv1D, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras import backend as K

def generate_1d_images(nsamples,image_size,prob):
    Xs=[]
    Ys=[]
    for i in range(0,nsamples):
        X=np.random.binomial(1, prob, size=image_size)
        Y=count_objects_1d(X)
        Xs.append(X)
        Ys.append(Y)
    Xs=np.array(Xs)
    Ys=np.array(Ys)
    return Xs,Ys


#count the number of objects in a 1d array
def count_objects_1d(array):
    count=0
    for i in range(len(array)):
        num=array[i]
        if num==0:
            if i==0 or array[i-1]==1:
                count+=1
    return count

def l1_reg(weight_matrix):
    return 0.01 * K.sum(K.abs(weight_matrix))    

def get_image_data_1d(tsize,image_size,prob):
    #prob controls the density of white pixels
    #tsize is the size of the training and test sets
    vsize=int(0.2*tsize)
    X_train,Y_train=generate_1d_images(tsize,image_size,prob)
    X_val,Y_val=generate_1d_images(vsize,image_size,prob)
    X_test,Y_test=generate_1d_images(tsize,image_size,prob)
    #reshape the input data for the convolutional layer
    X_train=np.expand_dims(X_train,axis=2)
    X_val=np.expand_dims(X_val,axis=2)
    X_test=np.expand_dims(X_test,axis=2)
    data=(X_train,Y_train,X_val,Y_val,X_test,Y_test)
    return data

tsize=1000
imsize=1024
kernel_s=2
stride=1
batch=1
data=get_image_data_1d(tsize,imsize,0.1)
(X_train,Y_train,X_val,Y_val,X_test,Y_test)=data
layer1=Conv1D(filters=1, kernel_size=kernel_s, strides=stride,use_bias=False,activation='relu',batch_size=batch,input_shape=(imsize,1),padding='same')
layer3=Dense(units=1, activation='linear',use_bias=False) # here
layers=[layer1,Flatten(),layer3]
model=Sequential()
for layer in layers:
    model.add(layer)
model.compile(loss='mse', optimizer=Adam())    
model.layers[0].set_weights([np.array([1/2,1/2]).reshape(2,1,1)]) # here
model.layers[-1].set_weights([np.ones(imsize).reshape(imsize,1)])
model.evaluate(X_test,Y_test)

# 32/32 [==============================] - 0s 2ms/step - loss: 101.6523

tsize=1000
imsize=1024
kernel_s=2
stride=1
batch=1
data=get_image_data_1d(tsize,imsize,0.1)
(X_train,Y_train,X_val,Y_val,X_test,Y_test)=data
layer1=Conv1D(filters=1, kernel_size=kernel_s, strides=stride,use_bias=False,activation='relu',batch_size=batch,input_shape=(imsize,1),padding='same')
layer3=Dense(units=1, activation='linear',use_bias=True) # here
layers=[layer1,Flatten(),layer3]
model=Sequential()
for layer in layers:
    model.add(layer)
model.compile(loss='mse', optimizer=Adam())    
model.layers[0].set_weights([np.array([1/2,1/2]).reshape(2,1,1)])
model.layers[-1].set_weights(np.array([np.ones(imsize).reshape(imsize,1),
                                       np.array([-10])])) # here
model.evaluate(X_test,Y_test)

# 32/32 [==============================] - 0s 2ms/step - loss: 12.7647
