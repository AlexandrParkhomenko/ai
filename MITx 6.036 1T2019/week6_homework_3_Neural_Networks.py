#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 20:30:54 2020

@author: AlexandrParkhomenko
"""

"""**Problem 3**"""
# 3) Neural Networks

import numpy as np

def SM(z):
    return np.exp(z)/np.sum(np.exp(z))

def relu(z):
    return np.maximum(z, np.zeros(z.shape))

# layer 1, 4 units, weights
w_1 = np.array([[1, 0, -1, 0], [0, 1, 0, -1]])
w_1_bias = np.array([[-1, -1, -1, -1]]).T
# layer 2, 2 units, weights
w_2 = np.array([[1, -1], [1, -1], [1, -1], [1, -1]])
w_2_bias = np.array([[0, 2]]).T

x = np.array([[3], [14]])

z_1 = w_1.T@x+w_1_bias
a_1 = relu(z_1)
print("a_1", a_1)

z_2 = w_2.T@a_1+w_2_bias
print("z_2", z_2)
a_2 = SM(z_2)
print("a_2", a_2)

# a_1 [[ 2.]
#  [13.]
#  [ 0.]
#  [ 0.]]
# z_2 [[ 15.]
#  [-13.]]
# a_2 [[1.00000000e+00]
#  [6.91440011e-13]]


