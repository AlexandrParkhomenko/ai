#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 10:39:25 2021

@author: Alexandr Parkhomenko

1B. Now, let's consider a network with two units (each with three inputs) on the first layer and one unit (with two inputs) on the output layer. All units have ReLU activation. Pick a set of weights and biases for these units that can do the edge detection task or determine if no such set of weights exist. Be prepared to explain your answer.

Enter a list of three lists of weights and bias, first for the two units on the first layer (in the order of [w1,w2,w3,b]) and then for the output unit (in the order of [w1,w2,b]), or 'None' in no such weights and biases exist.

Checking the solution listed as correct.
Solution: [[1, 0, -1, -1], [-1, 0, 1, -1], [1, 1, 0]]
"""

import numpy as np

# OK
image  = np.array([[0,1,1], # edge
                   [1,1,0], # edge
                   # [0,0,0],
                   # [1,1,1],
                   # [1,0,0],
                   # [0,0,1],
                   ])
w1, b1 = np.array([1, 0, -1]), -1
w2, b2 = np.array([-1, 0, 1]), -1
w3, b3 = np.array([1, 1]), 0
z = np.array([0,0,0,0,0,0]) #
for i in range(image.shape[0]):
     y1 = w1@image[i] + b1
     y2 = w2@image[i] + b2
     y3 = w3@(np.array([y1,y2])) + b3
     z[i] = y3
if z[0] == z[1]:  # and z[0] != z[2] and z[2] == z[3]:  # and z[3] == z[4] and z[4] == z[5]:
    print("[",repr(np.append(w1, b1)),",", repr(np.append(w2, b2)),",", repr(np.append(w3, b3)),"]")
    # print(w1, w2, w3, b1, b2, b3)

# NOT OK
image  = np.array([[0,1,1], # edge
                   [1,1,0], # edge
                    [0,0,0],
                    [1,1,1],
                   # [1,0,0],
                   # [0,0,1],
                   ])
w1, b1 = np.array([1, 0, -1]), -1
w2, b2 = np.array([-1, 0, 1]), -1
w3, b3 = np.array([1, 1]), 0
z = np.array([0,0,0,0,0,0]) #
for i in range(image.shape[0]):
     y1 = w1@image[i] + b1
     y2 = w2@image[i] + b2
     y3 = w3@(np.array([y1,y2])) + b3
     z[i] = y3
if z[0] == z[1] and z[0] != z[2] and z[0] != z[3]:  # and z[3] == z[4] and z[4] == z[5]:
    print("[",repr(np.append(w1, b1)),",", repr(np.append(w2, b2)),",", repr(np.append(w3, b3)),"]")
    # print(w1, w2, w3, b1, b2, b3)
else:
    print("Try another solution.")
