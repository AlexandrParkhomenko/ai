#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 22:03:13 2021

@author: Alexandr Parkhomenko

This is reasoning. This is not a solution!
"""

import numpy as np

b = -1
X = np.array([[ 1, 1, 0, 0, 0],
              [ 0, 0, 0, 1, 1]])
for x in X:
    # image size 3
    # x[0] and x[4] is padding
    print("image:",x)
    # bruteforce
    for a in range(-1,2):
        for b in range(-1,2):
            for c in range(-1,2):
                f = np.array([a,b,c])
                z = np.array([0,0,0])
                for i in range(x.shape[0]-f.shape[0]+1):
                    # z without padding
                    z[i] = x[i]*f[0] + x[i+1]*f[1] + x[i+2]*f[2]
                if z[0]<z[1] and z[1]>z[2]:
                    print("rezult:",z,"filter:",f)

# WOW rezult: [0 1 0] filter: [ 1 -1  1]  # padding independent

# image: [1 1 0 0 0]
# rezult: [0 1 0] filter: [ 1 -1 -1]
# rezult: [0 1 0] filter: [ 1 -1  0]
# rezult: [0 1 0] filter: [ 1 -1  1]
# image: [0 0 0 1 1]
# rezult: [0 1 0] filter: [-1 -1  1]
# rezult: [0 1 0] filter: [ 0 -1  1]
# rezult: [0 1 0] filter: [ 1 -1  1]

# image: [0 1 0 0 0]
# rezult: [-1  1  0] filter: [ 1 -1 -1]
# rezult: [-1  1  0] filter: [ 1 -1  0]
# rezult: [-1  1  0] filter: [ 1 -1  1]
# rezult: [0 1 0] filter: [ 1  0 -1]
# rezult: [0 1 0] filter: [1 0 0]
# rezult: [0 1 0] filter: [1 0 1]
# image: [0 0 0 1 0]
# rezult: [ 0  1 -1] filter: [-1 -1  1]
# rezult: [0 1 0] filter: [-1  0  1]
# rezult: [ 0  1 -1] filter: [ 0 -1  1]
# rezult: [0 1 0] filter: [0 0 1]
# rezult: [ 0  1 -1] filter: [ 1 -1  1]
# rezult: [0 1 0] filter: [1 0 1]
