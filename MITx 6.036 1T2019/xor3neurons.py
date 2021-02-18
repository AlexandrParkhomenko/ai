#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 21:33:33 2021

@author: a
"""

import numpy as np

# neuron(neuron(-1*x1 +1*x2 -0,5) + neuron(1*x1 -1*x2 -0.5) -0.5)
def a(m):
    return np.where(m>=0,1,0)

x = np.array([[0,0],[0,1],[1,0],[1,1]]).T
y = np.array([[0,1,1,0]]).T
print(a([[1,1]]@x-0.5)) # OR, use tsis
print(a([[1,1]]@x-1.5))

b = a([[-1,1]]@x-0.5)
print(b)
c = a([[1,-1]]@x-0.5)
print(c)

# [[0 1 1 1]]
# [[0 0 0 1]]
# [[0 1 0 0]]
# [[0 0 1 0]]