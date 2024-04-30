#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 12:38:56 2020
@author:  Alexandr Parkhomenko 
Ex2.2: What is the gradient contribution from each point to the parameters of the blue (lower) line

Hint: Please do not make assumptions more difficult than necessary.
"""

import numpy as np

θ0 = 0
θ = np.array([[1]])
x = np.array([[1,1,3,3]])
y = np.array([[3,1,2,6]])

z = np.array([ -2*(y -np.dot(θ.T,x) -θ0)*x, -2*(y -np.dot(θ.T,x) -θ0)])
print("z:\n",z.T)

θ0 = 1

z = np.array([ -2*(y -np.dot(θ.T,x) -θ0)*x, -2*(y -np.dot(θ.T,x) -θ0)])
print("z:\n",z.T)

# z:
#  [[[ -4  -4]]

#  [[  0   0]]

#  [[  6   2]]

#  [[-18  -6]]]
# z:
#  [[[ -2  -2]]

#  [[  2   2]]

#  [[ 12   4]]

#  [[-12  -4]]]
