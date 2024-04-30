#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
"""
Created on Tue Feb 16 21:41:13 2021

@author: a


transition = np.array([[0.1, 0.9, 0.0],
                       [0.1, 0.1, 0.8],
                       [0.0, 0.0, 1.0]])
reward = np.array([[-3,-3,-3]]).T
Reward = np.array([[ 0, 0,10]]).T

#print(transition@reward)
res = np.array([[0.0,0.0,0.0]]).T
res = res + transition@reward
res += transition@Reward
print("res",res)
res = res + transition@reward
print("res",res)
res -= transition@Reward
print("Res",res)
"""
"""

g = np.array([1,2,3])
b = np.array([4,5,6])

print(g*b)
"""
b = np.array([[0.0, 0.9, 0.1, 0.0],
              [0.9, 0.1, 0.0, 0.0],
              [0.0, 0.0, 0.1, 0.9],
              [0.9, 0.0, 0.0, 0.1]])

c = np.array([[0.0, 0.1, 0.9, 0.0],
              [0.9, 0.1, 0.0, 0.0],
              [0.0, 0.0, 0.1, 0.9],
              [0.9, 0.0, 0.0, 0.1]])

r = np.array([[0],[1],[0],[2]])
d = 0.9

v = np.linalg.solve(d*c-np.eye(4),-r)
print(v)





















