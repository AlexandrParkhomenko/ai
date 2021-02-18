#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
   s_0 = 0
   f(s, x_i) = max(s, x_i)
   g(s) = s * 2
   Input: [0, 1, 2, 1]
s = 0
def f(x):
    global s 
    s = max(s, x)
    return s

def g(s):
    return s * 2

Input = [0, 1, 2, 1]
y     = [0,0,0,0]

for i in range(Input.__len__()):
    y[i] = g(f(Input[i]))
print(y)
"""

"""
   s_0 = (0, 0)
   f(s, x_i) = (s[0] + x_i, s[1] + 1)
   g(s) = s[0] / s[1]
   Input: [0, 1, 2, 1]
s = [0, 0]
def f(x):
    global s 
    s = [s[0] + x, s[1] + 1]
    return s

def g(s):
    assert s[1]!=0
    return s[0] / s[1]

Input = [0, 1, 2, 1]
y     = [0,0,0,0]

for i in range(Input.__len__()):
    y[i] = g(f(Input[i]))
print(y)
"""

"""
"""
import numpy as np
states = [0, 1, 2, 3]
Tb = np.array([[0.0, 0.9, 0.1, 0.0],
               [0.9, 0.1, 0.0, 0.0],
               [0.0, 0.0, 0.1, 0.9],
               [0.9, 0.0, 0.0, 0.1]
               ])
Tc = np.array([[0.0, 0.1, 0.9, 0.0],
               [0.9, 0.1, 0.0, 0.0],
               [0.0, 0.0, 0.1, 0.9],
               [0.9, 0.0, 0.0, 0.1]
               ])
#s = [0, 0]
def R(s,a):
    #global s 
    r = 0
    if s == 1: r = 1
    if s == 3: r = 2
    return r

def T(s,a,s_):
    i = s
    j = s_
    return Tc[i,j]

# print(T([0,1],0))
#y     = [0, 1, 0, 2] #
Q = [0,1,0,2]
Q_ = [0,1,0,2]

s=np.array([0,1,2,3])
s_=np.array([0,1,2,3])

for i in range(states.__len__()):
    Z = 0
    for n in range(s_.shape[0]):
        Z += T(s_[i],1,s[n])*R(s_[i],1) #max(Q_)
        print(i,s_[i],s[n],T(s_[i],1,s[n]))
    Q[i] = R(i,0) + Z #T(0,1,0) #
print(Q)
