#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 20:42:00 2020

@author: a
"""

# Training
# Simple in form, but not in content

import numpy as np

def hinge(v):
    # print("v",v)
    # print("where",np.where(v < 1, 1-v, 0))
    return np.where(v < 1, 1-v, 0)

# x is dxn, y is 1xn, th is dx1, th0 is 1x1
def hinge_loss(x, y, th, th0):
    # return hinge( (y*(np.dot(th.T,x)+th0)) / (1/np.linalg.norm(th.T)) )
    return hinge( (y*(np.dot(th.T,x)+th0)) )

def hinge_loss_grad(x, y, a):
    # https://stats.stackexchange.com/questions/4608/gradient-of-hinge-loss
    return np.where(y*a < 1, -y*x, 0)

step = 0.5
x = np.array([1,1,2])
y = np.array([-1])
th = np.array([1,1,1])

th = th-step*hinge_loss_grad(x, y, np.dot(th.T,x))
print(th)
th = th-step*hinge_loss_grad(x, y, np.dot(th.T,x))
print(th)
th = th-step*hinge_loss_grad(x, y, np.dot(th.T,x))
print(th)
