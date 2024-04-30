#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 10:18:41 2021

@author: a
"""

import numpy as np
import math as m


def super_simple_separable():
    X = np.array([[2, 3, 9, 12],
                  [5, 2, 6, 5]])
    y = np.array([[1, 0, 1, 0]])
    return X, for_softmax(y)


def for_softmax(y):
    return np.vstack([1-y, y])

def mini_gd(self, X, Y, iters, lrate, notif_each=None, K=10):
    pass

X, Y = super_simple_separable()
D, N = X.shape
np.random.seed(0)
num_updates = 0
indices = np.arange(N)

np.random.shuffle(indices)
print(indices)
