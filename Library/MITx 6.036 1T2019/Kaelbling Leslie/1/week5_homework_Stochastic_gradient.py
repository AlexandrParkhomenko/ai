#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 09:47:46 2020

@author:  AlexandrParkhomenko 
"""

# 8) Stochastic gradient

import numpy as np
import code_for_hw5 as hw5

def downwards_line():
    X = np.array([[0.0, 0.1, 0.2, 0.3, 0.42, 0.52, 0.72, 0.78, 0.84, 1.0],
                  [1.0, 1.0, 1.0, 1.0, 1.0,  1.0,  1.0,  1.0,  1.0,  1.0]])
    y = np.array([[0.4, 0.6, 1.2, 0.1, 0.22, -0.6, -1.5, -0.5, -0.5, 0.0]])
    return X, y

X, y = downwards_line()

def compute_loss(theta, X, y):
    return ((np.linalg.norm(np.dot(X,theta) - y))**2)/(2*X.shape[0])

def J(Xi, yi, w):
    # translate from (1-augmented X, y, theta) to (separated X, y, th, th0) format
    return float(hw5.ridge_obj(Xi[:-1,:], yi, w[:-1,:], w[-1:,:], 0))

def dJ(Xi, yi, w):
    def f(w): return J(Xi, yi, w)
    return num_grad(f)(w)

def num_grad(f):
    def df(x):
        g = np.zeros(x.shape)
        delta = 0.001
        for i in range(x.shape[0]):
            xi = x[i,0]
            x[i,0] = xi - delta
            xm = f(x)
            x[i,0] = xi + delta
            xp = f(x)
            x[i,0] = xi
            g[i,0] = (xp - xm)/(2*delta)
        return g
    return df

def package_ans(p):
    w,fs,ws = p
    print("Weights:",w)
    print("fs first and last elements:",fs[0],fs[len(fs)-1])
    print("ws first and last elements:",ws[0],ws[len(ws)-1])
    print("Length of fs:",len(fs))
    print("Length of ws:",len(ws))

def cv(m):
    return hw5.cv(m)

def sgd(X, y, J, dJ, w0, step_size_fn, max_iter):
    ws,fs,t,w,n = [],[],0,w0,y.shape[1]
    while t < max_iter: # 3: #
        t += 1
        w0 = np.copy(w)
        ws.append(w)
        i = np.random.randint(n)
        X_i, y_i = cv(X[:,i]), cv(y[:,i])
        fs.append(J(X_i, y_i, w))
        w = w - step_size_fn(t) * dJ(X_i, y_i, w)  
    return (w0,fs,ws)

ans=package_ans(sgd(X, y, J, dJ, hw5.cv([-0.1, 0.1]), lambda i: 0.01, 1000)) #

# print(dJ(hw5.cv(X[:,0]), hw5.cv(y[:,0]), hw5.cv([-0.1, 0.1])))

# Our solution produced the following value for ans:
# Weights: [[-1.4118594928102899], [0.7705243986321614]]
# fs first and last elements: [0.36, 0.028653685126009333]
# ws first and last elements: [[[0.0], [0.0]], [[-1.4118594928102899], [0.7705243986321614]]]
# Length of fs: 1000
# Length of ws: 1000

# My solution produced the following value for ans:
# Weights: [[-1.091762  ]
#  [ 0.56427998]]
# fs first and last elements: 0.0 1.6338242281820081
# ws first and last elements: [[-0.1]
#  [ 0.1]] [[-1.091762  ]
#  [ 0.56427998]]
# Length of fs: 1000
# Length of ws: 1000

# or

# Weights: [[-1.23712057]
#  [ 0.59448897]]
# fs first and last elements: 0.272484 0.3038063031569558
# ws first and last elements: [[-0.1]
#  [ 0.1]] [[-1.23712057]
#  [ 0.59448897]]
# Length of fs: 1000
# Length of ws: 1000

# or anything else ...





