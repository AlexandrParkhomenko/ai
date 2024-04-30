#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 13:10:35 2020
@creator: AlexandrParkhomenko

7.2) Calculating the SVM gradient
WITH SOME PROBLEMS
"""

import numpy as np

# Returns the gradient of hinge(v) with respect to v.
def d_hinge(v):
    # print(v)
    return np.where(v > 0, 0, -1) # np.sign(v) #

# Returns the gradient of hinge_loss(x, y, th, th0) with respect to th
def d_hinge_loss_th(x, y, th, th0):
    #print("x:",x)
    a = d_hinge(y*(np.dot(th.T,x)+th0))
    b = np.copy(x)
    return -a*b
#    return d_hinge( y*(np.dot(th.T,x)+th0)/np.abs((np.dot(th.T,th)**0.5)) )

# Returns the gradient of hinge_loss(x, y, th, th0) with respect to th0
def d_hinge_loss_th0(x, y, th, th0):
    return -d_hinge(y*(np.dot(th.T,x)+th0))

# ATTENTION! The functions below are incorrect

# Returns the gradient of svm_obj(x, y, th, th0) with respect to th
def d_svm_obj_th(x, y, th, th0, lam):
    d = np.abs((np.dot(th.T,th)**0.5))
    h = d_hinge_loss_th(x, y, th, th0)
    print("h.shape", h.shape)
    print("h", h)
    # return np.sum(h)/h.shape[1] + lam*d
    return h

# Returns the gradient of svm_obj(x, y, th, th0) with respect to th0
def d_svm_obj_th0(x, y, th, th0, lam):
    d = np.abs((np.dot(th.T,th)**0.5))
    h = d_hinge_loss_th0(x, y, th, th0)
    print("h.shape", h.shape)
    return np.sum(h)/h.shape[1]

# Returns the full gradient as a single vector (which includes both th, th0)
def svm_obj_grad(X, y, th, th0, lam):
    return d_svm_obj_th(X, y, th, th0, lam) + d_svm_obj_th0(X, y, th, th0, lam)


X1 = np.array([[1, 2, 3, 9, 10]])
y1 = np.array([[1, 1, 1, -1, -1]])
th1, th10 = np.array([[-0.31202807]]), np.array([[1.834     ]])
X2 = np.array([[2, 3, 9, 12],
               [5, 2, 6, 5]])
y2 = np.array([[1, -1, 1, -1]])
th2, th20=np.array([[ -3.,  15.]]).T, np.array([[ 2.]])

ans = d_hinge(np.array([[ 71.]])).tolist()
print("d_hinge [[0]]:", ans, ans == np.array([[0]]))
ans = d_hinge(np.array([[ -23.]])).tolist()
print("d_hinge [[-1]]:", ans, ans == np.array([[-1]]))
ans = d_hinge(np.array([[ 71, -23.]])).tolist()
print("d_hinge [[0, -1]]:", ans, ans == np.array([[0, -1]]))

ans = d_hinge_loss_th(X2[:,0:1], y2[:,0:1], th2, th20).tolist()
print("d_hinge_loss_th [[0], [0]]:", ans, ans == np.array([[0], [0]]))
ans = d_hinge_loss_th(X2, y2, th2, th20).tolist()
print("d_hinge_loss_th [[0, 3, 0, 12], [0, 2, 0, 5]]:", ans, ans == np.array([[0, 3, 0, 12], [0, 2, 0, 5]]))
ans = d_hinge_loss_th0(X2[:,0:1], y2[:,0:1], th2, th20).tolist()
print("d_hinge_loss_th0 [[0]]:", ans, ans == np.array([[0]]))
ans = d_hinge_loss_th0(X2, y2, th2, th20).tolist()
print("d_hinge_loss_th0 [[0, 1, 0, 1]]:", ans, ans == np.array([[0, 1, 0, 1]]))

ans = d_svm_obj_th(X2[:,0:1], y2[:,0:1], th2, th20, 0.01).tolist()
print("d_svm_obj_th [[-0.06], [0.3]]:", ans, ans == np.array([[-0.06], [0.3]]))
ans = d_svm_obj_th(X2, y2, th2, th20, 0.01).tolist()
print("d_svm_obj_th [[3.69], [2.05]]:", ans, ans == np.array([[3.69], [2.05]]))
ans = d_svm_obj_th0(X2[:,0:1], y2[:,0:1], th2, th20, 0.01).tolist()
print("d_svm_obj_th0 [[0.0]]:", ans, ans == np.array([[0.0]]))
ans = d_svm_obj_th0(X2, y2, th2, th20, 0.01).tolist()
print("d_svm_obj_th0 [[0.5]]:", ans, ans == np.array([[0.5]]))

ans = svm_obj_grad(X2, y2, th2, th20, 0.01).tolist()
print("svm_obj_grad [[3.69], [2.05], [0.5]]:", ans, ans == np.array([[3.69], [2.05], [0.5]]))
ans = svm_obj_grad(X2[:,0:1], y2[:,0:1], th2, th20, 0.01).tolist()
print("svm_obj_grad [[-0.06], [0.3], [0.0]]:", ans, ans == np.array([[-0.06], [0.3], [0.0]]))
