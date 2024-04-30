#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 19:02:36 2020

@author: AlexandrParkhomenko 

# 6) Linear regression - going downhill
"""
import numpy as np

X = np.array([[1., 2., 3., 4.], [1., 1., 1., 1.]])
Y = np.array([[1., 2.2, 2.8, 4.1]])
th = np.array([[1.0],[0.05]])
th0 = np.array([[0.]])

# In all the following definitions:
# x is d by n : input data
# y is 1 by n : output regression values
# th is d by 1 : weights
# th0 is 1 by 1 or scalar

def lin_reg(x, th, th0):
    return np.dot(th.T, x) + th0

def square_loss(x, y, th, th0):
    return (y - lin_reg(x, th, th0))**2

def mean_square_loss(x, y, th, th0):
    # the axis=1 and keepdims=True are important when x is a full matrix
    return np.mean(square_loss(x, y, th, th0), axis = 1, keepdims = True)



# Write a function that returns the gradient of lin_reg(x, th, th0)
# with respect to th
def d_lin_reg_th(x, th, th0):
    return x #(np.matmul(th.T,x))
    
# Write a function that returns the gradient of square_loss(x, y, th, th0) with
# respect to th.  It should be a one-line expression that uses lin_reg and
# d_lin_reg_th.
def d_square_loss_th(x, y, th, th0):
    return -2 * (y - lin_reg(x, th, th0)) * d_lin_reg_th(x, th, th0)

# Write a function that returns the gradient of mean_square_loss(x, y, th, th0) with
# respect to th.  It should be a one-line expression that uses d_square_loss_th.
def d_mean_square_loss_th(x, y, th, th0):
    return np.mean(d_square_loss_th(x, y, th, th0), axis = 1, keepdims = True)

ans=d_lin_reg_th(X[:,0:1], th, th0).tolist()
print(ans, np.amin(ans == np.array([[1.0], [1.0]])))
ans=d_lin_reg_th(X, th, th0).tolist()
print(ans, np.amin(ans == X))
ans=d_square_loss_th(X[:,0:1], Y[:,0:1], th, th0).tolist()
print(ans, np.amin(ans == np.array([[0.10000000000000009], [0.10000000000000009]])))
ans=d_square_loss_th(X, Y, th, th0).tolist()
print(ans, np.amin(ans == np.array([[0.10000000000000009, -0.6000000000000014, 1.5, -0.3999999999999986], [0.10000000000000009, -0.3000000000000007, 0.5, -0.09999999999999964]])))
ans=d_mean_square_loss_th(X[:,0:1], Y[:,0:1], th, th0).tolist()
print(ans, np.amin(ans == np.array([[0.10000000000000009], [0.10000000000000009]])))
ans=d_mean_square_loss_th(X, Y, th, th0).tolist()
print(ans, np.amin(ans == np.array([[0.15000000000000002], [0.04999999999999993]])))

def rv(value_list):
    return np.array([value_list])

# Write a procedure that takes a list of numbers and returns a 2D numpy array representing a column vector containing those numbers. You can use the rv procedure.
def cv(value_list):
    return rv(value_list).T

# Write a function that returns the gradient of lin_reg(x, th, th0)
# with respect to th0. Hint: Think carefully about what the dimensions of the returned value should be!
def d_lin_reg_th0(x, th, th0):
    return rv(np.ones(x.shape[1]))
    
# Write a function that returns the gradient of square_loss(x, y, th, th0) with
# respect to th0.  It should be a one-line expression that uses lin_reg and
# d_lin_reg_th0.
def d_square_loss_th0(x, y, th, th0):
    return -2 * (y - lin_reg(x, th, th0)) * d_lin_reg_th0(x, th, th0)

# Write a function that returns the gradient of mean_square_loss(x, y, th, th0) with
# respect to th0.  It should be a one-line expression that uses d_square_loss_th0.
def d_mean_square_loss_th0(x, y, th, th0):
    return np.mean(d_square_loss_th0(x, y, th, th0), axis = 1, keepdims = True)

ans=d_lin_reg_th0(X[:,0:1], th, th0).tolist()
print(ans, np.amin(ans == np.array([[1.0]])))
ans=d_lin_reg_th0(X, th, th0).tolist()
print(ans, np.amin(ans == np.array([[1.0, 1.0, 1.0, 1.0]])))
ans=d_square_loss_th0(X[:,0:1], Y[:,0:1], th, th0).tolist()
print(ans, np.amin(ans == np.array([[0.10000000000000009]])))
ans=d_square_loss_th0(X, Y, th, th0).tolist()
print(ans, np.amin(ans == np.array([[0.10000000000000009, -0.3000000000000007, 0.5, -0.09999999999999964]])))
ans=d_mean_square_loss_th0(X[:,0:1], Y[:,0:1], th, th0).tolist()
print(ans, np.amin(ans == np.array([[0.10000000000000009]])))
ans=d_mean_square_loss_th0(X, Y, th, th0).tolist()
print(ans, np.amin(ans == np.array([[0.04999999999999993]])))

# In all the following definitions:
# x is d by n : input data
# y is 1 by n : output regression values
# th is d by 1 : weights
# th0 is 1 by 1 or scalar
def ridge_obj(x, y, th, th0, lam):
    return np.mean(square_loss(x, y, th, th0), axis = 1, keepdims = True) + lam * np.linalg.norm(th)**2

def d_ridge_obj_th(x, y, th, th0, lam):
    return d_mean_square_loss_th(x, y, th, th0) + 2*lam*th

def d_ridge_obj_th0(x, y, th, th0, lam):
    return d_mean_square_loss_th0(x, y, th, th0)

ans=d_ridge_obj_th(X[:,0:1], Y[:,0:1], th, th0, 0.01).tolist()
print(ans, np.amin(ans == np.array([[0.12000000000000009], [0.10100000000000009]])))
ans=d_ridge_obj_th(X, Y, th, th0, 0.05).tolist()
print(ans, np.amin(ans == np.array([[0.25], [0.05499999999999994]])))
ans=d_ridge_obj_th0(X[:,0:1], Y[:,0:1], th, th0, 0.01).tolist()
print(ans, np.amin(ans == np.array([[0.10000000000000009]])))
ans=d_ridge_obj_th0(X, Y, th, th0, 0.05).tolist()
print(ans, np.amin(ans == np.array([[0.04999999999999993]])))
