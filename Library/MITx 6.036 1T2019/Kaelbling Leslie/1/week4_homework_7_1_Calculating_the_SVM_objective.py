"""
Created on Mon Nov 16 11:19:50 2020
@author:  AlexandrParkhomenko 
# 7.1) Calculating the SVM objective
see https://openlearninglibrary.mit.edu/courses/course-v1:MITx+6.036+1T2019/courseware/Week4/week4_homework/?child=first
"""

import numpy as np

#print( np.amax(np.array([[0],[1]])) )

def super_simple_separable():
    X = np.array([[2, 3, 9, 12],
                  [5, 2, 6, 5]])
    y = np.array([[1, -1, 1, -1]])
    return X, y

sep_e_separator = np.array([[-0.40338351], [1.1849563]]), np.array([[-2.26910091]])

#Your Code Here
def hinge(v):
    # print("v",v)
    # print("where",np.where(v < 1, 1-v, 0))
    return np.where(v < 1, 1-v, 0)

# x is dxn, y is 1xn, th is dx1, th0 is 1x1
def hinge_loss(x, y, th, th0):
    return hinge(y*(np.dot(th.T,x)+th0))

# x is dxn, y is 1xn, th is dx1, th0 is 1x1, lam is a scalar
def svm_obj(x, y, th, th0, lam):
    # print("in", x, y, th, th0, lam)
    a = hinge_loss(x, y, th, th0)
    # print("a", a)
    b = np.sum(a)/a.shape[1]
    # print("b", b)
    d = np.dot(th.T,th) #**0.5
    # print("d", d)
    return  b + lam*d

# Test case 1
x_1,y_1=super_simple_separable()
th1,th1_0=sep_e_separator
ans=svm_obj(x_1, y_1, th1, th1_0, .1)
print(ans)

# Test case 2
ans = svm_obj(x_1, y_1, th1, th1_0, 0.0)
print(ans)














