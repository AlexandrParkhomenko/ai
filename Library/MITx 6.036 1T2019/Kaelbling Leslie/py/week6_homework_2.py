
# see https://youtu.be/rkAWbFwa0eI

import numpy as np

# # What probability distribution over the categories is represented by z?
# z = np.array([-1,0,1])
# print(np.exp(z)/np.sum(np.exp(z))) # def SM(z):

def SM(z):
    return np.exp(z)/np.sum(np.exp(z))

def SM_grad(x, y, a):
    return (-(y -a)*x.T).T
# Everything is the same, don't use "for"
# for k in range(2):
#     for j in range(3):
#         print(x[k]*(a[j]-y[j]))

# def nll(a, y):
#     return -np.sum(np.log(a) * y)

# def nll_delta(z, a, y):
#     return a-y



w = np.array([[1, -1, -2], 
              [-1, 2,  1]])
x = np.array([[1], 
              [1]])
y = np.array([[0, 1, 0]]).T

a = w.T@x
a = SM(a)
print("a:",a)
print("∇WL:", SM_grad(x, y, a) )

step = 0.5
#th = w
w = w-step*SM_grad(x, y, a)
print(w)

a = w.T@x
a = SM(a)
print("a:",a)
