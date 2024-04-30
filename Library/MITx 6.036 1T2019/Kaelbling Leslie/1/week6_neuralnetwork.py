#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 14:00:09 2020

@author: a
"""

"""
We implement a neural network with L layers.

sizes = a list of L layer sizes, starting with the size of the input layer
act_funs =  a list of L−1 activation functions, one for each layer beyond the input. 
act_deriv_funs = a list of L-2 derivatives for the layer activation functions, one for each layer beyond the input but not including the final layer.
loss_fun = a loss function for the network, which takes the final layer activations and the target
loss_delta_fun = a derivative function for the loss, which takes the final layer weighted inputs, activations and the target
class_fun = given the last activation layer returns a class index, useful for classification.

weights =  a list of L−1 n×m matrices where m is the size of layer i−1 and n is the size of layer i.
biases = a list of L−1 column vectors of size n×1, where n is the size of layer i, one for each layer beyond the input.

"""
import numpy as np

class NN:
    sizes = []
    act_funs = []
    act_deriv_funs = []
    loss_fun = None
    loss_delta_fun = None
    class_fun = None
    def evaluate(self, X, Y):
        count = 0                       # number of errors
        loss = 0                        # cumulative loss
        d, n = X.shape                  # d = # features, n = # points
        o, _ = Y.shape                  # o = # outputs, n = # points
        for i in range(n):
            zs, activations = self.forward(X[:,i:i+1]) # compute activations
            act_L = activations[-1]                    # last layer
            pj = self.class_fun(act_L)                 # predict a class
            y = Y[:,i:i+1]                             # ith target
            yj = self.class_fun(y)                     # predict target
            if pj != yj:
                count += 1              # increment # wrong
            loss += self.loss_fun(act_L, y) # increment loss
        # pct error, average loss
        return count/float(n), loss/float(n)
    
    def initialize(self):
      # initialize the biases to zero, no bias on input layer
      # initialize the weights to normal with variance 1/m (deviation 1/sqrt(m)) where m is # inputs
      # return network to enable checking
      biases = []
      print(self.sizes)
      for i in range(1, len(self.sizes)):
          biases.append(np.zeros(self.sizes[i]).reshape(-1, 1))

      weights = []

      for i in range(1, len(self.sizes)):
          entry  = np.random.normal(loc=0, scale=1, size =(self.sizes[i], self.sizes[i-1]))

          weights.append(entry)

      self.weights = weights
      self.biases = biases

      return self


def forward(self, x):    
    #x is a column vector of inputs, the activations of the input layer
    #It returns a tuple of two lists:
        #zs a list of column vectors of weighted inputs for each layer beyond the input
        #activations a list of column vectors of activations for each layer, including the input as the first entry
    zs = []
    input = x
    activations = [x]
    
    for i in range(len(self.weights)):
        entry = np.dot(self.weights[i], input) + self.biases[i]
        input = entry
        act_layer = self.act_funs[i]
        input = act_layer(input)
        
        zs.append(entry)
        activations.append(input)

    
    return (np.array(zs), np.array(activations))

"""
    backward takes an input vector (xx) and a target activation (yy) and returns the gradient of the loss wrt weights and the gradient of the loss wrt biases. We will use this function to compute the gradient for a step of SGD.

        Specifically:

        x is a column vector of inputs, the activations of the input layer
        y is a column vector of targets, the desired activations of the output layer
        It returns a tuple of two matrices:

        grad_w is a list of matrices, each is the gradient of the loss wrt weights for a layer, for the input point
        grad_b is a list of vectors, each is the gradient of the loss wrt biases for a layer, for the input point
"""
def backward(self, x, y):

    grad_w = []
    grad_b = []
    
    zs, activations = self.forward(x)
    delta_L = self.loss_delta_fun(zs[-1], activations[-1], y)
    
    dC_dw = np.dot(delta_L, activations[-2].T)
    dC_db = delta_L
    
    grad_w.append(dC_dw)
    grad_b.append(dC_db)
    
    for i in range(2, len(self.sizes)):
        deriv = self.act_deriv_funs[-i+1]
        
        new_delta = np.dot(self.weights[-i+1].T, delta_L) * deriv(zs[-i])
        
        dC_dw = np.dot(new_delta, activations[-i-1].T)
        dC_db = new_delta
        
        
        grad_w.append(dC_dw)
        grad_b.append(dC_db)
        
        delta_L = new_delta
    
    grad_w.reverse()
    grad_b.reverse()
    return (grad_w, grad_b)

"""
sgd_train is an application of the stochastic gradient descent approach, using the backward function to compute the gradient. 
The method takes a dataset as input and specification of the step size and number of iterations.

Specifically:

X is a dxn data matrix
Y is a oxn matrix of targets, where o is the size of the output activation layer
"""

def sgd_train(self, X, Y, n_iter, step_size):
    self.initialize()
    
    d, n = X.shape
    # num_layers = len(self.sizes) # newer used
    
    for i in range(n_iter):
        ind = np.random.randint(n)
        x_ind = X[:, ind:ind+1]
        y_ind = Y[:, ind:ind+1]
        
        grad_w, grad_b = self.backward(x_ind,y_ind)
        
        print(grad_w)
        print(grad_b)
        
        
        for j in range(len(self.weights)):
            w = grad_w[j]*step_size
            self.weights[j] -=  w
            
            b = grad_b[j] * step_size
            self.biases[j] -= b

    return self
 
def classify(X, Y, hidden=[10, 10], it=10000, lr=0.005):
    D = X.shape[0]
    N = X.shape[1]
    O = Y.shape[0]
    # Create the network
    nn = NN()
    nn.sizes = [D] + list(hidden) + [O]
    nn.act_funs = [relu for l in list(hidden)] + [softmax]
    nn.act_deriv_funs = [relu_deriv for l in list(hidden)]
    nn.loss_fun = nll
    nn.loss_delta_fun = nll_delta
    nn.class_fun = softmax_class     # index of class
    # Modifies the weights and biases
    nn.sgd_train(X, Y, it, lr)
    return nn
  
def f(z):
    return z
def f_deriv(z):
    return 1.0
def relu(z):
    #print(np.maximum(z, np.zeros(z.shape)))
    return np.maximum(z, np.zeros(z.shape))
def relu_deriv(z):
    return np.where(z<0, 0, 1)
def softmax(z):
    return np.exp(z)/np.sum(np.exp(z))
def softmax_class(a):
    return np.argmax(a)

def hinge(a, y):
    v = y*a
    return np.where(v < 1, 1-v, 0)

def nll(a, y):
    return -np.sum(np.log(a) * y)
def nll_delta(z, a, y):
    return a-y
    
nn = NN()
nn.sizes = 1
nn.act_funs = f
nn.act_deriv_funs = f_deriv
nn.loss_fun = hinge
# nn.loss_delta_fun = 
# nn.class_fun = None
