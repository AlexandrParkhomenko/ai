# week3_homework 3) Polynomial Features

import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import colors
# import pdb
import itertools
import operator
import functools

# Takes a list of numbers and returns a column vector:  n x 1
def cv(value_list):
    return np.transpose(rv(value_list))

# Takes a list of numbers and returns a row vector: 1 x n
def rv(value_list):
    return np.array([value_list])

def mul(seq):
    return functools.reduce(operator.mul, seq, 1)

def make_polynomial_feature_fun(order):
    # raw_features is d by n
    # return is k by n where k = sum_{i = 0}^order  multichoose(d, i)
    def f(raw_features):
        d, n = raw_features.shape
        result = []   # list of column vectors
        for j in range(n):
            features = []
            for o in range(order+1):
                indexTuples = \
                          itertools.combinations_with_replacement(range(d), o)
                for it in indexTuples:
                    features.append(mul(raw_features[i, j] for i in it))
            result.append(cv(features))
        return np.hstack(result)
    return f

degrees = np.array([1, 10,  20,  30,  40,   50])
for i in degrees:
    print(i, make_polynomial_feature_fun(i)(cv(np.ones(2))).shape[0] ) 

# [1, 10,  20,  30,  40,   50] 
# [3, 66, 231, 496, 861, 1326]
