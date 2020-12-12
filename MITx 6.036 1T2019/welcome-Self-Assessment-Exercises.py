# simple routines

# non-exercise function 
def mul_two_lists(a, b):
    return [x*y for x,y in zip(a,b)]

# Given two lists of numbers, write a procedure that returns a list of
# the element-wise sum of the number in those two lists. In the
# following, no imports should be used.
def add_two_lists(a, b):
    return [x+y for x,y in zip(a,b)]

# Given two column vectors (each represented as a list of numbers),
# write a procedure dot that returns the (scalar) dot product of two input
# vectors, each represented as a list of numbers.
def dot(v1, v2):
    return sum(x*y for x,y in zip(v1,v2))

# dot([1,2,3], [4,5,6])
# 32

# Write a function add_n that takes a single numeric argument n, and returns
# a function. The returned function should take a vector v as an argument and 
# return a new vector with the value for n added to each element of vector v.
# For example, add_n(10)([1, 5, 3]) should return [11, 15, 13].
def add_n(n):
    def fun(v):
        return [n+y for y in v]
    return fun

add_n(10)([1,5,3])

# matrix multiplication without numpy
M1 = [[1, 2, 3], [-2, 3, 7]]
M2 = [[1,0,0],[0,1,0],[0,0,1]]

def array_mult(M1, M2):
    l = len(M1)
    m = len(M1[0])
    n = len(M2[0])
    assert len(M1[0]) == len(M2)
    o = [ [ None for y in range( n ) ] 
                 for x in range( l ) ] 
    
    for i in range(l):
        #print(o)
        for j in range(n):
            #print(j)
            s = 0
            for r in range(m):
                s += M1[i][r]*M2[r][j]
            o[i][j] = s
    return o

print(array_mult(M1, M2))

# Transpose list of lists
list(map(list, zip(*l)))



import numpy as np
# Write a procedure that takes a list of numbers and returns a 2D numpy array representing a row vector containing those numbers.
def rv(value_list):
    return np.array([value_list])

# Write a procedure that takes a list of numbers and returns a 2D numpy array representing a column vector containing those numbers. You can use the rv procedure.
def cv(value_list):
    return rv(value_list).T

# Write a procedure that takes a column vector and returns the vector's Euclidean length (or equivalently, its magnitude) as a scalar. You may not use np.linalg.norm, and you may not use a loop.
def length(col_v):
    return np.asscalar(np.dot(col_v.T,col_v)**0.5)
    
# Write a procedure that takes a 2D array and returns the final column as a two dimensional array. You may not use a for loop.
def index_final_col(A):
    return A[:,A.shape[1]-1].reshape(A.shape[0],1)


# abs(transpose(theta)@p+theta_0)/norm(theta) # как найти расстояние из произвольной точки до плоскости

def signed_dist(x, th, th0):
    return (np.dot(th.T,x)+th0)/np.linalg.norm(th)

# в какой части плоскости X
import numpy as np
def positive(x, th, th0):
    return np.sign(np.matmul(th.T,x)+th0)
