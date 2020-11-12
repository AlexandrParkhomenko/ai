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
