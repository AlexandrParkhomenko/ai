"""
3) Simply inseparable

https://stackoverflow.com/questions/7852519/ternary-operator-for-numpy-ndarray
>>> print numpy.where(numpy.arange(10) < 3, 'a', 'b')
['a', 'a', 'a', 'b', 'b', 'b', 'b', 'b', 'b', 'b']
"""

import numpy as np

data = np.array([[1.1, 1, 4],[3.1, 1, 2]])
labels = np.array([[1, -1, -1]])
th = np.array([[1, 1]]).T
th0 = -4
g_ref = (2**0.5)/2

def hinge_loss(x, thetas, theta0, label, g_ref):
    d = label*(np.dot(thetas.T,x)+theta0)/np.abs((np.dot(thetas.T,thetas)**0.5))
    #print("margin =",d)
    return np.where(d < g_ref, 1 -d/g_ref, 0)
    

print(hinge_loss(data, th, th0, labels, g_ref))
