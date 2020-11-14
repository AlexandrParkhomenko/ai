
# 1) Margin

import numpy as np

data = np.array([[1, 2, 1, 2, 10, 10.3, 10.5, 10.7],
                 [1, 1, 2, 2,  2,  2,  2, 2]])
labels = np.array([[-1, -1, 1, 1, 1, 1, 1, 1]])
blue_th = np.array([[0, 1]]).T
blue_th0 = -1.5
red_th = np.array([[1, 0]]).T
red_th0 = -2.5

def margin(x, thetas, theta0, label):
    #thetas,theta0 = np.array([1,1]),np.array([-4])
    #x = np.array([4,2])
    d = label*(np.dot(thetas.T,x)+theta0)/np.abs((np.dot(thetas.T,thetas)**0.5))
    print("margin =",d)
    print("sum =",np.sum(d))
    print("min =",np.amin(d))
    print("max =",np.amax(d))

margin(data, blue_th, blue_th0, labels) # separator maximizes S_min
margin(data, red_th,  red_th0,  labels)

# maximize S_min
