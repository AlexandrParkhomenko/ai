# 1.2) General hyperplane, distance to origin
import numpy as np
thetas,theta0 = np.array([3,4]),np.array([5])
x = np.array([0,0])
d = (np.dot(thetas.T,x)+theta0)/(np.dot(thetas.T,thetas)**0.5)
print("distance =",d)
#1
