import numpy as np

data = np.array([[1, 1, 3, 3],[3, 1, 4, 2]])
labels = np.array([[-1, -1, 1, 1]])
th = np.array([[0, 1]]).T
th0 = -3

def margin(x, thetas, theta0, label):
    #thetas,theta0 = np.array([1,1]),np.array([-4])
    #x = np.array([4,2])
    d = label*(np.dot(thetas.T,x)+theta0)/np.abs((np.dot(thetas.T,thetas)**0.5))
    print("distance =",d)

for i in range(data.shape[1]):
    margin(data[:,i], th, th0, labels[:,i])
# or
margin(data, th, th0, labels)
