import numpy as np
def perceptron(data, labels, params={}, hook=None):
    # if T not in params, default to 100
    T = params.get('T', 100)
    N = data.shape[1]
    theta = np.zeros((data.shape[0],1))
    theta0 = np.zeros(1)
    #print("theta:",theta,theta0)
    for t in range(T):
        for n in range(N):
            x = data[:,n]
            y = labels[0,n]
            print("x=",x,", y=",y)
            if y*(np.dot(x,theta)+theta0) <= 0:
                theta[:,0] = theta[:,0]+ y*x # my error was here
                theta0 = theta0 + y
    return (theta,theta0)
