# see https://openlearninglibrary.mit.edu/courses/course-v1:MITx+6.036+1T2019/courseware/Week3/week3_homework
# 2) Encoding Discrete Values

import numpy as np

def perceptron(data, labels, params={}, hook=None):
    # if T not in params, default to 100
    T = params.get('T', 100)
    N = data.shape[1]
    #print("N=",N)
    theta = np.zeros((data.shape[0],1))
    theta0 = np.zeros(1)
    #print("theta:",theta,theta0)
    e=0
    for t in range(T):
        for n in range(N):
            x = data[:,n]
            y = labels[0,n]
            #print("x=",x,", y=",y)
            if y*(np.dot(x,theta)+theta0) <= 0:
                theta[:,0] = theta[:,0]+ y*x # my error was here
                theta0 = theta0 + y
                e+=1
    print("all_errors==",e)
    return (theta,theta0)

data =   np.array([[2, 3,  4,  5]])
labels = np.array([[1, 1, -1, -1]])
k = 6

def one_hot(x, k):
    r=np.zeros(k)
    r[x-1]=1
    return np.array([r])

data2 = np.array([])
for i in range(data[0].shape[0]):
    data2 = np.append( data2, [[one_hot(data[0][i],k)]] ) #
print(np.reshape(data2,[data[0].shape[0],k]).T)

# data_full == data2
data_full =   np.array([
    [0,0,0,0],
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,1],
    [0,0,0,0]
    ])
print(perceptron(data_full, labels, params = {'T':100}, hook = None))
# [ 0., 2., 1., -2., -1., 0., 0. ]
