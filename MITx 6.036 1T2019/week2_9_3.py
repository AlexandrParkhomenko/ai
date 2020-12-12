import numpy as np
from code_for_hw02 import *

def perceptron(data, labels, params={}, hook=None):
    # if T not in params, default to 100
    T = params.get('T', 100)
    N = data.shape[1]
    #print("N=",N)
    theta = np.zeros((data.shape[0],1))
    theta0 = np.zeros(1)
    #print("theta:",theta,theta0)
    for t in range(T):
        for n in range(N):
            x = data[:,n]
            y = labels[0,n]
            #print("x=",x,", y=",y)
            if y*(np.dot(x,theta)+theta0) <= 0:
                theta[:,0] = theta[:,0]+ y*x # my error was here
                theta0 = theta0 + y
    return (theta,theta0)

def xval_learning_alg(learner, data, labels, k):
    #cross validation of learning algorithm
#divide D into k parts, as equally as possible;  call them D_i for i == 0 .. k-1
## be sure the data is shuffled in case someone put all the positive examples first in the data!
    data_split = np.array(np.array_split(data, k, axis=1))
    #print("data.shape:", np.array(data).shape)
    #print("data_split.shape:", data_split[0].shape)
    labels_split = np.array(np.array_split(labels, k, axis=1))
    #print("labels_split.shape:", labels_split[0].shape)
    s = 0
    #data_minus_j = np.concatenate(data_split, axis=1)
    #labels_minus_j = np.concatenate(labels_split, axis=1)
    for j in range(k): # 0 to k-1
        #D_minus_j = union of all the datasets D_i, except for D_j
        # hide elements
        data_j,  labels_j = data_split[j], labels_split[j]
        data_split = np.delete(data_split, j, axis=0) #
        #print("data_split.shape:", data_split[0].shape)
        labels_split = np.delete(labels_split, j, axis=0) #
        # we concantenate rest
        data_minus_j = np.concatenate(data_split, axis=1)
        #print("data_minus_j.shape:", data_minus_j.shape)
        labels_minus_j = np.concatenate(labels_split, axis=1)
        # show elements
        data_split = np.insert(data_split, j, data_j, axis=0) #
        labels_split = np.insert(labels_split, j, labels_j, axis=0) #
        
        h_j = learner(data_minus_j, labels_minus_j) #L(D_minus_j)
        #score_j = accuracy of h_j measured on D_j
        s += score(data_split[j],  labels_split[j], h_j[0], h_j[1])/labels_split[j].shape[1]
        #print("s=",s)
    return s/k  #average(score0, ..., score(k-1))

result = xval_learning_alg(perceptron, big_data, big_data_labels, 5)
print("result:", result, "== 0.61", result == 0.61)
