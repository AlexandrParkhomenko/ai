import numpy as np

# each column of X is an example, and each “column” of Y is the corresponding target output value
# X = np.array([])
# Y = np.array([])
X = np.array([[1, 2, 3, 4], [1, 1, 1, 1]])
Y = np.array([[1, 2.2, 2.8, 4.1]])
th = np.dot(np.linalg.inv(np.dot(X, np.transpose(X))),np.dot(X,np.transpose(Y)))
ans=th.tolist()
print(ans)

Z = X.T # n by d
T = Y.T # n by 1
th = np.dot(np.linalg.inv(np.dot(np.transpose(Z), Z)),np.dot(np.transpose(Z),T))
ans=th.tolist()
print(ans)

# [[0.9900000000000002], [0.05000000000000249]]
# [[0.9900000000000002], [0.05000000000000249]]
