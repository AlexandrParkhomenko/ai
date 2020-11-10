import numpy as np

def raw_mnist_features(x):
    """
    @param x (n_samples,m,n) array with values in (0,1)
    @return (m*n,n_samples) reshaped array where each entry is preserved
    """
    n,r,c = x.shape
    print(x.shape)
    #x = np.append(np.array([]),x) # this line is redundant
    return np.array(x).reshape((n,r*c)).T

x = np.array([[[1,2,3,4],[5,6,7,8],[9,10,11,12]],[[13,14,15,16],[17,18,19,20],[21,22,23,24]]])
y = x.reshape((4,2)).T
print("y =",y)

print("raw_mnist_features(x) =", raw_mnist_features(x))
