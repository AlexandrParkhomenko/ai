# 6.2B)
# Using the extracted features from above, run the perceptron algorithm
# on the set of 0 vs. 1 images.

# original file name: hw3_part2_main.py

import pdb
import numpy as np
import code_for_hw3_part2 as hw3

#-------------------------------------------------------------------------------
# MNIST Data
#-------------------------------------------------------------------------------

"""
Returns a dictionary formatted as follows:
{
    0: {
        "images": [(m by n image), (m by n image), ...],
        "labels": [0, 0, ..., 0]
    },
    1: {...},
    ...
    9
}
Where labels range from 0 to 9 and (m, n) images are represented
by arrays of floats from 0 to 1
"""
mnist_data_all = hw3.load_mnist_data(range(10))

print('mnist_data_all loaded. shape of single images is', mnist_data_all[0]["images"][0].shape)

# HINT: change the [0] and [1] if you want to access different images
d0 = mnist_data_all[0]["images"]
d1 = mnist_data_all[1]["images"]
y0 = np.repeat(-1, len(d0)).reshape(1,-1)
y1 = np.repeat(1, len(d1)).reshape(1,-1)

# data goes into the feature computation functions
data = np.vstack((d0, d1))
# labels can directly go into the perceptron algorithm
labels = np.vstack((y0.T, y1.T)).T

#-------------------------------------------------------------------------------
# Your code here to process the MNIST data

def raw_mnist_features(x): #x.shape = (160, 28, 28)
    """
    @param x (n_samples,m,n) array with values in (0,1)
    @return (m*n,n_samples) reshaped array where each entry is preserved
    """
    n,r,c = x.shape
    return np.array(x).reshape((n,r*c)).T

def row_mnist_features(x):
    """
    @param x (n_samples,m,n) array with values in (0,1)
    @return (m,n_samples) reshaped array where each entry is preserved
    """
    n,r,c = x.shape
    ret = []
    for i in range(n):
        ret.append(row_average_features(x[i,:,:]))
    return np.array(ret).reshape((n,c)).T

def col_mnist_features(x):
    """
    @param x (n_samples,m,n) array with values in (0,1)
    @return (n,n_samples) reshaped array where each entry is preserved
    """
    n,r,c = x.shape
    ret = []
    for i in range(n):
        ret.append(col_average_features(x[i,:,:]))
    return np.array(ret).reshape((n,r)).T

def top_bottom_mnist_features(x):
    """
    @param x (n_samples,m,n) array with values in (0,1)
    @return (2,n_samples) reshaped array where each entry is preserved
    """
    n,r,c = x.shape
    ret = []
    for i in range(n):
        ret.append(top_bottom_features(x[i,:,:]))
    return np.array(ret).reshape((n,2)).T


def row_average_features(x):
    """
    @param x (m,n) array with values in (0,1)
    @return (m,1) array where each entry is the average of a row
    """
    #print(np.array(x).shape)
    return np.array([np.sum(x, axis=1)/np.array(x).shape[1]]).T


def col_average_features(x):
    """
    @param x (m,n) array with values in (0,1)
    @return (n,1) array where each entry is the average of a column
    """
    return np.array([np.sum(x, axis=0)/np.array(x).shape[0]]).T



def top_bottom_features(x):
    """
    @param x (m,n) array with values in (0,1)
    @return (2,1) array where the first entry is the average of the
    top half of the image = rows 0 to floor(m/2) [exclusive]
    and the second entry is the average of the bottom half of the image
    = rows floor(m/2) [inclusive] to m
    """
    x = np.array(x)
    r,c = x.shape
    #return row_average_features(np.array(x).reshape((2,r*c//2)))
    #print((r+1)//2)
    a = np.array(x[0:r//2,:]).reshape(1,c*(r//2))
    b = np.array(x[r//2:r,:]).reshape(1,c*(r-(r//2)))
    d = np.array([])
    d = np.append(d,row_average_features(a))
    d = np.append(d,row_average_features(b))
    d = np.array([d]).T
    return d


# use this function to evaluate accuracy
#acc = hw3.get_classification_accuracy(raw_mnist_features(data), labels)

#-------------------------------------------------------------------------------
# Analyze MNIST data
#-------------------------------------------------------------------------------

# Your code here to process the MNIST data
acc = hw3.get_classification_accuracy(row_mnist_features(data), labels)
print(acc)
acc = hw3.get_classification_accuracy(col_mnist_features(data), labels)
print(acc)
acc = hw3.get_classification_accuracy(top_bottom_mnist_features(data), labels)
print(acc)

# 0.48125
# 0.6375
# 0.48125
