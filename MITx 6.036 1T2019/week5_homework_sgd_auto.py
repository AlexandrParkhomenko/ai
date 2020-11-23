
import time    

#time measuring decorator that is called as a wrapper by @timeit above func names
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        #print('%r (%r, %r) %2.2f sec' %(method.__name__, args, kw, te-ts))
        print('%r %2.5f sec' %(method.__name__, te-ts))
        return result
    return timed

#------------------------------------------------------------------------------

# Takes a list of numbers and returns a column vector:  n x 1
def cv(value_list):
    """ Return a d x 1 np array.
        value_list is a python list of values of length d.

    >>> cv([1,2,3])
    array([[1],
           [2],
           [3]])
    """
    return np.transpose(rv(value_list))

# Takes a list of numbers and returns a row vector: 1 x n
def rv(value_list):
    """ Return a 1 x d np array.
        value_list is a python list of values of length d.

    >>> rv([1,2,3])
    array([[1, 2, 3]])
    """
    return np.array([value_list])

def lin_reg(x, th, th0):
    """ Returns the predicted y

    >>> X = np.array([[ 1.,  2.,  3.,  4.], [ 1.,  1.,  1.,  1.]])
    >>> th = np.array([[ 1.  ], [ 0.05]]) ; th0 = np.array([[ 0.]])
    >>> lin_reg(X, th, th0).tolist()
    [[1.05, 2.05, 3.05, 4.05]]
    >>> th = np.array([[ 1.  ], [ 0.05]]) ; th0 = np.array([[ 2.]])
    >>> lin_reg(X, th, th0).tolist()
    [[3.05, 4.05, 5.05, 6.05]]
    """
    return np.dot(th.T, x) + th0

def square_loss(x, y, th, th0):
    """ Returns the squared loss between y_pred and y

    >>> X = np.array([[ 1.,  2.,  3.,  4.], [ 1.,  1.,  1.,  1.]])
    >>> Y = np.array([[ 1. ,  2.2,  2.8,  4.1]])
    >>> th = np.array([[ 1.  ], [ 0.05]]) ; th0 = np.array([[ 2.]])
    >>> square_loss(X, Y, th, th0).tolist()
    [[4.2025, 3.4224999999999985, 5.0625, 3.8025000000000007]]
    """
    return (y - lin_reg(x, th, th0))**2

def mean_square_loss(x, y, th, th0):
    """ Return the mean squared loss between y_pred and y

    >>> X = np.array([[ 1.,  2.,  3.,  4.], [ 1.,  1.,  1.,  1.]])
    >>> Y = np.array([[ 1. ,  2.2,  2.8,  4.1]])
    >>> th = np.array([[ 1.  ], [ 0.05]]) ; th0 = np.array([[ 2.]])
    >>> mean_square_loss(X, Y, th, th0).tolist()
    [[4.1225]]
    """
    # the axis=1 and keepdims=True are important when x is a full matrix
    return np.mean(square_loss(x, y, th, th0), axis = 1, keepdims = True)

def ridge_obj(x, y, th, th0, lam):
    """ Return the ridge objective value

    >>> X = np.array([[ 1.,  2.,  3.,  4.], [ 1.,  1.,  1.,  1.]])
    >>> Y = np.array([[ 1. ,  2.2,  2.8,  4.1]])
    >>> th = np.array([[ 1.  ], [ 0.05]]) ; th0 = np.array([[ 2.]])
    >>> ridge_obj(X, Y, th, th0, 0.0).tolist()
    [[4.1225]]
    >>> ridge_obj(X, Y, th, th0, 0.5).tolist()
    [[4.623749999999999]]
    >>> ridge_obj(X, Y, th, th0, 100.).tolist()
    [[104.37250000000002]]
    """
    return np.mean(square_loss(x, y, th, th0), axis = 1, keepdims = True) + lam * np.linalg.norm(th)**2


def compute_loss(theta, X, y):
    return ((np.linalg.norm(np.dot(X,theta) - y))**2)/(2*X.shape[0])

def J(Xi, yi, w):
    # translate from (1-augmented X, y, theta) to (separated X, y, th, th0) format
    return float(ridge_obj(Xi[:-1,:], yi, w[:-1,:], w[-1:,:], 0))

def dJ(Xi, yi, w):
    def f(w): return J(Xi, yi, w)
    return num_grad(f)(w)

def num_grad(f):
    def df(x):
        g = np.zeros(x.shape)
        delta = 0.001
        for i in range(x.shape[0]):
            xi = x[i,0]
            x[i,0] = xi - delta
            xm = f(x)
            x[i,0] = xi + delta
            xp = f(x)
            x[i,0] = xi
            g[i,0] = (xp - xm)/(2*delta)
        return g
    return df

def package_ans(p):
    w,fs,ws = p
    print("Weights:",w)
    print("fs first and last elements:",fs[0],fs[len(fs)-1])
    print("ws first and last elements:",ws[0],ws[len(ws)-1])
    print("Length of fs:",len(fs))
    print("Length of ws:",len(ws))

def sgd(X, y, J, dJ, w0, step_size_fn, max_iter):
    ws,fs,t,w,n = [],[],0,w0,y.shape[1]
    while t < max_iter: # 3: #
        t += 1
        w0 = np.copy(w)
        ws.append(w)
        i = np.random.randint(n)
        X_i, y_i = cv(X[:,i]), cv(y[:,i])
        fs.append(J(X_i, y_i, w))
        w = w - step_size_fn(t) * dJ(X_i, y_i, w)  
    return (w0,fs,ws)

#------------------------------------------------------------------------------

import numpy as np
import code_for_hw3_part2 as hw3

auto_data = hw3.load_auto_data("./auto-mpg.tsv")
features = [('cylinders',hw3.one_hot),
            ('displacement',hw3.standard),
            ('horsepower',hw3.standard),
            ('weight',hw3.standard),
            ('acceleration',hw3.standard),
            ('origin',hw3.one_hot)
            ]
'''
1. mpg:           continuous
2. cylinders:     multi-valued discrete #  one_hot
3. displacement:  continuous
4. horsepower:    continuous
5. weight:        continuous
6. acceleration:  continuous
#  7. model year: multi-valued discrete
8. origin:        multi-valued discrete
#  9. car name:   string (many values)
'''
@timeit
def xval_averaged_perceptron():
    data, labels = hw3.auto_data_and_labels(auto_data, features)
    result = hw3.xval_learning_alg(hw3.averaged_perceptron, data, labels, 10)
    print(result)

# xval_averaged_perceptron()

data, labels = hw3.auto_data_and_labels(auto_data, features)
result = hw3.xval_learning_alg(hw3.averaged_perceptron, data, labels, 10) # TODO need SGD function
print(result)
