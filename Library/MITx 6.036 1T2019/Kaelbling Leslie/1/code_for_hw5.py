# 9) Predicting mpg values

import numpy as np
import csv
import itertools, functools, operator
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

def package_ans(p):
    w,fs,ws = p
    print("Weights:",w)
    print("fs first and last elements:",fs[0],fs[len(fs)-1])
    print("ws first and last elements:",ws[0],ws[len(ws)-1])
    print("Length of fs:",len(fs))
    print("Length of ws:",len(ws))

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

# In all the following definitions:
# x is d by n : input data
# y is 1 by n : output regression values
# th is d by 1 : weights
# th0 is 1 by 1 or scalar
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

def d_lin_reg_th(x, th, th0):
    """ Returns the gradient of lin_reg(x, th, th0) with respect to th

    Note that for array (rather than vector) x, we get a d x n 
    result. That is to say, this function produces the gradient for
    each data point i ... n, with respect to each theta, j ... d.

    >>> X = np.array([[ 1.,  2.,  3.,  4.], [ 1.,  1.,  1.,  1.]])
    >>> th = np.array([[ 1.  ], [ 0.05]]); th0 = np.array([[ 2.]])
    >>> d_lin_reg_th(X[:,0:1], th, th0).tolist()
    [[1.0], [1.0]]

    >>> d_lin_reg_th(X, th, th0).tolist()
    [[1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0]]
    """
    #Your code here
    return x

def d_square_loss_th(x, y, th, th0):
    """Returns the gradient of square_loss(x, y, th, th0) with respect to
       th.

       Note: should be a one-line expression that uses lin_reg and
       d_lin_reg_th (i.e., uses the chain rule).

       Should work with X, Y as vectors, or as arrays. As in the
       discussion of d_lin_reg_th, this should give us back an n x d
       array -- so we know the sensitivity of square loss for each
       data point i ... n, with respect to each element of theta.

    >>> X = np.array([[ 1.,  2.,  3.,  4.], [ 1.,  1.,  1.,  1.]])
    >>> Y = np.array([[ 1. ,  2.2,  2.8,  4.1]])
    >>> th = np.array([[ 1.  ], [ 0.05]]) ; th0 = np.array([[ 2.]])
    >>> d_square_loss_th(X[:,0:1], Y[:,0:1], th, th0).tolist()
    [[4.1], [4.1]]

    >>> d_square_loss_th(X, Y, th, th0).tolist()
    [[4.1, 7.399999999999999, 13.5, 15.600000000000001], [4.1, 3.6999999999999993, 4.5, 3.9000000000000004]]

    """
    #Your code here
    return -2 * (y - lin_reg(x, th, th0)) * d_lin_reg_th(x, th, th0)

def d_mean_square_loss_th(x, y, th, th0):
    """ Returns the gradient of mean_square_loss(x, y, th, th0) with
        respect to th.  

        Note: It should be a one-line expression that uses d_square_loss_th.

    >>> X = np.array([[ 1.,  2.,  3.,  4.], [ 1.,  1.,  1.,  1.]])
    >>> Y = np.array([[ 1. ,  2.2,  2.8,  4.1]])
    >>> th = np.array([[ 1.  ], [ 0.05]]) ; th0 = np.array([[ 2.]])
    >>> d_mean_square_loss_th(X[:,0:1], Y[:,0:1], th, th0).tolist()
    [[4.1], [4.1]]

    >>> d_mean_square_loss_th(X, Y, th, th0).tolist()
    [[10.15], [4.05]]
    """
    # print("X =", repr(X))
    # print("Y =", repr(Y))
    # print("th =", repr(th), "th0 =", repr(th0))
    #Your code here
    return np.mean(d_square_loss_th(x, y, th, th0), axis = 1, keepdims = True)

def d_lin_reg_th0(x, th, th0):
    """ Returns the gradient of lin_reg(x, th, th0) with respect to th0.

    >>> x = np.array([[ 1.,  2.,  3.,  4.], [ 1.,  1.,  1.,  1.]])
    >>> th = np.array([[ 1.  ], [ 0.05]]) ; th0 = np.array([[ 2.]])
    >>> d_lin_reg_th0(x, th, th0).tolist()
    [[1.0, 1.0, 1.0, 1.0]]
    """
    #Your code here
    return rv(np.ones(x.shape[1]))

def d_square_loss_th0(x, y, th, th0):
    """ Returns the gradient of square_loss(x, y, th, th0) with
        respect to th0.

    # Note: uses broadcasting!

    >>> X = np.array([[ 1.,  2.,  3.,  4.], [ 1.,  1.,  1.,  1.]])
    >>> Y = np.array([[ 1. ,  2.2,  2.8,  4.1]])
    >>> th = np.array([[ 1.  ], [ 0.05]]) ; th0 = np.array([[ 2.]])
    >>> d_square_loss_th0(X, Y, th, th0).tolist()
    [[4.1, 3.6999999999999993, 4.5, 3.9000000000000004]]
    """
    #Your code here
    return -2 * (y - lin_reg(x, th, th0)) * d_lin_reg_th0(x, th, th0)

def d_mean_square_loss_th0(x, y, th, th0):
    """ Returns the gradient of mean_square_loss(x, y, th, th0) with
    respect to th0.

    >>> X = np.array([[ 1.,  2.,  3.,  4.], [ 1.,  1.,  1.,  1.]])
    >>> Y = np.array([[ 1. ,  2.2,  2.8,  4.1]])
    >>> th = np.array([[ 1.  ], [ 0.05]]) ; th0 = np.array([[ 2.]])
    >>> d_mean_square_loss_th0(X, Y, th, th0).tolist()
    [[4.05]]
    """
    #Your code here
    return np.mean(d_square_loss_th0(x, y, th, th0), axis = 1, keepdims = True)

def d_ridge_obj_th(x, y, th, th0, lam):
    """Return the derivative of tghe ridge objective value with respect
    to theta.

    Note: uses broadcasting to add d x n to d x 1 array below

    >>> X = np.array([[ 1.,  2.,  3.,  4.], [ 1.,  1.,  1.,  1.]])
    >>> Y = np.array([[ 1. ,  2.2,  2.8,  4.1]])
    >>> th = np.array([[ 1.  ], [ 0.05]]) ; th0 = np.array([[ 2.]])
    >>> d_ridge_obj_th(X, Y, th, th0, 0.0).tolist()
    [[10.15], [4.05]]
    >>> d_ridge_obj_th(X, Y, th, th0, 0.5).tolist()
    [[11.15], [4.1]]
    >>> d_ridge_obj_th(X, Y, th, th0, 100.).tolist()
    [[210.15], [14.05]]
    """
    #Your code here
    return d_mean_square_loss_th(x, y, th, th0) + 2*lam*th

def d_ridge_obj_th0(x, y, th, th0, lam):
    """Return the derivative of tghe ridge objective value with respect
    to theta.

    Note: uses broadcasting to add d x n to d x 1 array below

    >>> X = np.array([[ 1.,  2.,  3.,  4.], [ 1.,  1.,  1.,  1.]])
    >>> Y = np.array([[ 1. ,  2.2,  2.8,  4.1]])
    >>> th = np.array([[ 1.  ], [ 0.05]]) ; th0 = np.array([[ 2.]])
    >>> d_ridge_obj_th0(X, Y, th, th0, 0.0).tolist()
    [[4.05]]
    >>> d_ridge_obj_th0(X, Y, th, th0, 0.5).tolist()
    [[4.05]]
    >>> d_ridge_obj_th0(X, Y, th, th0, 100.).tolist()
    [[4.05]]
    """
    #Your code here
    return d_mean_square_loss_th0(x, y, th, th0)

#Concatenates the gradients with respect to theta and theta_0
def ridge_obj_grad(x, y, th, th0, lam):
    grad_th = d_ridge_obj_th(x, y, th, th0, lam)
    grad_th0 = d_ridge_obj_th0(x, y, th, th0, lam)
    return np.vstack([grad_th, grad_th0])    

def sgd(X, y, J, dJ, w0, step_size_fn, max_iter):
    """Implements stochastic gradient descent

    Inputs:
    X: a standard data array (d by n)
    y: a standard labels row vector (1 by n)

    J: a cost function whose input is a data point (a column vector),
    a label (1 by 1) and a weight vector w (a column vector) (in that
    order), and which returns a scalar.

    dJ: a cost function gradient (corresponding to J) whose input is a
    data point (a column vector), a label (1 by 1) and a weight vector
    w (a column vector) (also in that order), and which returns a
    column vector.

    w0: an initial value of weight vector www, which is a column
    vector.

    step_size_fn: a function that is given the (zero-indexed)
    iteration index (an integer) and returns a step size.

    max_iter: the number of iterations to perform

    Returns: a tuple (like gd):
    w: the value of the weight vector at the final step
    fs: the list of values of JJJ found during all the iterations
    ws: the list of values of www found during all the iterations

    """
    #Your code here
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

############################################################
#From HW04; Used in the test case for sgd, below
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

#Test case for sgd
@timeit
def sgdTest():
    def downwards_line():
        X = np.array([[0.0, 0.1, 0.2, 0.3, 0.42, 0.52, 0.72, 0.78, 0.84, 1.0],
                      [1.0, 1.0, 1.0, 1.0, 1.0,  1.0,  1.0,  1.0,  1.0,  1.0]])
        y = np.array([[0.4, 0.6, 1.2, 0.1, 0.22, -0.6, -1.5, -0.5, -0.5, 0.0]])
        return X, y
    
    X, y = downwards_line()
    
    def J(Xi, yi, w):
        # translate from (1-augmented X, y, theta) to (separated X, y, th, th0) format
        return float(ridge_obj(Xi[:-1,:], yi, w[:-1,:], w[-1:,:], 0))
    
    def dJ(Xi, yi, w):
        def f(w): return J(Xi, yi, w)
        return num_grad(f)(w)

    #Insert code to call sgd on the above
    #Your code here
    ans=package_ans(sgd(X, y, J, dJ, cv([0., 0.]), lambda i: 0.1, 1000))
    print(ans)

# sgdTest()

############################################################

def ridge_min(X, y, lam):
    """ Returns th, th0 that minimize the ridge regression objective
    
    Assumes that X is NOT 1-extended. Interfaces to our sgd by 1-extending
    and building corresponding initial weights.
    """
    def svm_min_step_size_fn(i):
        return 0.01/(i+1)**0.5

    d, n = X.shape
    X_extend = np.vstack([X, np.ones((1, n))])
    w_init = np.zeros((d+1, 1))

    def J(Xj, yj, th):
        return float(ridge_obj(Xj[:-1,:], yj, th[:-1,:], th[-1:,:], lam))

    def dJ(Xj, yj, th):
        return ridge_obj_grad(Xj[:-1,:], yj, th[:-1,:], th[-1:,:], lam)
    
    np.random.seed(0)
    w, fs, ws = sgd(X_extend, y, J, dJ, w_init, svm_min_step_size_fn, 1000)
    return w[:-1,:], w[-1:,:]

#######################################################################

def mul(seq):
    '''
    Given a list or numpy array of float or int elements, return the product 
    of all elements in the list/array.  
    '''
    return functools.reduce(operator.mul, seq, 1)

def make_polynomial_feature_fun(order):
    '''
    Transform raw features into polynomial features or order 'order'.
    If raw_features is a d by n numpy array, return a k by n numpy array 
    where k = sum_{i = 0}^order multichoose(d, i) (the number of all possible terms in the polynomial feature or order 'order')
    '''
    def f(raw_features):
        d, n = raw_features.shape
        result = []   # list of column vectors
        for j in range(n):
            features = []
            for o in range(1, order+1):
                indexTuples = \
                          itertools.combinations_with_replacement(range(d), o)
                for it in indexTuples:
                    features.append(mul(raw_features[i, j] for i in it))
            result.append(cv(features))
        return np.hstack(result)
    return f

######################################################################

#First finds a predictor on X_train and X_test using the specified value of lam
#Then runs on X_test, Y_test to find the RMSE
def eval_predictor(X_train, Y_train, X_test, Y_test, lam):
    th, th0 = ridge_min(X_train, Y_train, lam)
    return np.sqrt(mean_square_loss(X_test, Y_test, th, th0))

#Returns the mean RMSE from cross validation given a dataset (X, y), a value of lam,
#and number of folds, k
def xval_learning_alg(X, y, lam, k):
    '''
    Given a learning algorithm and data set, evaluate the learned classifier's score with k-fold
    cross validation. 
    
    learner is a learning algorithm, such as perceptron.
    data, labels = dataset and its labels.

    k: the "k" of k-fold cross validation
    '''
    _, n = X.shape
    idx = list(range(n))
    np.random.seed(0)
    np.random.shuffle(idx)
    X, y = X[:,idx], y[:,idx]

    split_X = np.array_split(X, k, axis=1)
    split_y = np.array_split(y, k, axis=1)

    score_sum = 0
    for i in range(k):
        X_train = np.concatenate(split_X[:i] + split_X[i+1:], axis=1)
        y_train = np.concatenate(split_y[:i] + split_y[i+1:], axis=1)
        X_test = np.array(split_X[i])
        y_test = np.array(split_y[i])
        score_sum += eval_predictor(X_train, y_train, X_test, y_test, lam)
    return score_sum/k

######################################################################
# For auto dataset

def load_auto_data(path_data):
    """
    Returns a list of dict with keys:
    """
    numeric_fields = {'mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                      'acceleration', 'model_year', 'origin'}
    data = []
    with open(path_data) as f_data:
        for datum in csv.DictReader(f_data, delimiter='\t'):
            for field in list(datum.keys()):
                if field in numeric_fields and datum[field]:
                    datum[field] = float(datum[field])
            data.append(datum)
    return data

# Feature transformations
def std_vals(data, f):
    '''
    Helper function to be used inside auto_data_and_labels. Returns average and standard deviation of 
    data's f-th feature. 
    >>> data = np.array([[1,2,3,4,5],[6,7,8,9,10]])
    >>> f=0
    >>> std_vals(data, f)
    (3.5, 2.5)
    >>> f=3
    >>> std_vals(data, f)
    (6.5, 2.5)
    '''
    vals = [entry[f] for entry in data]
    avg = sum(vals)/len(vals)
    dev = [(entry[f] - avg)**2 for entry in data]
    sd = (sum(dev)/len(vals))**0.5
    return (avg, sd)

def standard(v, std):
    '''
    Helper function to be used in auto_data_and_labels. Center v by the 0-th element of std and scale by the 1-st element of std. 
    >>> data = np.array([1,2,3,4,5])
    >>> standard(data, (3,1))
    [array([-2., -1.,  0.,  1.,  2.])]
    >>> data = np.array([1,2,5,7,8])
    >>> standard(data, (3,1))
    [array([-2., -1.,  2.,  4.,  5.])]
    '''
    return [(v-std[0])/std[1]]

def raw(x):
    '''
    Make x into a nested list. Helper function to be used in auto_data_and_labels.
    >>> data = [1,2,3,4]
    >>> raw(data)
    [[1, 2, 3, 4]]
    '''
    return [x]

def one_hot(v, entries):
    '''
    Outputs a one hot vector. Helper function to be used in auto_data_and_labels.
    v is the index of the "1" in the one-hot vector.
    entries is range(k) where k is the length of the desired onehot vector. 

    >>> one_hot(2, range(4))
    [0, 0, 1, 0]
    >>> one_hot(1, range(5))
    [0, 1, 0, 0, 0]
    '''
    vec = len(entries)*[0]
    vec[entries.index(v)] = 1
    return vec

# The class (mpg) added to the front of features
def auto_data_and_values(auto_data, features):
    features = [('mpg', raw)] + features
    std = {f:std_vals(auto_data, f) for (f, phi) in features if phi==standard}
    entries = {f:list(set([entry[f] for entry in auto_data])) \
               for (f, phi) in features if phi==one_hot}
    vals = []
    for entry in auto_data:
        phis = []
        for (f, phi) in features:
            if phi == standard:
                # print("std>",std[f])
                phis.extend(phi(entry[f], std[f]))
            elif phi == one_hot:
                phis.extend(phi(entry[f], entries[f]))
            else:
                phis.extend(phi(entry[f]))
        vals.append(np.array([phis]))
    data_labels = np.vstack(vals)
    return data_labels[:, 1:].T, data_labels[:, 0:1].T

######################################################################

#standardizes a row vector of y values
#returns both the standardized vector and the mean, variance
def std_y(row):
    '''
    >>> std_y(np.array([[1,2,3,4]]))
    (array([[-1.34164079, -0.4472136 ,  0.4472136 ,  1.34164079]]), array([2.5]), array([1.11803399]))
    '''
    mu = np.mean(row, axis=1)
    sigma = np.sqrt(np.mean((row - mu)**2, axis=1))
    return np.array([(val - mu)/(1.0*sigma) for val in row]), mu, sigma

@timeit
def xval_learning_alg_timeit(data_poly, labels, lam, k):
    result = xval_learning_alg(data_poly, labels, lam, k)
    return result

def s():
    results = np.array([])
    features = [[('cylinders',standard),
            ('displacement',standard),
            ('horsepower',standard),
            ('weight',standard),
            ('acceleration',standard),
            ('origin',one_hot)
            ],
            
            [('cylinders',one_hot),
            ('displacement',standard),
            ('horsepower',standard),
            ('weight',standard),
            ('acceleration',standard),
            ('origin',one_hot)
            ]]
    auto_data = load_auto_data("./auto-mpg-regression.tsv")
    for feature in range(len(features)): 
        data, labels = auto_data_and_values(auto_data, features[feature])
        lams = [0.0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
        # lam = 0
        k = 10
        # please, use standard labels
        labels, mu, sigma = std_y(labels)
        for polynomial_order in range(1,3):
            data_poly = make_polynomial_feature_fun(polynomial_order)(data)
            for lam in lams:
                result = xval_learning_alg(data_poly, labels, lam, k)
                result = np.amin(result)*sigma
                results = np.append(results,[feature,polynomial_order, lam, result])
                print("feature:",feature,"polynomial_order:",polynomial_order,"lam:", lam, "result:", result)
        for polynomial_order in range(3,4):
            print("polynomial_order",polynomial_order)
            data_poly = make_polynomial_feature_fun(polynomial_order)(data)
            for lam in range(0,220,20):
                result = xval_learning_alg(data_poly, labels, lam, k)
                result = np.amin(result)*sigma
                results = np.append(results,[feature,polynomial_order, lam, result])
                print("feature:",feature,"polynomial_order:",polynomial_order,"lam:", lam, "result:", result)
    return results.reshape((results.shape[0]//4,4))

# ...
# feature: 0 polynomial_order: 1 lam: 0.1 result: [4.28635766]
# feature: 0 polynomial_order: 2 lam: 0.0 result: [4.02467658] # !!!
# feature: 0 polynomial_order: 2 lam: 0.01 result: [4.02559093]
# ...

@timeit
def r():
    features = [('cylinders',one_hot),
            ('displacement',raw),
            ('horsepower',raw),
            ('weight',raw),
            ('acceleration',raw),
            ('origin',one_hot)
            ]
    auto_data = load_auto_data("./auto-mpg-regression.tsv")
    data, labels = auto_data_and_values(auto_data, features)
    
    lam = 0
    k = 10
    result = xval_learning_alg(data, labels, lam, k)
    print(result)

results = s() 
# [[5.0054957]]
# 's' 1.88939 sec
# poly
# [[4.95144256]]
# 's' 1.96218 sec

# r() #overflow encountered in multiply

# code_for_hw5.py:76: RuntimeWarning: overflow encountered in square
#   return (y - lin_reg(x, th, th0))**2
# code_for_hw5.py:103: RuntimeWarning: invalid value encountered in double_scalars
#   return np.mean(square_loss(x, y, th, th0), axis = 1, keepdims = True) + lam * np.linalg.norm(th)**2
# code_for_hw5.py:146: RuntimeWarning: overflow encountered in multiply
#   return -2 * (y - lin_reg(x, th, th0)) * d_lin_reg_th(x, th, th0)
# code_for_hw5.py:146: RuntimeWarning: invalid value encountered in multiply
#   return -2 * (y - lin_reg(x, th, th0)) * d_lin_reg_th(x, th, th0)
# code_for_hw5.py:225: RuntimeWarning: invalid value encountered in multiply
#   return d_mean_square_loss_th(x, y, th, th0) + 2*lam*th
# code_for_hw5.py:193: RuntimeWarning: overflow encountered in multiply
#   return -2 * (y - lin_reg(x, th, th0)) * d_lin_reg_th0(x, th, th0)
# [[nan]]
# 'r' 1.80017 sec

# features = [('cylinders',one_hot),
#             ('displacement',standard),
#             ('horsepower',standard),
#             ('weight',standard),
#             ('acceleration',standard),
#             ('origin',one_hot)
#             ]
# lam = 0
# k = 10
# polynomial_order = 2
# auto_data = load_auto_data("./auto-mpg-regression.tsv")
# data, labels = auto_data_and_values(auto_data, features)
# data_poly = make_polynomial_feature_fun(polynomial_order)(data)
# # please, use standard labels
# labels, mu, sigma = std_y(labels)
# result = xval_learning_alg(data_poly, labels, lam, k)
# result = np.amin(result)*sigma
# # results = np.append(results,[feature,polynomial_order, lam, result])
# print("feature:",2,"polynomial_order:",polynomial_order,"lam:", lam, "result:", result)

# APPENDIX. Full log
# feature: 0 polynomial_order: 1 lam: 0.0 result: [4.27495712]
# feature: 0 polynomial_order: 1 lam: 0.01 result: [4.27588322]
# feature: 0 polynomial_order: 1 lam: 0.02 result: [4.27685901]
# feature: 0 polynomial_order: 1 lam: 0.03 result: [4.27788368]
# feature: 0 polynomial_order: 1 lam: 0.04 result: [4.27895641]
# feature: 0 polynomial_order: 1 lam: 0.05 result: [4.28007639]
# feature: 0 polynomial_order: 1 lam: 0.06 result: [4.28124283]
# feature: 0 polynomial_order: 1 lam: 0.07 result: [4.28245494]
# feature: 0 polynomial_order: 1 lam: 0.08 result: [4.28371196]
# feature: 0 polynomial_order: 1 lam: 0.09 result: [4.28501311]
# feature: 0 polynomial_order: 1 lam: 0.1 result: [4.28635766]
# feature: 0 polynomial_order: 2 lam: 0.0 result: [4.02467658] # !!!
# feature: 0 polynomial_order: 2 lam: 0.01 result: [4.02559093]
# feature: 0 polynomial_order: 2 lam: 0.02 result: [4.02656024]
# feature: 0 polynomial_order: 2 lam: 0.03 result: [4.02758354]
# feature: 0 polynomial_order: 2 lam: 0.04 result: [4.02865984]
# feature: 0 polynomial_order: 2 lam: 0.05 result: [4.02978816]
# feature: 0 polynomial_order: 2 lam: 0.06 result: [4.03096757]
# feature: 0 polynomial_order: 2 lam: 0.07 result: [4.03219712]
# feature: 0 polynomial_order: 2 lam: 0.08 result: [4.03347589]
# feature: 0 polynomial_order: 2 lam: 0.09 result: [4.03480296]
# feature: 0 polynomial_order: 2 lam: 0.1 result: [4.03617743]
# polynomial_order 3
# feature: 0 polynomial_order: 3 lam: 0 result: [1.07852355e+08]
# feature: 0 polynomial_order: 3 lam: 20 result: [7.16079608]
# feature: 0 polynomial_order: 3 lam: 40 result: [5.98771428]
# feature: 0 polynomial_order: 3 lam: 60 result: [6.02795894]
# feature: 0 polynomial_order: 3 lam: 80 result: [6.03063491]
# feature: 0 polynomial_order: 3 lam: 100 result: [6.03438616]
# feature: 0 polynomial_order: 3 lam: 120 result: [6.05072868]
# feature: 0 polynomial_order: 3 lam: 140 result: [6.08145809]
# feature: 0 polynomial_order: 3 lam: 160 result: [6.12517968]
# feature: 0 polynomial_order: 3 lam: 180 result: [6.18069472]
# feature: 0 polynomial_order: 3 lam: 200 result: [6.2473764]

# feature: 1 polynomial_order: 1 lam: 0.0 result: [4.14470244]
# feature: 1 polynomial_order: 1 lam: 0.01 result: [4.14624393]
# feature: 1 polynomial_order: 1 lam: 0.02 result: [4.14784927]
# feature: 1 polynomial_order: 1 lam: 0.03 result: [4.14951726]
# feature: 1 polynomial_order: 1 lam: 0.04 result: [4.15124669]
# feature: 1 polynomial_order: 1 lam: 0.05 result: [4.15303638]
# feature: 1 polynomial_order: 1 lam: 0.06 result: [4.15488519]
# feature: 1 polynomial_order: 1 lam: 0.07 result: [4.15679197]
# feature: 1 polynomial_order: 1 lam: 0.08 result: [4.15875558]
# feature: 1 polynomial_order: 1 lam: 0.09 result: [4.16077492]
# feature: 1 polynomial_order: 1 lam: 0.1 result: [4.1628489]
# feature: 1 polynomial_order: 2 lam: 0.0 result: [3.88198768] # !!! OOPS. new combination
# feature: 1 polynomial_order: 2 lam: 0.01 result: [3.88283672]
# feature: 1 polynomial_order: 2 lam: 0.02 result: [3.8837323]
# feature: 1 polynomial_order: 2 lam: 0.03 result: [3.88467363]
# feature: 1 polynomial_order: 2 lam: 0.04 result: [3.88565989]
# feature: 1 polynomial_order: 2 lam: 0.05 result: [3.88669028]
# feature: 1 polynomial_order: 2 lam: 0.06 result: [3.88776403]
# feature: 1 polynomial_order: 2 lam: 0.07 result: [3.88888036]
# feature: 1 polynomial_order: 2 lam: 0.08 result: [3.89003851]
# feature: 1 polynomial_order: 2 lam: 0.09 result: [3.89123773]
# feature: 1 polynomial_order: 2 lam: 0.1 result: [3.89247729]
# polynomial_order 3
# feature: 1 polynomial_order: 3 lam: 0 result: [3194382.74539733]
# feature: 1 polynomial_order: 3 lam: 20 result: [5.7186364]
# feature: 1 polynomial_order: 3 lam: 40 result: [5.90900594]
# feature: 1 polynomial_order: 3 lam: 60 result: [5.99867152]
# feature: 1 polynomial_order: 3 lam: 80 result: [6.04795367]
# feature: 1 polynomial_order: 3 lam: 100 result: [6.09143702]
# feature: 1 polynomial_order: 3 lam: 120 result: [6.13967847]
# feature: 1 polynomial_order: 3 lam: 140 result: [6.19509489]
# feature: 1 polynomial_order: 3 lam: 160 result: [6.25749434]
# feature: 1 polynomial_order: 3 lam: 180 result: [6.3257749]
# feature: 1 polynomial_order: 3 lam: 200 result: [6.3988287]