'''
Code for MIT 6.036 Homework 2
'''
# Implement perceptron, average perceptron

import pdb
import operator
import itertools

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import colors

print("Importing code_for_hw02")

######################################################################
# Plotting

def tidy_plot(xmin, xmax, ymin, ymax, center = False, title = None,
                 xlabel = None, ylabel = None):
    '''
    Set up axes for plotting
    xmin, xmax, ymin, ymax = (float) plot extents
    Return matplotlib axes
    '''
    plt.ion()
    plt.figure(facecolor="white")
    ax = plt.subplot()
    if center:
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_color('none')
        ax.spines['left'].set_smart_bounds(True)
        ax.spines['bottom'].set_smart_bounds(True)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    else:
        ax.spines["top"].set_visible(False)    
        ax.spines["right"].set_visible(False)    
        ax.get_xaxis().tick_bottom()  
        ax.get_yaxis().tick_left()
    eps = .05
    plt.xlim(xmin-eps, xmax+eps)
    plt.ylim(ymin-eps, ymax+eps)
    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    return ax

def plot_separator(ax, th, th_0):
    '''
    Plot separator in 2D
    ax = (matplotlib plot) plot axis
    th = (numpy array) theta
    th_0 = (float) theta_0
    '''
    xmin, xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    pts = []
    eps = 1.0e-6
    # xmin boundary crossing is when xmin th[0] + y th[1] + th_0 = 0
    # that is, y = (-th_0 - xmin th[0]) / th[1]
    if abs(th[1,0]) > eps:
        pts += [np.array([x, (-th_0 - x * th[0,0]) / th[1,0]]) \
                                                        for x in (xmin, xmax)]
    if abs(th[0,0]) > 1.0e-6:
        pts += [np.array([(-th_0 - y * th[1,0]) / th[0,0], y]) \
                                                         for y in (ymin, ymax)]
    in_pts = []
    for p in pts:
        if (xmin-eps) <= p[0] <= (xmax+eps) and \
           (ymin-eps) <= p[1] <= (ymax+eps):
            duplicate = False
            for p1 in in_pts:
                if np.max(np.abs(p - p1)) < 1.0e-6:
                    duplicate = True
            if not duplicate:
                in_pts.append(p)
    if in_pts and len(in_pts) >= 2:
        # Plot separator
        vpts = np.vstack(in_pts)
        ax.plot(vpts[:,0], vpts[:,1], 'k-', lw=2)
        # Plot normal
        vmid = 0.5*(in_pts[0] + in_pts[1])
        scale = np.sum(th*th)**0.5
        diff = in_pts[0] - in_pts[1]
        dist = max(xmax-xmin, ymax-ymin)        
        vnrm = vmid + (dist/10)*(th.T[0]/scale)
        vpts = np.vstack([vmid, vnrm])
        ax.plot(vpts[:,0], vpts[:,1], 'k-', lw=2)
        # Try to keep limits from moving around
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((ymin, ymax))
    else:
        print('Separator not in plot range')

def plot_data(data, labels, ax = None, clear = False,
                  xmin = None, xmax = None, ymin = None, ymax = None):
    '''
    Make scatter plot of data.
    data = (numpy array)
    ax = (matplotlib plot)
    clear = (bool) clear current plot first
    xmin, xmax, ymin, ymax = (float) plot extents
    returns matplotlib plot on ax 
    '''
    if ax is None:
        if xmin == None: xmin = np.min(data[0, :]) - 0.5
        if xmax == None: xmax = np.max(data[0, :]) + 0.5
        if ymin == None: ymin = np.min(data[1, :]) - 0.5
        if ymax == None: ymax = np.max(data[1, :]) + 0.5
        ax = tidy_plot(xmin, xmax, ymin, ymax)

        x_range = xmax - xmin; y_range = ymax - ymin
        if .1 < x_range / y_range < 10:
            ax.set_aspect('equal')
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
    elif clear:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.clear()
    else:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
    colors = np.choose(labels > 0, cv(['r', 'g']))[0]
    ax.scatter(data[0,:], data[1,:], c = colors,
                    marker = 'o', s=50, edgecolors = 'none')
    # Seems to occasionally mess up the limits
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.grid(True, which='both')
    #ax.axhline(y=0, color='k')
    #ax.axvline(x=0, color='k')
    return ax

def plot_nonlin_sep(predictor, ax = None, xmin = None , xmax = None,
                        ymin = None, ymax = None, res = 30):
    '''
    Must either specify limits or existing ax
    Shows matplotlib plot on ax
    '''
    if ax is None:
        ax = tidy_plot(xmin, xmax, ymin, ymax)
    else:
        if xmin == None:
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
        else:
            ax.set_xlim((xmin, xmax))
            ax.set_ylim((ymin, ymax))

    cmap = colors.ListedColormap(['black', 'white'])
    bounds=[-2,0,2]
    norm = colors.BoundaryNorm(bounds, cmap.N)            
            
    ima = np.array([[predictor(x1i, x2i) \
                         for x1i in np.linspace(xmin, xmax, res)] \
                         for x2i in np.linspace(ymin, ymax, res)])
    im = ax.imshow(np.flipud(ima), interpolation = 'none',
                       extent = [xmin, xmax, ymin, ymax],
                       cmap = cmap, norm = norm)

######################################################################
#   Utilities

def cv(value_list):
    '''
    Takes a list of numbers and returns a column vector:  n x 1
    '''
    return np.transpose(rv(value_list))

def rv(value_list):
    '''
    Takes a list of numbers and returns a row vector: 1 x n
    '''
    return np.array([value_list])

def y(x, th, th0):
    '''
    x is dimension d by 1
    th is dimension d by 1
    th0 is a scalar
    return a 1 by 1 matrix
    '''
    return np.dot(np.transpose(th), x) + th0

def positive(x, th, th0):
    '''
    x is dimension d by 1
    th is dimension d by 1
    th0 is dimension 1 by 1
    return 1 by 1 matrix of +1, 0, -1
    '''
    return np.sign(y(x, th, th0))

def score(data, labels, th, th0):
    '''
    data is dimension d by n
    labels is dimension 1 by n
    ths is dimension d by 1
    th0s is dimension 1 by 1
    return 1 by 1 matrix of integer indicating number of data points correct for
    each separator.
    '''
    return np.sum(positive(data, th, th0) == labels)

######################################################################
#   Data Sets

def super_simple_separable_through_origin():
    '''
    Return d = 2 by n = 4 data matrix and 1 x n = 4 label matrix
    '''
    X = np.array([[2, 3, 9, 12],
                  [5, 1, 6, 5]])
    y = np.array([[1, -1, 1, -1]])
    return X, y

def super_simple_separable():
    '''
    Return d = 2 by n = 4 data matrix and 1 x n = 4 label matrix
    '''
    X = np.array([[2, 3, 9, 12],
                  [5, 2, 6, 5]])
    y = np.array([[1, -1, 1, -1]])
    return X, y

def xor():
    '''
    Return d = 2 by n = 4 data matrix and 1 x n = 4 label matrix
    '''
    X = np.array([[1, 2, 1, 2],
                  [1, 2, 2, 1]])
    y = np.array([[1, 1, -1, -1]])
    return X, y

def xor_more():
    '''
    Return d = 2 by n = 4 data matrix and 1 x n = 4 label matrix
    '''
    X = np.array([[1, 2, 1, 2, 2, 4, 1, 3],
                  [1, 2, 2, 1, 3, 1, 3, 3]])
    y = np.array([[1, 1, -1, -1, 1, 1, -1, -1]])
    return X, y

# Test data for problem 2.1
data1, labels1, data2, labels2 = \
(np.array([[-2.97797707,  2.84547604,  3.60537239, -1.72914799, -2.51139524,
         3.10363716,  2.13434789,  1.61328413,  2.10491257, -3.87099125,
         3.69972003, -0.23572183, -4.19729119, -3.51229538, -1.75975746,
        -4.93242615,  2.16880073, -4.34923279, -0.76154262,  3.04879591,
        -4.70503877,  0.25768309,  2.87336016,  3.11875861, -1.58542576,
        -1.00326657,  3.62331703, -4.97864369, -3.31037331, -1.16371314],
       [ 0.99951218, -3.69531043, -4.65329654,  2.01907382,  0.31689211,
         2.4843758 , -3.47935105, -4.31857472, -0.11863976,  0.34441625,
         0.77851176,  1.6403079 , -0.57558913, -3.62293005, -2.9638734 ,
        -2.80071438,  2.82523704,  2.07860509,  0.23992709,  4.790368  ,
        -2.33037832,  2.28365246, -1.27955206, -0.16325247,  2.75740801,
         4.48727808,  1.6663558 ,  2.34395397,  1.45874837, -4.80999977]]), 
 np.array([[-1., -1., -1., -1., -1., -1.,  1.,  1.,  1., -1., -1., -1., -1.,
        -1.,  1., -1.,  1., -1., -1., -1.,  1.,  1.,  1.,  1.,  1., -1.,
        -1., -1., -1., -1.]]), np.array([[ 0.6894022 , -4.34035772,  3.8811067 ,  4.29658177,  1.79692041,
         0.44275816, -3.12150658,  1.18263462, -1.25872232,  4.33582168,
         1.48141202,  1.71791177,  4.31573568,  1.69988085, -2.67875489,
        -2.44165649, -2.75008176, -4.19299345, -3.15999758,  2.24949368,
         4.98930636, -3.56829885, -2.79278501, -2.21547048,  2.4705776 ,
         4.80481986,  2.77995092,  1.95142828,  4.48454942, -4.22151738],
       [-2.89934727,  1.65478851,  2.99375325,  1.38341854, -4.66701003,
        -2.14807131, -4.14811829,  3.75270334,  4.54721208,  2.28412663,
        -4.74733482,  2.55610647,  3.91806508, -2.3478982 ,  4.31366925,
        -0.92428271, -0.84831235, -3.02079092,  4.85660032, -1.86705397,
        -3.20974025, -4.88505017,  3.01645974,  0.03879148, -0.31871427,
         2.79448951, -2.16504256, -3.91635569,  3.81750006,  4.40719702]]),
 np.array([[-1., -1.,  1.,  1., -1., -1., -1.,  1.,  1.,  1., -1.,  1.,  1.,
        -1.,  1.,  1.,  1., -1., -1., -1.,  1., -1.,  1., -1.,  1., -1.,
        -1.,  1.,  1.,  1.]]))

# Test data for problem 2.2
big_data, big_data_labels = (np.array([[-2.04297103, -1.85361169, -2.65467827, -1.23013149, -0.31934782,
         1.33112127,  2.3297942 ,  1.47705445, -1.9733787 , -2.35476882,
        -4.97193554,  3.49851995,  4.00302943,  0.83369183,  0.41371989,
         4.37614714,  1.03536965,  1.2354608 , -0.7933465 , -3.85456759,
         3.22134658, -3.39787483, -1.31182253, -2.61363628, -1.14618119,
        -0.2174626 ,  1.32549116,  2.54520221,  0.31565661,  2.24648287,
        -3.33355258, -0.98689271, -0.24876636, -3.16008017,  1.22353111,
         4.77766994, -1.81670773, -3.58939471, -2.16268851,  2.88028351,
        -3.42297827, -2.74992813, -0.40293356, -3.45377267,  0.62400624,
        -0.35794507, -4.1648704 , -1.08734116,  0.22367444,  1.09067619,
         1.28738004,  2.07442478,  4.61951855,  4.47029706,  2.86510481,
         4.12532285,  0.48170777,  0.60089857,  4.50287515,  2.95549453,
         4.22791451, -1.28022286,  2.53126681,  2.41887277, -4.9921717 ,
         4.15022718,  0.49670572,  2.0268248 , -4.63475897, -4.20528418,
         1.77013481, -3.45389325,  1.0238472 , -1.2735185 ,  4.75384686,
         1.32622048, -0.13092625,  1.23457116, -1.69515197,  2.82027615,
        -1.01140935,  3.36451016,  4.43762708, -4.2679604 ,  4.76734154,
        -4.14496071, -4.38737405, -1.13214501, -2.89008477,  3.22986894,
         1.84103699, -3.91906092, -2.8867831 ,  2.31059245, -3.62773189,
        -4.58459406, -4.06343392, -3.10927054,  1.09152472,  2.99896855],
       [-2.1071566 , -3.06450052, -3.43898434,  0.71320285,  1.51214693,
         4.14295175,  4.73681233, -2.80366981,  1.56182223,  0.07061724,
        -0.92053415, -3.61953464,  0.39577344, -3.03202474, -4.90408303,
        -0.10239158, -1.35546287,  1.31372748, -1.97924525, -3.72545813,
         1.84834303, -0.13679709,  1.36748822, -2.92886952, -2.48367819,
        -0.0894489 , -2.99090327,  0.35494698,  0.94797491,  4.20393035,
        -3.14009852, -4.86292242,  3.2964068 , -0.9911453 ,  4.39465   ,
         3.64956975, -0.72225648, -0.15864119, -2.0340774 , -4.00758749,
         0.8627915 ,  3.73237594, -0.70011824,  1.07566463, -4.05063547,
        -3.98137177,  4.82410619,  2.5905222 ,  0.34188269, -1.44737803,
         3.27583966,  2.06616486, -4.43584161,  0.27795053,  4.37207651,
        -4.48564119,  0.7183541 ,  1.59374552, -0.13951634,  0.67825519,
        -4.02423434,  4.15893861, -1.52110278,  2.1320374 ,  3.31118893,
        -4.04072252,  2.41403912, -1.04635499,  3.39575642,  2.2189097 ,
         4.78827245,  1.19808069,  3.10299723,  0.18927394,  0.14437543,
        -4.17561642,  0.6060279 ,  0.22693751, -3.39593567,  1.14579319,
         3.65449494, -1.27240159,  0.73111639,  3.48806017,  2.48538719,
        -1.83892096,  1.42819622, -1.37538641,  3.4022984 ,  0.82757044,
        -3.81792516,  2.77707152, -1.49241173,  2.71063994, -3.33495679,
        -4.00845675,  0.719904  , -2.3257032 ,  1.65515972, -1.90859948]]), np.array([[-1., -1., -1.,  1.,  1., -1.,  1., -1.,  1., -1., -1., -1.,  1.,
         1., -1.,  1., -1., -1., -1.,  1., -1., -1.,  1., -1.,  1., -1.,
        -1.,  1.,  1., -1., -1., -1.,  1., -1.,  1.,  1.,  1., -1., -1.,
         1., -1.,  1., -1.,  1., -1.,  1.,  1.,  1.,  1., -1.,  1.,  1.,
        -1.,  1.,  1., -1., -1.,  1.,  1.,  1., -1.,  1.,  1., -1., -1.,
         1.,  1.,  1.,  1., -1.,  1., -1.,  1., -1., -1., -1.,  1.,  1.,
        -1.,  1.,  1.,  1.,  1., -1., -1.,  1., -1., -1., -1.,  1., -1.,
        -1., -1.,  1., -1., -1., -1., -1.,  1.,  1.]]))

def gen_big_data():
    '''
    Return method that generates a dataset of input size n of X, y drawn from big_data
    '''
    nd = big_data.shape[1]
    current = [0]
    def f(n):
        cur = current[0]
        vals = big_data[:,cur:cur+n], big_data_labels[:,cur:cur+n]
        current[0] += n
        if current[0] >= nd: current[0] = 0
        return vals
    return f

def gen_lin_separable(num_points=20, th=np.array([[3],[4]]), th_0=np.array([[0]]), dim=2):
    ''' 
    Generate linearly separable dataset X, y given theta and theta0
    Return X, y where
    X is a numpy array where each column represents a dim-dimensional data point
    y is a column vector of 1s and -1s
    '''
    X = np.random.uniform(low=-5, high=5, size=(dim, num_points))
    y = np.sign(np.dot(np.transpose(th), X) + th_0)
    return X, y

def big_higher_dim_separable():
    X, y = gen_lin_separable(num_points=50, dim=6, th=np.array([[3],[4],[2],[1],[0],[3]]))
    return X, y

def gen_flipped_lin_separable(num_points=20, pflip=0.25, th=np.array([[3],[4]]), th_0=np.array([[0]]), dim=2):
    '''
    Generate difficult (usually not linearly separable) data sets by
    "flipping" labels with some probability.
    Return method which takes num_points and flips labels with pflip
    '''
    def flip_generator(num_points=20):
        X, y = gen_lin_separable(num_points, th, th_0, dim)
        flip = np.random.uniform(low=0, high=1, size=(num_points,))
        for i in range(num_points):
            if flip[i] < pflip: y[0,i] = -y[0,i]
        return X, y
    return flip_generator

######################################################################
#   tests

def test_linear_classifier(dataFun, learner, learner_params = {},
                             draw = False, refresh = False, pause = False):
    '''
    Prints score of your classifier on given dataset
    dataFun method that returns a dataset
    learner your classifier method
    learner_params parameters for the learner
    '''
    data, labels = dataFun()
    d, n = data.shape
    if draw:
        ax = plot_data(data, labels)
        def hook(params):
            (th, th0) = params
            if refresh: plot_data(data, labels, ax, clear = True)
            plot_separator(ax, th, th0)
            #print('th', th.T, 'th0', th0)
            plt.pause(0.05)
            if pause: input('go?')
    else:
        hook = None
    th, th0 = learner(data, labels, hook = hook, params = learner_params)
    print("Final score", float(score(data, labels, th, th0)) / n)
    print("Params", np.transpose(th), th0)

expected_perceptron = [(np.array([[-9.0], [18.0]]), np.array([[2.0]])),(np.array([[0.0], [-3.0]]), np.array([[0.0]]))]
expected_averaged = [(np.array([[-9.0525], [17.5825]]), np.array([[1.9425]])),(np.array([[1.47], [-1.7275]]), np.array([[0.985]]))]
datasets = [super_simple_separable_through_origin,xor]

def incorrect(expected,result):
    print("Test Failed.")
    print("Your code output ",result)
    print("Expected ",expected)
    print("\n")

def correct():
    print("Passed! \n")


def test_perceptron(perceptron):
    '''
    Checks perceptron theta and theta0 values for 100 iterations
    '''
    for index in range(len(datasets)):
        data, labels = datasets[index]()
        th,th0 = perceptron(data, labels, {"T": 100})
        expected_th,expected_th0 = expected_perceptron[index]
        print("-----------Test Perceptron "+str(index)+"-----------")
        if((th==expected_th).all() and (th0==expected_th0).all()):
            correct()
        else:
            incorrect("th: "+str(expected_th.tolist())+", th0: "+str(expected_th0.tolist()), "th: "+str(th.tolist())+", th0: "+str(th0.tolist()))
    
def test_averaged_perceptron(averaged_perceptron):
    '''
    Checks average perceptron theta and theta0 values for 100 iterations
    '''
    for index in range(2):
        data, labels = datasets[index]()
        th,th0 = averaged_perceptron(data, labels, {"T": 100})
        expected_th,expected_th0 = expected_averaged[index]
        print("-----------Test Averaged Perceptron "+str(index)+"-----------")
        if((th==expected_th).all() and (th0==expected_th0).all()):
            correct()
        else:
            incorrect("th: "+str(expected_th.tolist())+", th0: "+str(expected_th0.tolist()), "th: "+str(th.tolist())+", th0: "+str(th0.tolist()))
        
def test_eval_classifier(eval_classifier,perceptron):
    '''
    Checks your classifier's performance on data1
    '''
    expected = [0.5333333333333333,0.6333333333333333]
    dataset_train = [(data1,labels1),(data2,labels2)]
    for index in range(len(dataset_train)):
        data_train,labels_train = dataset_train[index]
        #print(data_train,labels_train)
        result = eval_classifier(perceptron, data_train, labels_train,data2,labels2)
        print("-----------Test Eval Classifier "+str(index)+"-----------")
        if(result==expected[index]):
            correct()
        else:
            incorrect(expected[index],result)

def test_eval_learning_alg(eval_learning_alg,perceptron):
    '''
    Checks your learning algorithm's performance on big_data
    eval_learning_alg method for evaluating learning algorithm
    perceptron your perceptron learning algorithm method
    '''
    expected = 0.5599999999999999
    result = eval_learning_alg(perceptron, gen_big_data(), 10, 10, 5)
    print("-----------Test Eval Learning Algo-----------")
    if result == expected:
        correct()
    else:
        incorrect(expected,result)
        
def test_xval_learning_alg(xval_learning_alg,perceptron):
    '''
    Checks your learning algorithm's performance on big_data using cross validation
    xval_learning_alg method for evaluating learning algorithm using cross validation
    perceptron your perceptron learning algorithm method
    '''
    expected = 0.61
    result=xval_learning_alg(perceptron, big_data, big_data_labels, 5)
    print("-----------Test Cross-eval Learning Algo-----------")
    if result == expected:
        correct()
    else:
        incorrect(expected,result)
    
#-----------------------------------------------------------------------------

print("New procedures added: tidy_plot, plot_separator, plot_data, plot_nonlin_sep, cv,")
print("                      rv, y, positive, score")
print("Data Sets: super_simple_separable_through_origin(), super_simple_separable(), xor(),")
print("           xor_more()")
print("Test data for problem 2.1: data1, labels1, data2, labels2")
print("Test data for problem 2.2: big_data, big_data_labels, gen_big_data(), gen_lin_separable(),")
print("                           big_higher_dim_separable(), gen_flipped_lin_separable()")
print("Test functions: test_linear_classifier(), test_perceptron(), test_averaged_perceptron(),")
print("                test_eval_classifier(), test_eval_learning_alg(), test_xval_learning_alg()")
print("")
print("For more information, use 'help', e.g. 'help tidy_plot'")

print("Done with import of code_for_hw02")





