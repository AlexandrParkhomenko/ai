# This code file pertains to problems 3 onwards from homework 8
# Here, we use out-of-the-box neural network frameworks Keras and Tensorflow 
# to build and train our models

import pdb
import numpy as np
import itertools

import math as m 

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Conv1D, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras import utils as np_utils  # 
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import VarianceScaling
from matplotlib import pyplot as plt

######################################################################
# Problem 3 - 2D data
######################################################################

def archs(classes):
    return [[Dense(input_dim=2, units=classes, activation="softmax")],
            [Dense(input_dim=2, units=10, activation='relu'),
             Dense(units=classes, activation="softmax")],
            [Dense(input_dim=2, units=100, activation='relu'),
             Dense(units=classes, activation="softmax")],
            [Dense(input_dim=2, units=10, activation='relu'),
             Dense(units=10, activation='relu'),
             Dense(units=classes, activation="softmax")],
            [Dense(input_dim=2, units=100, activation='relu'),
             Dense(units=100, activation='relu'),
             Dense(units=classes, activation="softmax")]]

# Read the simple 2D dataset files
def get_data_set(name):
    try:
        data = np.loadtxt(name, skiprows=0, delimiter = ' ')
    except:
        return None, None, None
    np.random.shuffle(data)             # shuffle the data
    # The data uses ROW vectors for a data point, that's what Keras assumes.
    _, d = data.shape
    X = data[:,0:d-1]
    Y = data[:,d-1:d]
    y = Y.T[0]
    classes = set(y)
    if classes == set([-1.0, 1.0]):
        print('Convert from -1,1 to 0,1')
        y = 0.5*(y+1)
    print('Loading X', X.shape, 'y', y.shape, 'classes', set(y))
    return X, y, len(classes)

######################################################################
# General helpers for Problems 3-5
######################################################################

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.keys = ['loss', 'accuracy', 'val_loss', 'val_accuracy']
        self.values = {}
        for k in self.keys:
            self.values['batch_'+k] = []
            self.values['epoch_'+k] = []

    def on_batch_end(self, batch, logs={}):
        for k in self.keys:
            bk = 'batch_'+k
            if k in logs:
                self.values[bk].append(logs[k])

    def on_epoch_end(self, epoch, logs={}):
        for k in self.keys:
            ek = 'epoch_'+k
            if k in logs:
                self.values[ek].append(logs[k])

    def plot(self, keys):
        # print("plot keys", keys)
        # print("self.values", self.values)
        for key in keys:
            plt.plot(np.arange(len(self.values[key])), np.array(self.values[key]), label=key)
        plt.legend()

def run_keras(X_train, y_train, X_val, y_val, X_test, y_test, layers, epochs, split=0, verbose=True):
    # Model specification
    model = Sequential()
    for layer in layers:
        model.add(layer)
    # Define the optimization
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    N = X_train.shape[0]
    # Pick batch size
    batch = 32 if N > 1000 else 1     # batch size
    history = LossHistory()
    # Fit the model
    if X_val is None:
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch, validation_split=split,
                  callbacks=[history], verbose=verbose)
    else:
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch, validation_data=(X_val, y_val),
                  callbacks=[history], verbose=verbose)
    # Evaluate the model on validation data, if any
    if X_val is not None or split > 0:
        # print("HISTORY",history.values)
        if history.values['epoch_val_accuracy'] == []:
            val_accuracy = None
        else:
            val_accuracy  = history.values['epoch_val_accuracy'][-1]
        val_loss = history.values['epoch_val_loss'][-1]
        print ("\nLoss on validation set:"  + str(val_loss) + " Accuracy on validation set: " + str(val_accuracy))
    else:
        val_accuracy = None
    # Evaluate the model on test data, if any
    if X_test is not None:
        test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=batch)
        print ("\nLoss on test set:"  + str(test_loss) + " Accuracy on test set: " + str(test_accuracy))
    else:
        test_accuracy = None
    return model, history, val_accuracy, test_accuracy

def dataset_paths(data_name):
    return ["data/data"+data_name+"_"+suffix+".csv" for suffix in ("train", "validate", "test")]

# The name is a string such as "1" or "Xor"
def run_keras_2d(data_name, layers, epochs, display=True, split=0.25, verbose=True, trials=1):
    print('Keras FC: dataset=', data_name)
    (train_dataset, val_dataset, test_dataset) = dataset_paths(data_name)
    # Load the datasets
    X_train, y, num_classes = get_data_set(train_dataset)
    X_val, y2, _ = get_data_set(val_dataset)
    X_test, y3, _ = get_data_set(test_dataset)
    # Categorize the labels
    y_train = np_utils.to_categorical(y, num_classes) # one-hot
    y_val = y_test = None
    if X_val is not None:
        y_val = np_utils.to_categorical(y2, num_classes) # one-hot        
    if X_test is not None:
        y_test = np_utils.to_categorical(y3, num_classes) # one-hot
    val_accuracy, test_accuracy = 0, 0
    for trial in range(trials):
        # Reset the weights
        # See https://github.com/keras-team/keras/issues/341
        # session = K.get_session()
        # for layer in layers:
        #     for v in layer.__dict__:
        #         v_arg = getattr(layer, v)
        #         if hasattr(v_arg, 'initializer'):
        #             initializer_func = getattr(v_arg, 'initializer')
        #             initializer_func.run(session=session)
        # Run the model
        model, history, vacc, tacc, = \
               run_keras(X_train, y_train, X_val, y_val, X_test, y_test, layers, epochs,
                         split=split, verbose=verbose)
        val_accuracy += vacc if vacc else 0
        test_accuracy += tacc if tacc else 0
        if display:
            # plot classifier landscape on training data
            plot_heat(X_train, y, model)
            plt.title('Training data')
            plt.show()
            if X_test is not None:
                # plot classifier landscape on testing data
                plot_heat(X_test, y3, model)
                plt.title('Testing data')
                plt.show()
            
            # Plot epoch loss
            detailed = False
            if detailed:
                p = 'batch'
            else:
                p = 'epoch'
            history.plot([p+'_loss', p+'_val_loss'])
            plt.xlabel(p)
            plt.ylabel('loss')
            plt.title('Epoch val_loss and loss')
            plt.show()
            # Plot epoch accuracy
            history.plot([p+'_accuracy', p+'_val_accuracy'])
            plt.xlabel(p)
            plt.ylabel('accuracy')
            plt.title('Epoch val_accuracy and accuracy')
            plt.show()
            
    if val_accuracy:
        print ("\nAvg. validation accuracy:"  + str(val_accuracy/trials))
    if test_accuracy:
        print ("\nAvg. test accuracy:"  + str(test_accuracy/trials))
    return X_train, y, model

######################################################################
# Helper functions for 
# OPTIONAL: Problem 4 - Weight Sharing
######################################################################

def generate_1d_images(nsamples,image_size,prob):
    Xs=[]
    Ys=[]
    for i in range(0,nsamples):
        X=np.random.binomial(1, prob, size=image_size)
        Y=count_objects_1d(X)
        Xs.append(X)
        Ys.append(Y)
    Xs=np.array(Xs)
    Ys=np.array(Ys)
    return Xs,Ys


#count the number of objects in a 1d array
def count_objects_1d(array):
    count=0
    for i in range(len(array)):
        num=array[i]
        if num==0:
            if i==0 or array[i-1]==1:
                count+=1
    return count

def l1_reg(weight_matrix):
    return 0.01 * K.sum(K.abs(weight_matrix))    


# def filter_reg(weights):
#     lam=0
#     return lam* val

def get_image_data_1d(tsize,image_size,prob):
    #prob controls the density of white pixels
    #tsize is the size of the training and test sets
    vsize=int(0.2*tsize)
    X_train,Y_train=generate_1d_images(tsize,image_size,prob)
    X_val,Y_val=generate_1d_images(vsize,image_size,prob)
    X_test,Y_test=generate_1d_images(tsize,image_size,prob)
    #reshape the input data for the convolutional layer
    X_train=np.expand_dims(X_train,axis=2)
    X_val=np.expand_dims(X_val,axis=2)
    X_test=np.expand_dims(X_test,axis=2)
    data=(X_train,Y_train,X_val,Y_val,X_test,Y_test)
    return data

def train_neural_counter(layers,data,loss_func='mse',display=True):
    (X_train,Y_train,X_val,Y_val,X_test,Y_test)=data
    epochs=10
    batch=1
    
    model=Sequential()
    for layer in layers:
        model.add(layer)
    model.summary()    
    model.compile(loss=loss_func, optimizer=Adam())
    history = LossHistory()    
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch, validation_data=(X_val, Y_val),callbacks=[history], verbose=True)
    err=model.evaluate(X_test,Y_test)
    ws=model.layers[-1].get_weights()[0]
    # Evaluate the model on validation data, if any
    if X_val is not None or split > 0:
        # print("HISTORY",history.values)
        if history.values['epoch_val_accuracy'] == []:
            val_accuracy = None
        else:
            val_accuracy  = history.values['epoch_val_accuracy'][-1]
        val_loss = history.values['epoch_val_loss'][-1]
        print ("\nLoss on validation set:"  + str(val_loss) + " Accuracy on validation set: " + str(val_accuracy))
    else:
        val_accuracy = None
    if display:
        plt.plot(ws)
        plt.show()
    return model,err

######################################################################
# Problem 5
######################################################################

def shifted(X, shift):
    n = X.shape[0]
    m = X.shape[1]
    size = m + shift
    X_sh = np.zeros((n, size, size))
    plt.ion()
    for i in range(n):
        sh1 = np.random.randint(shift)
        sh2 = np.random.randint(shift)
        X_sh[i, sh1:sh1+m, sh2:sh2+m] = X[i, :, :]
        # If you want to see the shifts, uncomment
        #plt.figure(1); plt.imshow(X[i])
        #plt.figure(2); plt.imshow(X_sh[i])
        #plt.show()
        #input('Go?')
    return X_sh
  
def get_MNIST_data(shift=0):
    (X_train, y1), (X_val, y2) = mnist.load_data()
    if shift:
        size = 28+shift
        X_train = shifted(X_train, shift) #* (1/256.0)
        X_val = shifted(X_val, shift) #* (1/256.0)
    return (X_train, y1), (X_val, y2)

# Example Usage:
# train, validation = get_MNIST_data()

def run_keras_fc_mnist(train, test, layers, epochs, split=0.1, verbose=True, trials=1):
    (X_train, y1), (X_val, y2) = train, test
    # Flatten the images
    m = X_train.shape[1]
    X_train = X_train.reshape((X_train.shape[0], m*m))
    X_val = X_val.reshape((X_val.shape[0], m*m))
    # Categorize the labels
    num_classes = 10
    y_train = np_utils.to_categorical(y1, num_classes)
    y_val = np_utils.to_categorical(y2, num_classes)
    # Train, use split for validation
    val_accuracy, test_accuracy = 0, 0
    for trial in range(trials):
        # Reset the weights
        # See https://github.com/keras-team/keras/issues/341
        # session = K.get_session()
        # for layer in layers:
        #     for v in layer.__dict__:
        #         v_arg = getattr(layer, v)
        #         if hasattr(v_arg, 'initializer'):
        #             initializer_func = getattr(v_arg, 'initializer')
        #             initializer_func.run(session=session)
        # Run the model
        model, history, vacc, tacc = \
                run_keras(X_train, y_train, X_val, y_val, None, None, layers, epochs, split=split, verbose=verbose)
        val_accuracy += vacc if vacc else 0
        test_accuracy += tacc if tacc else 0
    if val_accuracy:
        print ("\nAvg. validation accuracy:"  + str(val_accuracy/trials))
    if test_accuracy:
        print ("\nAvg. test accuracy:"  + str(test_accuracy/trials))

def run_keras_cnn_mnist(train, test, layers, epochs, split=0.1, verbose=True, trials=1):
    # Load the dataset
    (X_train, y1), (X_val, y2) = train, test
    # Add a final dimension indicating the number of channels (only 1 here)
    m = X_train.shape[1]
    X_train = X_train.reshape((X_train.shape[0], m, m, 1))
    X_val = X_val.reshape((X_val.shape[0], m, m, 1))
    # Categorize the labels
    num_classes = 10
    y_train = np_utils.to_categorical(y1, num_classes)
    y_val = np_utils.to_categorical(y2, num_classes)
    # Train, use split for validation
    val_accuracy, test_accuracy = 0, 0
    for trial in range(trials):
        # Reset the weights
        # See https://github.com/keras-team/keras/issues/341
        # session = K.get_session()
        # for layer in layers:
        #     for v in layer.__dict__:
        #         v_arg = getattr(layer, v)
        #         if hasattr(v_arg, 'initializer'):
        #             initializer_func = getattr(v_arg, 'initializer')
        #             initializer_func.run(session=session)
        # Run the model
        model, history, vacc, tacc = \
                run_keras(X_train, y_train, X_val, y_val, None, None, layers, epochs, split=split, verbose=verbose)
        val_accuracy += vacc if vacc else 0
        test_accuracy += tacc if tacc else 0
    if val_accuracy:
        print ("\nAvg. validation accuracy:"  + str(val_accuracy/trials))
    if test_accuracy:
        print ("\nAvg. test accuracy:"  + str(test_accuracy/trials))

# Example usage:
# train, validation = get_MNIST_data()
# layers = [Dense(input_dim=???, units=???, activation='softmax')]
# run_keras_fc_mnist(train, validation, layers, 1, split=0.1, verbose=True, trials=5)
# Same pattern applies to the function: run_keras_cnn_mnist
epochs = 1
train, validation = get_MNIST_data()
# layers = [Dense(input_dim=784, units=10, activation='softmax')]
# Avg. validation accuracy:0.8876599907875061 #############oops not 0.6
layers = [Dense(input_dim=784, units=10, activation='softmax',
                kernel_initializer=VarianceScaling(scale=0.001, mode='fan_in', distribution='normal', seed=None))]
# Avg. validation accuracy:0.8814800024032593
run_keras_fc_mnist(train, validation, layers, epochs, split=0.1, trials=5)
# imsize = 784
# batch = 1
# layers = [Conv2D(filters=1, kernel_size=2, strides=1,use_bias=False,activation='relu',batch_size=batch,input_shape=(imsize,1),padding='same')
          
#           ]
# run_keras_cnn_mnist(train, validation, layers, epochs, split=0.1, verbose=True, trials=5)

######################################################################
# Plotting Functions
######################################################################

def plot_heat(X, y, model, res = 200):
    eps = .1
    xmin = np.min(X[:,0]) - eps; xmax = np.max(X[:,0]) + eps
    ymin = np.min(X[:,1]) - eps; ymax = np.max(X[:,1]) + eps
    ax = tidyPlot(xmin, xmax, ymin, ymax, xlabel = 'x', ylabel = 'y')
    xl = np.linspace(xmin, xmax, res)
    yl = np.linspace(ymin, ymax, res)
    xx, yy = np.meshgrid(xl, yl, sparse=False)
    zz = np.argmax(model.predict(np.c_[xx.ravel(), yy.ravel()]), axis=1)
    im = ax.imshow(np.flipud(zz.reshape((res,res))), interpolation = 'none',
                   extent = [xmin, xmax, ymin, ymax],
                   cmap = 'viridis')
    plt.colorbar(im)
    for yi in set([int(_y) for _y in set(y)]):
        color = ['r', 'g', 'b'][yi]
        marker = ['X', 'o', 'v'][yi]
        cl = np.where(y==yi)
        ax.scatter(X[cl,0], X[cl,1], c = color, marker = marker, s=80,
                   edgecolors = 'none')
    return ax

def tidyPlot(xmin, xmax, ymin, ymax, center = False, title = None,
                 xlabel = None, ylabel = None):
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

def plot_decision(data, cl, diff=False):
    layers = archs(cl)[0]
    X, y, model = run_keras_2d(data, layers, 10, trials=1, verbose=False, display=False)
    ax = plot_heat(X,y,model)
    W = layers[0].get_weights()[0]
    W0 = layers[0].get_weights()[1].reshape((cl,1))
    if diff:
        for i,j in list(itertools.combinations(range(cl),2)):
            plot_separator(ax, W[:,i:i+1] - W[:,j:j+1], W0[i:i+1,:] - W0[j:j+1,:])
    else:
        for i in range(cl):
            plot_separator(ax, W[:,i:i+1], W0[i:i+1,:])
    plt.show()


# run_keras_2d('3class', archs(3)[5-1], epochs=10, split=0.5, display=True, trials=5)
