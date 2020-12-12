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

def averaged_perceptron(data, labels, params={}, hook=None):
    # if T not in params, default to 100
    T = params.get('T', 100)
    N = data.shape[1]
    theta  = np.zeros((data.shape[0],1))
    theta0 = np.zeros(1)
    ths  = np.zeros((data.shape[0],1))
    th0s = np.zeros(1)
    #print("theta:",theta,theta0)
    for t in range(T):
        for n in range(N):
            x = data[:,n]
            y = labels[0,n]
            if y*(np.dot(x,theta)+theta0) <= 0:
                theta[:,0] = theta[:,0]+ y*x # my error was here
                theta0 = theta0 + y
            ths  += theta
            th0s += theta0
#    return (ths/(N*T),rv(th0s/(N*T)))
    return (ths/(N*T),th0s/(N*T))

def rv(value_list):
    return np.array([value_list])
def positive(x, th, th0):
    return np.sign(np.matmul(th.T,x)+th0)
def score(data, labels, th, th0):
    labels_multy = np.ones((labels.shape[1],th.shape[1])).T * labels
    return np.sum(positive(data, th, th0)==labels_multy, axis=1)
def test_score():
    data = np.transpose(np.array([[1, 2], [1, 3], [2, 1], [1, -1], [2, -1]]))
    labels = rv([-1, -1, +1, +1, +1])
    ths = np.array([[ 0.98645534, -0.02061321, -0.30421124, -0.62960452,  0.61617711,  0.17344772, -0.21804797, 0.26093651, 0.47179699, 0.32548657], [ 0.87953335, 0.39605039, -0.1105264, 0.71212565, -0.39195678, 0.00999743, -0.88220145, -0.73546501, -0.7769778, -0.83807759]])
    th0s = np.array([[ 0.65043158, 0.61626967, 0.84632592, -0.43047804, -0.91768579, -0.3214327, 0.0682113, -0.20678004, -0.33963784, 0.74308104]])
    r = np.argmax(score(data, labels, ths, th0s.T))
    print("r",r)
    
def eval_classifier(learner, data_train, labels_train, data_test, labels_test):
    thetas = learner(data_train, labels_train, params = {}, hook = None)
    #print(thetas)
    #learner(data_test,  labels_test,  params = {}, hook = None)
    #labels = positive(x, th, th0) #
    s = score(data_test,  labels_test, thetas[0], thetas[1])
    #print(labels_test.shape)
    return s[0]/labels_test.shape[1]

def eval_learning_alg(learner, data_gen, n_train, n_test, it):
    a = 0
    for i in range(it):
        data_train, labels_train = data_gen(n_train)
        data_test, labels_test = data_gen(n_test) # data_train, labels_train #  # modifiations
        a += eval_classifier(learner, data_train, labels_train, data_test, labels_test)
    return a/it

#gen_flipped_lin_separable(num_points=20, pflip=0.25, th=np.array([[3],[4]]), th_0=np.array([[0]]), dim=2)

n_train, n_test, it = 20, 20, 20
print(eval_learning_alg(perceptron, gen_flipped_lin_separable(pflip = 0.1), n_train, n_test, it))
print(eval_learning_alg(averaged_perceptron, gen_flipped_lin_separable(pflip = 0.1), n_train, n_test, it))
#print(eval_learning_alg(perceptron, gen_flipped_lin_separable(pflip = 0.25), n_train, n_test, it))
#print(eval_learning_alg(averaged_perceptron, gen_flipped_lin_separable(pflip = 0.25), n_train, n_test, it))
