import numpy as np
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
    print(labels_test.shape)
    return s[0]/labels_test.shape[1]
