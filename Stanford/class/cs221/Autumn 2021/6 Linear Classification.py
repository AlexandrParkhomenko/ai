# Linear Classification

import numpy as np
###############################################################################
# Optimization problem
###############################################################################
trainExamples = [
    # (x,y) pairs
    (( 0,  2),  1),
    ((-2,  0),  1),
    (( 1, -1), -1)
    ]
def phi(x):
    return np.array(x)

def initialWeightVector():
    return np.zeros(2)

# Hinge loss
def trainLoss(w): # x, y
    return 1.0 / len(trainExamples) * sum(max( (1 - w.dot(phi(x))*y), 0)
                                          for x, y in trainExamples )
def gradientTrainLoss(w): # x, y
    return 1.0 / len(trainExamples) * sum(-phi(x)*y 
                                          if 1 - w.dot(phi(x))*y > 0 else 0
                                          for x, y in trainExamples )

###############################################################################
# Optimization algorithm
###############################################################################
def gradientDescent(F, gradientF, initialWeightVector):
    w = initialWeightVector()
    eta = 0.1
    for t in range(10):
        value = F(w)
        gradient = gradientF(w)
        w = w - eta * gradient
        print(f'epoch {t}: w = {w}, F(w) = {value}, gradientF = {gradient}')
    
gradientDescent(trainLoss,gradientTrainLoss,initialWeightVector)


'''
Prediction f_w(x)      score                  sign(score)
Relate to target y     resudial(score - y)    margin(score y)

                                              zero-one
Loss function          squared                hinge
                       absolute deviation     logistic

'''




