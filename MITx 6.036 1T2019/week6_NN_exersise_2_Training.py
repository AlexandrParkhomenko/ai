
# Training
# Simple in form, but not in content

import numpy as np

def hinge_loss_grad(x, y, a):
    # https://stats.stackexchange.com/questions/4608/gradient-of-hinge-loss
    return np.where(y*a < 1, -y*x, 0)

step = 0.5
x = np.array([1,1,2])
y = np.array([-1])
th = np.array([1,1,1])

th = th-step*hinge_loss_grad(x, y, np.dot(th.T,x))
print(th)
th = th-step*hinge_loss_grad(x, y, np.dot(th.T,x))
print(th)
th = th-step*hinge_loss_grad(x, y, np.dot(th.T,x))
print(th)
