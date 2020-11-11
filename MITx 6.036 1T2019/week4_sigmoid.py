# see notes_chapter_Logistic_regression.pdf

import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import colors

# for a list of floating point numbers. Ayush https://stackoverflow.com/questions/7267226/range-for-floats
from itertools import count, takewhile
def frange(start, stop, step):
    return takewhile(lambda x: x< stop, count(start, step))

def sigmoid(x0):
    return 1.0/(1.0+np.exp(-x0))

start, stop, step = -4, 5, .1
x = np.array([]) # x = np.arange(start, stop, step) #mismatch
y = np.array([])

for z in list(frange(start, stop, step)):
    x = np.append(x, z)
    y = np.append(y,(
    sigmoid(10*z + 1),
    sigmoid(-2*z + 1),
    sigmoid(2*z -3)
    ))
#print(x.shape, y.shape)
y = y.reshape((int(y.shape[0]/3),3)) 
 
fig, ax = plt.subplots(figsize=(6, 3))

ax.plot(x, y[:,0], color = 'red', label='σ(10x + 1)')
ax.plot(x, y[:,1], color = 'blue', label='σ(−2x + 1)')
ax.plot(x, y[:,2], color = 'green', label='σ(2x − 3)')

#  What governs the steepness of the curve?\n
#  What governs the x value where the output is equal to 0.5?
ax.set_title('Which plot is which?')
ax.legend(loc='upper left')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_xlim(xmin=start, xmax=stop)
ax.set_ylim(ymin=-0.25, ymax=1.25)
ax.grid(True)
fig.tight_layout()
 
plt.show()
