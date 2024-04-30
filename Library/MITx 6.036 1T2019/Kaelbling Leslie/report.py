# https://openlearninglibrary.mit.edu/courses/course-v1:MITx+6.036+1T2019/courseware/Week2/week2_homework/?child=first

# 1.1c) How many mistakes does the algorithm make if it starts with data point x(2) (and then does x(3) and x(1)?

import numpy as np

x1, label1 = [1,-1], 1
x2, label2 = [0,1], -1
x3, label3 = [-1.5,-1], 1
## x3, label3 = [-10,-1], 1

data = np.array([x1,x2,x3]).T
labels = np.array([label1,label2,label3])

x1, label1 = [-3,2], 1
x2, label2 = [-1,1], -1
x3, label3 = [-1,-1], -1
x4, label4 = [2,2], -1
x5, label5 = [1,-1], -1

data = np.array([x1,x2,x3,x4,x5]).T
labels = np.array([label1,label2,label3,label4,label5])

x1, label1 = [1,1,1], 1
x2, label2 = [1,1,0], -1
x3, label3 = [1,0,1], -1
x4, label4 = [0,1,1], -1

data = np.array([x1,x2,x3,x4]).T
labels = np.array([label1,label2,label3,label4])

def perceptron(data, labels, params = {}, hook = None):    
    # if T not in params, default to 100
    T = params.get('T', 100)
    # Your implementation here
    d, n = data.shape
    theta = np.zeros((d,1))
    theta_0 = np.zeros(1)
    print("d = {}, n = {}, theta shape = {}, theta_0 shape = {}".format(d,n,theta.shape,theta_0.shape))
  
    for t in range(T):     
      for i in range(n):
        y = labels[i]  #// 1 dimension array
        x = data[:,i]
        
        a = np.dot(x,theta)+theta_0
        #print("a = {}".format(a))
        positive = np.sign(y*a)
        
        if np.sign(y*a) <=0: # update the thetas
          theta[:,0] = theta[:,0]+ y*x
          theta_0 = theta_0 + y #//
          print("update the thetas:", theta[:,0], "i:", i+1) #// show
          
    print("shape x = {}, y = {}, theta = {}, theta_0 = {}".format(x.shape,y.shape,theta.shape,theta_0.shape))
    return (theta,theta_0)

perceptron(data, labels, params = {}, hook = None)

# output with theta_0 update:
# d = 2, n = 3, theta shape = (2, 1), theta_0 shape = (1,)
# update the thetas: [ 0. -1.]
# update the thetas: [-1.5 -2. ]
# shape x = (2,), y = (), theta = (2, 1), theta_0 = (1,)
# (array([[-1.5],
#        [-2. ]]), array([0.]))

# Number of mistakes is: 2, but  100.00% is 1


# It doesnt matter 
# d = 2, n = 3, theta shape = (2, 1), theta_0 shape = (1,)
# update the thetas: [ 0. -1.]
# shape x = (2,), y = (), theta = (2, 1), theta_0 = (1,)
# (array([[ 0.],
#        [-1.]]), array([0.]))


# https://openlearninglibrary.mit.edu/courses/course-v1:MITx+6.036+1T2019/courseware/Week2/week2_homework/?child=first

Consider the following plots. For each one estimate plausible values of R (an upper bound on the magnitude of the training vectors) and gamma (the margin of the separator for the dataset). Consider values of R in the range [1,10] and values of gamma in the range [0.01,2]]. 
6.1a) 
Enter a Python list with 2 floats, a value of R and a value of gamma: 
[0,0.1] 100.00% #! OMG R=0  error < 0

# https://openlearninglibrary.mit.edu/courses/course-v1:MITx+6.036+1T2019/courseware/Week3/week3_lab/?child=first
3A)
now: Black pixels have value 000 while while pixels have value 1 in the matrix.
ok:  Black pixels have value 000 while white pixels have value 1 in the matrix.

# https://openlearninglibrary.mit.edu/courses/course-v1:MITx+6.036+1T2019/courseware/Week1/week1_exercises/1?activate_block_id=block-v1%3AMITx%2B6.036%2B1T2019%2Btype%40vertical%2Bblock%40week1_exercises_1_vert
np.asscalar(a) is deprecated since NumPy v1.16, use a.item() instead

not found image: https://openlearninglibrary.mit.edu/asset-v1:MITx+6.036+1T2019+type@asset+block/images_logreg3d.png

on page https://openlearninglibrary.mit.edu/courses/course-v1:MITx+6.036+1T2019/courseware/Week4/week4_exercises/?child=first
not found link https://openlearninglibrary.mit.edu/courses/course-v1:MITx+6.036+1T2019/courseware/Week4/margin_maximization/1 ?
on page https://openlearninglibrary.mit.edu/courses/course-v1:MITx+6.036+1T2019/courseware/Week4/week4_exercises/?child=first
not found link https://openlearninglibrary.mit.edu/courses/course-v1:MITx+6.036+1T2019/courseware/Week4/margin_maximization/3

Please publish
https://openlearninglibrary.mit.edu/assets/courseware/v1/030eba9b066b079e2b16dc863c18ea39/asset-v1:MITx+6.036+1T2019+type@asset+block/notes_chapter_Margin_Maximization.pdf

no subtitles
https://openlearninglibrary.mit.edu/courses/course-v1:MITx+6.036+1T2019/courseware/Week7/neural_networks_2/?child=first


import numpy as np
def perceptron(data, labels, params={}, hook=None):
    # if T not in params, default to 100
    T = params.get('T', 100)
    N = data.shape[1]
    theta = np.zeros((data.shape[0],1))
    theta0 = np.zeros(1)
    #print("theta:",theta,theta0)
    for t in range(T):
        for n in range(N):
            x = data[:,n]
            y = labels[0,n]
            print("x=",x,", y=",y)
            if y*(np.dot(x,theta)+theta0) <= 0:
                theta[:,0] = theta[:,0]+ y*x # my error was here
                theta0 = theta0 + y
    return (theta,theta0)




