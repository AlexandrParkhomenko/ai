# week5_homework

# What am I doing wrong?

'''
Is XX^T invertible?  If not, what's the problem?  Mark all that are true.
 It is invertible
 It is not invertible because XXX is not square
-It is not invertible because two columns of X are linearly dependent
-It is not invertible because the rows of XX^T are linearly dependent
 It is not invertible because n is smaller than d
 We cannot compute the transpose of X
'''

import numpy as np

x = np.array([[1, 2], [2, 3], [3, 5], [1, 4]])
print("x:\n",x)
print("x.T:\n",x.T)
print("np.matmul(x,x.T):\n",np.dot(x,x.T)) # equal np.matmul
try:
    print("np.linalg.inv:\n",np.linalg.inv(np.dot(x,x.T)))
except Exception:
    print("LinAlgError: Singular matrix")
print("np.linalg.inv:\n",np.linalg.inv(np.dot(x,x.T)+np.diag([0.1,0.1,0.1,0.1]))) #

# x:
#  [[1 2]
#  [2 3]
#  [3 5]
#  [1 4]]
# x.T:
#  [[1 2 3 1]
#  [2 3 5 4]]
# np.matmul(x.T,x):
#  [[ 5  8 13  9]
#  [ 8 13 21 14]
#  [13 21 34 23]
#  [ 9 14 23 17]]
# np.linalg.inv:
#  [[ 5.62949953e+14  2.81474977e+15 -2.17137839e+15  3.21685688e+14]
#  [ 5.62949953e+14 -1.12589991e+15  6.43371375e+14 -2.41264266e+14]
#  [-5.62949953e+14 -0.00000000e+00  1.60842844e+14  8.04214219e+13]
#  [-0.00000000e+00 -5.62949953e+14  4.02107110e+14 -8.04214219e+13]]
# np.linalg.inv:
#  [[ 9.26060744 -1.11477648 -1.85416904 -1.46740985]
#  [-1.11477648  6.78079854 -4.33397793  0.86452053]
#  [-1.85416904 -4.33397793  3.81185303 -0.60288932]
#  [-1.46740985  0.86452053 -0.60288932  0.93390968]]
