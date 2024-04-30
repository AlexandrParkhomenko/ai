# Regularizing linear regression
# What if np.matmul(x.T,x) is not invertible? 

import numpy as np

x = np.array([[1,1],
              [2,2]])
print("x:\n",x)
print("x.T:\n",x.T)
print("np.matmul(x.T,x):\n",np.matmul(x.T,x))
try:
    print("np.linalg.inv:\n",np.linalg.inv(np.matmul(x.T,x)))
except Exception:
    print("LinAlgError: Singular matrix")
print("np.linalg.inv:\n",np.linalg.inv(np.matmul(x.T,x)+np.diag([0.1,0.1]))) #
