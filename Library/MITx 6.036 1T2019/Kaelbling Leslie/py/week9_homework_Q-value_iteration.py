'''
Where I wrong?
Second iter is incorect.
'''

import numpy as np

## for s ∈ S, a ∈ A :
    ## Q[s, a] = 0
## s = s0 // Or draw an s randomly from S
## while True:
    ## a = select_action(s, Q)
    ## r, s1 = execute(a)
    ## Q[s, a] = (1 − α)Q[s, a] + α(r + γ max_a1 Q[s1 , a1 ])
    ## s = s1
def Qvalue_iterations(T,R,gamma,n_iters=2):
    nS = T.shape[0]
    sum_ = 0
    #Q = np.zeros(nS)
    Q = R
    for _ in range(n_iters):
        sum_ = np.zeros(nS)
        for s in range(nS):
            for s_ in range(nS):
                sum_[s] += T[s][s_] * R[0][s_]
        Q = Q + gamma * sum_
    return Q

R = np.array([[0.0, 1.0, 0.0, 2.0]
              ])
Tb = np.array([[0.0, 0.9, 0.1, 0.0],
               [0.9, 0.1, 0.0, 0.0],
               [0.0, 0.0, 0.1, 0.9],
               [0.9, 0.0, 0.0, 0.1]
               ])
Tc = np.array([[0.0, 0.1, 0.9, 0.0],
               [0.9, 0.1, 0.0, 0.0],
               [0.0, 0.0, 0.1, 0.9],
               [0.9, 0.0, 0.0, 0.1]
               ])
iters = 1
print(Qvalue_iterations(Tb, R, gamma=0.9, n_iters=iters))
print(Qvalue_iterations(Tc, R, gamma=0.9, n_iters=iters))
