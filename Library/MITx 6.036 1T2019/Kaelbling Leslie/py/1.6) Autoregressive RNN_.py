#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 16:29:40 2021

@author: a
import numpy as np
class SM:
    start_state = None

    def transduce(self, input_seq):
        '''input_seq: a list of inputs to feed into SM
           returns:   a list of outputs of SM'''
        state = self.start_state
        output = []
        for inp in input_seq:
            state = self.transition_fn(state, inp)
            output.append(self.output_fn(state))
        return output
class RNN(SM):
  def __init__(self, Wsx, Wss, Wo, Wss_0, Wo_0, f1, f2, start_state):
      self.Wsx = Wsx
      self.Wss = Wss
      self.Wo = Wo
      self.Wss_0 = Wss_0
      self.Wo_0 = Wo_0
      self.start_state = start_state
      self.f1 = f1
      self.f2 = f2
  def transition_fn(self, s, x):
      print("s: ", s)
      print("x: ", x)
      return self.f1(np.dot(self.Wss, s) + np.dot(self.Wsx, x) + self.Wss_0)
  def output_fn(self, s):
      return self.f2(np.dot(self.Wo, s) + self.Wo_0)

# Your code here
ok = np.array([[[0.0]], [[1.0]], [[0.0]], [[2.0]], [[4.0]], [[6.0]], [[8.0]], [[10.0]], [[12.0]], [[14.0]]])
# Wsx =    np.array([[1,0,0]]).T
# Wo =     np.array([[1,0,0]])
# Wss_0 =  np.zeros(shape=(3,1))
# Wo_0 =   0
# f1 =     lambda x: x# Your code here
# f2 =     lambda x: x# Your code here
# s = range(-3,4)
# if False: #for a in s:
#     if True: #for b in s:
#         if True: #for c in s:
#             Wss =    np.array([[-1,0,0],
#                                [0,2,0],
#                                [0,0,-3]])
#             start_state = np.array([[2,1,0]]).T
#             auto = RNN(Wsx, Wss, Wo, Wss_0, Wo_0, f1, f2, start_state)
            
#             ans=[x.tolist() for x in auto.transduce([np.array([[x]]) for x in range(10)])]
#             print(ans,ok)
#             if(np.allclose(ans, ok)):
#                 print(Wss,start_state)
                
Wsx =    np.zeros((3,1))
Wss =    np.array([[1, -2, 3],
                   [1, 0, 0],
                   [0, 1, 0]])
Wo =     np.array([[1, 0, 0]])
Wss_0 =  np.array([[0, 0, 0]]).T
Wo_0 =   np.array([[0]])
f1 =     lambda x: x
f2 =     lambda x: x
start_state = np.array([[-2, 0, 0]]).T
auto = RNN(Wsx, Wss, Wo, Wss_0, Wo_0, f1, f2, start_state)

ans=[x.tolist() for x in auto.transduce([np.array([[x]]) for x in range(10)])]
print(ans,ok)
print(np.allclose(ans, ok))
y1 = 0
y2 = 0
y3 = 0
for x in range(10):
    y3 = y2
    y2 = y1
    y1 = x
    print(x, y1-2*y2+3*y3)
"""

import numpy as np





    ## for s ∈ S, a ∈ A :
        ## Q[s, a] = 0
    ## s = s0 // Or draw an s randomly from S
    ## while True:
            ## a = select_action(s, Q)
            ## r, s0 = execute(a)
            ## Q[s, a] = (1 − α)Q[s, a] + α(r + γ max a0 Q[s0 , a0 ])
            ## s = s0
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



'''
# These examples are reproducible only if random seed is set to 0 in
# both the random and numpy.random modules.
import mdptoolbox
import numpy as np
P = np.array([[[0.5, 0.5],[0.8, 0.2]],[[0, 1],[0.1, 0.9]]])
R = np.array([[5, 10], [-1, 2]])
np.random.seed(0)
ql = mdptoolbox.mdp.QLearning(P, R, 1.0, n_iter=1)
ql.run()
print(ql.Q)
# [[33.33010866 40.82109565]
#  [34.37431041 29.67236845]]
expected = (40.82109564847122, 34.37431040682546)
all(expected[k] - ql.V[k] < 1e-12 for k in range(len(expected)))
# True
print(ql.policy)
# (1, 0)
'''
