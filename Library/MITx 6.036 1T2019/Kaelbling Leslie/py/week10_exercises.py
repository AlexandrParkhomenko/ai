#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 21:24:23 2021

@author: https://github.com/AlexandrParkhomenko <it@52tour.ru>

Let's simulate the Q-learning algorithm!
"""

import numpy as np

alpha = 0.5
gamma = 0.9

#              s   a   r
experience = [(0, 'b', 0), #t = 0
              (2, 'b', 0),
              (3, 'b', 2),
              (0, 'b', 0), #t = 3
              (2, 'b', 0),
              (3, 'c', 2),
              (0, 'c', 0), #t = 6
              (1, 'b', 1),
              (0, 'b', 0),
              (2, 'c', 0), #t = 9
              (3, 'c', 2),
              (0, 'c', 0),
              (1, 'c', 1), #t = 12
              (0, 'c', 0),
              (2, 'b', 0),
              (3, 'b', 2), #t = 15
              (0, 'b', 0),
              (2, 'c', 0),
              (3,  '', 0), #t = 18
              ]
# print(experience[0][0])
# Q = 0
# USE <class 'numpy.float64'> !
Q = np.array([[0.,0.,0.,0.], # 4 states, action b
              [0.,0.,0.,0.]
             ]).T
for j in range(0,18):
    # break
    s = experience[j][0]
    s2 = experience[j+1][0]
    # определимся с действием
    if experience[j][1] == 'b':
        a = 0
    else: # == 'c'
        a = 1
    r = experience[j][2]
    # print('Qnew (',s,', ',experience[j][1],') = 0.5 ⋅ Qold (',s,', ',experience[j][1],') + 0.5(',r,' + 0.9 ⋅ maxa′ Qold (',s2,', a′ ))')
    # print('0.5 ⋅ ',Q[s][a],' + 0.5 ⋅ (',r,' + 0.9 ⋅ ',np.max(Q[s2]),')')
    # print(type(Q[s,a]))
    # Q[s][a] = 0.5 * Q[s][a] + 0.5 * (r + 0.9 * np.max(Q[s2]))
    Q[s][a] = (1 - alpha) * Q[s][a] + alpha * (r + gamma * np.max(Q[s2]))
    # print(Q[s][a])
    # print(Q,Q[s,a])
    print(Q[s,a])
    if j%3 == 2:
        print('---')









