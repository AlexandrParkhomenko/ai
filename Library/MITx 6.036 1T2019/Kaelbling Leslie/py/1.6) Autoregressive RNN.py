#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 16:29:40 2021

@author: a
"""
import numpy as np
class SM:
    start_state = None

    def transduce(self, input_seq):
        '''input_seq: a list of inputs to feed into SM
           returns:   a list of outputs of SM'''
        ret = []
        for x in input_seq:
            s = self.transition_fn(self.start_state,x)
            self.start_state = s
            ret.append(self.output_fn(s))
        return ret
class RNN(SM):
    def __init__(self, Wsx, Wss, Wo, Wss_0, Wo_0, f1, f2, start_state=None):
        self.Wsx=Wsx
        self.Wss=Wss
        self.Wo=Wo
        self.Wss_0=Wss_0
        self.Wo_0=Wo_0
        self.f1=f1
        self.f2=f2
        if start_state is None:
            self.start_state=np.zeros(self.Wsx.shape[0])
        else:
            self.start_state = start_state
    
    def transition_fn(self, s, x):
        # Your code here
        print("s:",s)
        ret = self.f1(self.Wsx@x+self.Wss@s+self.Wss_0)
        #print(ret)
        self.start_state=s
        return ret
    def output_fn(self, s):
        # Your code here
        ret = self.f2(self.Wo@s+self.Wo_0)
        # print(ret[:,0])
        ret = ret[:,0]
        ## print("ret",ret,"shape",ret.shape)
        return np.array([ret]).T #.reshape(3,2,1)

# Your code here
ok = np.array([[[0.0]], [[1.0]], [[0.0]], [[2.0]], [[4.0]], [[6.0]], [[8.0]], [[10.0]], [[12.0]], [[14.0]]])
Wsx =    np.ones(shape=(3,1))
Wo =     np.array([[1,1,1]])
Wss_0 =  np.zeros(shape=(3,1))
Wo_0 =   0
f1 =     lambda x: x# Your code here
f2 =     lambda x: x# Your code here
s = range(-3,4)
for a in s:
    for b in s:
        for c in s:
            Wss =    np.array([[a,b,c],
                               [0,1,0],
                               [0,0,1]])
            start_state = np.array([[1,0,0]]).T
            auto = RNN(Wsx, Wss, Wo, Wss_0, Wo_0, f1, f2, start_state)
            
            ans=[x.tolist() for x in auto.transduce([np.array([[x]]) for x in range(10)])]
            #print(ans)
            if(np.allclose(ans, ok)):
                









