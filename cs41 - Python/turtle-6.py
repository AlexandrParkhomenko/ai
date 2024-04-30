#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 17:42:56 2024

@author: AlexandrParkhomenko
"""

from turtle import *
#clear() 
pendown() # опустить хвост
for i in range(2):
    forward(12*20)
    right(90)
    forward(19*20)
    right(90)
penup()
forward(4*20)
right(90)
forward(6*20)
left(90)
pendown() # опустить хвост
for i in range(2):
    forward(12*20)
    right(90)
    forward(6*20)
    right(90)
penup()
for x in range(0,25):
    for y in range(-25,0):
        setpos(x*20, y*20)
        dot(3,'red')
# 7*9 = 63
