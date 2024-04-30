#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 20:14:38 2020

@author: a
"""
import numpy as np

def row_average_features(x):
    """
    @param x (m,n) array with values in (0,1)
    @return (m,1) array where each entry is the average of a row
    """
    print(np.array(x).shape)
    return np.array([np.sum(x, axis=1)/np.array(x).shape[1]]).T


def col_average_features(x):
    """
    @param x (m,n) array with values in (0,1)
    @return (n,1) array where each entry is the average of a column
    """
    return np.array([np.sum(x, axis=0)/np.array(x).shape[0]]).T



def top_bottom_features(x):
    """
    @param x (m,n) array with values in (0,1)
    @return (2,1) array where the first entry is the average of the
    top half of the image = rows 0 to floor(m/2) [exclusive]
    and the second entry is the average of the bottom half of the image
    = rows floor(m/2) [inclusive] to m
    """
    x = np.array(x)
    r,c = x.shape
    #return row_average_features(np.array(x).reshape((2,r*c//2)))
    #print((r+1)//2)
    a = np.array(x[0:r//2,:]).reshape(1,c*(r//2))
    b = np.array(x[r//2:r,:]).reshape(1,c*(r-(r//2)))
    d = np.array([])
    d = np.append(d,row_average_features(a))
    d = np.append(d,row_average_features(b))
    d = np.array([d]).T
    return d

ans=top_bottom_features(np.array([[1,2,3],[3,9,2]])).tolist()
print(ans)

ans=top_bottom_features(np.array([[1,2,3],[3,9,2],[2,1,9]])).tolist()
print(ans)


