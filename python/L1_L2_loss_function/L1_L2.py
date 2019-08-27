# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 14:34:41 2019

@author: derekh
"""

import numpy as np

def L1(yhat, y):
    loss = np.sum(np.absolute(y-yhat))
    return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = " + str(L1(yhat,y)))



def L2(yhat, y):
    loss = np.sum(np.square(y-yhat))
    return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L2 = " + str(L2(yhat,y)))