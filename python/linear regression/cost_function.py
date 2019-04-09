# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 17:00:14 2019

@author: derekh
"""
import numpy as np

def computeCost(X, y, theta):
    thetaT = theta.T
    inner = np.power(((X * thetaT) - y), 2)
    return np.sum(inner) / (2 * len(X))
