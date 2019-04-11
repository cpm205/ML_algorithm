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

def myComputeCost(X, y, theta):
    thetaTrans = theta.T
    values = np.power((X * thetaTrans - y), 2)
    sumValues = np.sum(values)
    return sumValues / (2 * len(X))


def myComputeCost2(X, y, theta):
    temp = np.power((X * theta.T - y), 2)
    result = np.sum(temp)/(2*len(X))
    return result

            
        