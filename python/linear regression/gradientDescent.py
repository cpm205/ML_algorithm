# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 17:12:02 2019

@author: derekh
"""
import numpy as np
import cost_function as cf

def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    
    for i in range(iters):
        error = (X * theta.T) - y
        
        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))
            
        theta = temp
        cost[i] = cf.computeCost(X, y, theta)
        
    return theta, cost
