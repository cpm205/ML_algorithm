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


def myGD2(X, y, theta, learningRate, iters):
    tempThetaResult = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    
    for i in range(iters):
        diff = (X * theta.T) - y
        
        for j in range(parameters):
            tempThetaResult[0,j] = tempThetaResult[0,j] - learningRate/len(X) * np.sum(np.multiply(diff, X[:,j]))
        theta = tempThetaResult
        cost[i] = cf.computeCost(X, y, theta)
    
    return theta, cost
            



def myGD(X, y, theta, learningRate, iters):
    #Create numpy array with all zeros based on shape of weight(paramter)
    zeroArray = np.zeros(theta.shape)
    tempThetaResult = np.matrix(zeroArray) # iters * 2
    params = theta.ravel().shape[1] # = 2, because 2 params
    intParams = int(params)# conver to int
    cost = np.zeros(iters)
    
    for i in range(iters):
        diff = (X * theta.T) - y
        
        #update Theta values
        for j in range(intParams):
            term = np.multiply(diff, X[:,j])
            tempThetaResult[0,j] = theta[0,j] - learningRate/len(X) * np.sum(term)
        theta = tempThetaResult
        cost[i] = cf.computeCost(X, y, theta)
    return theta, cost


        
        
            
    
