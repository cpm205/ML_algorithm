#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 21:54:15 2019

@author: Derek
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cost_function as cf
import gradientDescent as gd
#%matplotlib inline

path = os.getcwd() + '/ex1data2.txt'
data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
print(data2.head())

#calculates some basic statistics on a data set
print(data2.describe())

#Notice that the scale of the values for each variable is vastly different.
#we need to adjust the scale of the features to level the playing field. 
#One way to do this is by subtracting from each value in a feature the mean of that feature, 
#and then dividing by the standard deviation
data2 = (data2 - data2.mean()) / data2.std()
print(data2.head())

# add ones column
data2.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data2.shape[1]
X2 = data2.iloc[:,0:cols-1]
y2 = data2.iloc[:,cols-1:cols]

# convert to matrices and initialize theta
X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([0,0,0]))

# initialize variables for learning rate and iterations
alpha = 0.01  #learning rate
iters = 2000

# perform linear regression on the data set
g2, cost2 = gd.gradientDescent(X2, y2, theta2, alpha, iters)

# get the cost (error) of the model
cf.computeCost(X2, y2, g2)

print("Gradient Descent: ", g2)
print("cost: ", cf.computeCost(X2, y2, g2))


fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost2, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')