# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 15:26:30 2019

@author: derekh
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cost_function as cf
import gradientDescent as gd
#%matplotlib inline

path = os.getcwd() + '\ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
print(data.head())

#calculates some basic statistics on a data set
print(data.describe())

data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))

"""
In order to make this cost function work seamlessly with the pandas data frame we created above, 
we need to do some manipulating. First, we need to insert a column of 1s at 
the beginning of the data frame in order to make the matrix operations work 
correctly (I won't go into detail on why this is needed, but it's in the exercise 
text if you're interested - basically it accounts for the intercept term in 
the linear equation). Second, we need to separate our data into independent 
variables X and our dependent variable y.
"""
# append a ones column to the front of the data set
data.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

X = np.matrix(X.values)
y = np.matrix(y.values)

# instantiate a parameter matirx
theta = np.matrix(np.array([0,0]))

print(X.shape)
print(theta.shape)
print(y.shape)

print("Cost Function:",cf.computeCost(X, y, theta))
print("my Cost Function:",cf.myComputeCost(X, y, theta))


# initialize variables for learning rate and iterations
alpha = 0.01  #learning rate
iters = 1000

# perform gradient descent to "fit" the model parameters
g, cost = gd.gradientDescent(X, y, theta, alpha, iters)
print(g)
print(cost)
print(cf.computeCost(X, y, g))

x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0, 0] + (g[0, 1] * x)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')



fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')

