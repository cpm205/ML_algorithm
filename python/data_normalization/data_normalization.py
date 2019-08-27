# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 12:23:07 2019

@author: derekh
"""
"""
It is a technique we use in Machine Learning and Deep Learning is to normalize our data. 
It often leads to a better performance because gradient descent converges faster after normalization.
"""

"""
 Implement normalizeRows() to normalize the rows of a matrix. 
 After applying this function to an input matrix x, each row of x should be a vector of unit length (meaning length 1).
"""
import numpy as np

def normalizeRows(x):
    """
    Implement a function that normalizes each row of the matrix x (to have unit length).
    
    Argument:
    x -- A numpy matrix of shape (n, m)
    
    Returns:
    x -- The normalized (by row) numpy matrix. You are allowed to modify x.
    """
    
    ### START CODE HERE ### (≈ 2 lines of code)
    # Compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
    x_norm = np.linalg.norm(x, axis=1, keepdims = True)
    
    # Divide x by its norm.
    x = x/x_norm
    ### END CODE HERE ###

    return x



x = np.array([
    [0, 3, 4],
    [1, 6, 4]])
print("normalizeRows(x) = " + str(normalizeRows(x)))