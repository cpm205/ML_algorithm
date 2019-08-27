# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 10:50:50 2019

@author: derekh
"""

"""
Use math package to implement sigmoid
"""
import math

def basic_sigmoid(x):
    s = 1/(1+math.exp(-x))
    return s
    
print(basic_sigmoid(3))



"""
Actually, we rarely use the "math" library in deep learning because the inputs of the functions are real numbers. 
In deep learning we mostly use matrices and vectors. This is why numpy is more useful.
Use numpy package to implement sigmoid
"""
import numpy as np 

def sigmoid(x):
     s = 1/(1+np.exp(-x))
     return s

x = np.array([1, 2, 3])
print(sigmoid(x))