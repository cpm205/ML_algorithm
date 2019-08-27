# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 11:45:41 2019

@author: derekh
"""

"""
 Implement the function sigmoid_grad() to compute the gradient of the sigmoid function with respect to its input x.
 sigmoid_derivative(x)=σ′(x)=σ(x)(1−σ(x))
"""

import numpy as np
def sigmoid_derivative(x):
    s = 1/(1+np.exp(-x))
    ds = s*(1-s)
    return ds