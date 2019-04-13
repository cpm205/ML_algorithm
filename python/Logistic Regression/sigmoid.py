#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 23:10:06 2019

@author: derek
"""
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))