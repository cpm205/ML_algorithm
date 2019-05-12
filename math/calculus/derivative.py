#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 00:50:19 2019

@author: Derek
"""
def get_derivative(func, x):
    """Compute the derivative of `func` at the location `x`."""
    h = 0.0001                          # step size
    return (func(x+h) - func(x)) / h    # rise-over-run

def f(x): return x**2                   # some test function f(x)=x^2
x = 3                                   # the location of interest
computed = get_derivative(f, x)
actual = 2*x

computed, actual
