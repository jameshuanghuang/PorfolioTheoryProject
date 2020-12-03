#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 13:41:32 2020

@author: jameshuang
"""

import numpy as np
import scipy 


def fun(weights, dt, mu):                           
    
    pbar = np.matrix(np.mean(dt, axis=1)).T
    S = np.matrix(np.cov(dt))
    weights = np.matrix([weights]).T 
    
    return 0.5*mu*np.dot(weights.T, S*weights) - pbar.T*weights
     
def cons0(weights, A, b):
    #all cash Iw = 0 
    # equal
    weights = np.matrix([weights]).T 
    
    return np.sum(A*weights - b)

def cons1(weights, beta, b0):
    #beta neutral bw = 0
    # equal
    weights = np.matrix([weights]).T 
    
    return np.sum(beta*weights - b0)

def cons2(weights, S, track):
    #tracking error
    #inequ >=0
    weights = np.matrix([weights]).T 
    resi = np.std(np.dot(weights.T, S*weights))
    return -resi + track



def sci_opt(dt, beta, mu):
    
    n = len(dt)
    A = np.matrix( np.eye(n) )
    b = np.matrix(0.0)
    #beta = np.random.randint(-100,100,(1,n))/100
    b0 = np.matrix(0.0)
    S = np.matrix( np.cov(dt) )
    track = 0.03/250
    #N_tol = 80 
    
    arg_fun = (dt, mu)  
    arg_cos0 = (A, b)  # aw = 0  
    arg_cos1 =(beta, b0)   #beta neutral 
    arg_cos2 = (S, track)  #tracking error 
    bnds = ((-1/n, 1.001),) * len(dt) # long only 
    cons = (
           {'type': 'eq', 'fun': cons0, 'args': arg_cos0},
           {'type': 'eq', 'fun': cons1, 'args': arg_cos1},
           {'type': 'ineq', 'fun': cons2, 'args': arg_cos2}
           )
    
    guess = (1/n,) * len(dt)     
    result = scipy.optimize.minimize(fun, x0 = guess, method = 'SLSQP',
                                     args = arg_fun, constraints = cons,
                                     bounds=bnds,
                                     options={'maxiter': 101})  
    #print(result)     
    #assert   abs(np.sum(result.x)) <= 0.0001
    #assert   np.sum(np.array(result.x)>-1/n) >= n-5
    #assert   abs(np.sum(beta*result.x)) <= 0.001
    print(result.success)
    return np.array(result.x) 




