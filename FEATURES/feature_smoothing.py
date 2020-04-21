#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 17:28:44 2019

@author: anaiak
"""
import numpy as np
#from pybasicbayes.distributions import Regression, DiagonalRegression
#from pylds.models import LDS
from scipy.signal import savgol_filter

def smoothing(features, window=31, degree=3):
    return savgol_filter(features, window, degree)

#def lds_smoothing(data, iterations):
#    # Parameters
#    D_obs = 1
#    D_latent = 2
#    D_input = 0
#    T = data.shape[0]
#    data = data.reshape((T,1))
#    # Simulate from an LDS with diagonal observation noise
#    inputs = np.random.randn(T, D_input)
#    
#    # Fit with an LDS with diagonal observation noise
#    diag_model = LDS(
#        dynamics_distn=Regression(nu_0=D_latent + 2,
#                                  S_0=D_latent * np.eye(D_latent),
#                                  M_0=np.zeros((D_latent, D_latent + D_input)),
#                                  K_0=(D_latent + D_input) * np.eye(D_latent + D_input)),
#        emission_distn=DiagonalRegression(D_obs, D_latent+D_input))
#    diag_model.add_data(data, inputs=inputs)
#    
#    
#    for i in range(iterations):
#        diag_model.resample_model()
#    
#    
#    # Smooth the data (TODO: Clean this up)
#    return diag_model.smooth(data, inputs).reshape((T,))