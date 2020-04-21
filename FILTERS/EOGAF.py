#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: %(Mikel Val Calvo)s
@email: %(mikel1982mail@gmail.com)
@institution: %(Dpto. de Inteligencia Artificial, Universidad Nacional de Educación a Distancia (UNED))
"""
# entropy
from ENTROPY import entropy
# filters
from FILTERS import filter_bank as fb
# metrics
from METRICS import metrics
# infomax based ICA method
from ica import ica1
# external common libraries
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from sklearn import preprocessing
from scipy.stats import zscore
import numpy as np

from scipy.signal import savgol_filter

#%% ENHANCED WAVELET-ICA 
def eawica(EEG_sample, Fs, numChannels, n_samples, n_epochs, wavelet, low_k, up_k, low_r, up_r, alpha, path):

 #%%   # COMPUTE WAVELET DECOMPOSED wcs_delta
#    plt.figure()
    wcs_control, wcs_delta, wcs_theta, wcs_alpha, wcs_beta, wcs_gamma = [],[],[],[],[],[]
    for i in range(numChannels):
        GAMMA, BETA, ALPHA, THETA, DELTA, CONTROL = fb.eawica_wavelet_band_pass_200_new(EEG_sample[i,:], wavelet)

        wcs_delta.append(DELTA)
        wcs_theta.append(THETA)
        wcs_alpha.append(ALPHA)
        wcs_beta.append(BETA)
        wcs_gamma.append(GAMMA)
        wcs_control.append(CONTROL)
     
    S = np.asarray(wcs_control)     
    size = 100
    step = 50
    n_epochs = int((S.shape[1] - size )/step)+1

    control_c = np.zeros((S.shape[0],n_epochs))
    for indx1 in range(S.shape[0]):
        for indx2 in range((n_epochs)):
            ini = indx2*step
            end = ini + size
            epoch = S[indx1,ini:end]
            control_c[indx1,indx2] = np.sum(np.abs(np.diff(np.sign(zscore(epoch)))))
  
    control_c = zscore(control_c.T).T
    for indice in range(8):
#        low_threshold = np.percentile(control_c[indice,:], 50)
        table = control_c[indice,:] < 0
        
        for indx1 in range(table.shape[0]):          
            if table[indx1]:
                ini = indx1*step
                end = ini + step
                # epochs zeroing
                wcs_delta[indice][ini:end] = 0    
        wcs_delta[indice] = savgol_filter(wcs_delta[indice], step+1, 3)
    
    data_cleaned = np.zeros(EEG_sample.shape)
    for i in range(numChannels):
        data_cleaned[i,:] = wcs_delta[i] + wcs_theta[i] + wcs_alpha[i] + wcs_beta[i] + wcs_gamma[i]
  
    return data_cleaned
