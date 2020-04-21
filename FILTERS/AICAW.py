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
from FILTERS.filter_bank_manager import filter_bank_class 
# ICA methods
from sklearn.decomposition import FastICA
from ica import ica1
from sklearn.preprocessing import StandardScaler
from scipy.stats import kurtosis, zscore
import numpy as np
import pywt

#%% ICA-WAVELET 
def aicaw(data, constants, m=2, r=.2, tau=4, method='infomax'):
    fb = filter_bank_class(constants)
    n_components = constants.CHANNELS
    n_channels = constants.CHANNELS
    ############## ICA DECOMPOSITION ###################
    #------ FastICA
    if method == 'fast_ica':
        ica = FastICA(n_components, tol=.2)
        S = ica.fit_transform(data.T).T
        A = ica.mixing_ 
    #-------- ICA infomax 1
    elif method == 'infomax':
        A,S,W = ica1(data, n_components)

    S_std = StandardScaler().fit_transform(S)
    reco = np.zeros(S.shape)
    mmse_list = np.zeros((n_channels,))
    k_list = np.zeros((n_channels,))
    for indx in range(n_channels):     
        signal = S_std[indx] 
        
        mmse_list[indx] = entropy.rcmse(signal, m, r, tau)
        k_list[indx] = kurtosis(signal)
      
    upper_limit = k_list.mean() + (k_list.std()/n_channels)*2.3646 #2.3646 for 8 channels 2.201 for 12 channels 2.04 for 32 channels
    lower_limit =  mmse_list.mean() - (mmse_list.std()/n_channels)*2.04
    for indx in range(n_channels): 
        signal = S_std[indx] 
        
        bands = []
        if mmse_list[indx] < lower_limit or k_list[indx] > upper_limit:
            coefs = fb.wavelet_filter_aicaw(signal, 'bior1.1')
                
            for c in range(len(coefs)):
                if c == 4:
                    for k in range(len(coefs[c])):
                        if np.sum(coefs[c][k]) != 0:
                            sigma = np.median(np.abs(coefs[c][k])/.6745)
                            threshold = np.sqrt(2*np.log(n_channels)*sigma)
            
                            if kurtosis(coefs[c][k]) > threshold:
                                coefs[c][k] = np.zeros((len(coefs[c][k]),))
                bands.append(pywt.waverec(coefs[c], 'bior1.1'))
                
            reco[indx] = bands[0] + bands[1] + bands[2] + bands[3] + bands[4]
        else:
            reco[indx] = signal
        
    if method == 'fast_ica':
        signals = (np.dot(reco.T, A.T) + ica.mean_).T
    elif method == 'infomax':
        signals = A.dot(reco)
    for i in range(n_channels):
        signals[i,:] = zscore(signals[i,:])*10 + 100*(i+1)
    return signals

