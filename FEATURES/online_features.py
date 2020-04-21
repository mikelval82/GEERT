#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: %(Mikel Val Calvo)s
@email: %(mikel1982mail@gmail.com)
@institution: %(Dpto. de Inteligencia Artificial, Universidad Nacional de Educación a Distancia (UNED))
"""
from FILTERS.filter_bank import filter_bank 
from FEATURES import features
from ENTROPY import entropy
import pandas as pd
import numpy as np

def compute_online_features(sample, constants):
    print('compute_online_features')
    fb = filter_bank(constants)
    fb.update_filters()
    
    left_channels = ['FP1','F7','F3','C3']
    right_channels = ['FP2','F8','F4','C4']
    all_channels = ['FP1','F7','F3','C3','FP2','F8','F4','C4']
    
    reco = pd.DataFrame(sample.T)    
    reco.columns = all_channels
   
    computed_features = []
    for channel in range(len(left_channels)):
        print('channel: ', channel)
        ch_left = np.asarray(reco[left_channels[channel]])
        ch_right = np.asarray(reco[right_channels[channel]])

        ## band pass
        bands_left = fb.filter_bank(ch_left, constants.SAMPLE_RATE, constants.FILTER_RANGES, order=5)
        bands_right = fb.filter_bank(ch_right, constants.SAMPLE_RATE, constants.FILTER_RANGES, order=5)
        
        
        for band in range(len(bands_left)):
            print('band: ', band)
           

            #--- differential entropy -----
   
            DE_left = entropy.differential_entropy(bands_left[band])
            DE_right = entropy.differential_entropy(bands_right[band])
            computed_features.append(DE_left/DE_right)
       
            #--- amplitude envelope -----
       
            AMP_left = features.amplitude_envelope(bands_left[band], constants.SAMPLE_RATE, tuple(constants.FILTER_RANGES[band]))
            AMP_right = features.amplitude_envelope(bands_right[band], constants.SAMPLE_RATE, tuple(constants.FILTER_RANGES[band]))
            computed_features.append(AMP_left/AMP_right)
      
            #--- instantaneous frequency -----
        
            IF_left = features.instantaneous_frequency(bands_left[band], constants.SAMPLE_RATE, tuple(constants.FILTER_RANGES[band]))
            IF_right = features.instantaneous_frequency(bands_right[band], constants.SAMPLE_RATE, tuple(constants.FILTER_RANGES[band]))
            computed_features.append(IF_left/IF_right)
   
            #----- Petrosian Fractal Dimension --------
     
            PFD_left = features.pfd(bands_left[band])
            PFD_right = features.pfd(bands_right[band])
            computed_features.append(PFD_left/PFD_right)
       
            #-------- Hjorth Fractal Dimension ---------
   
            HFD_left = features.hfd(bands_left[band], 6) # kmax = 6 to 60
            HFD_right = features.hfd(bands_right[band], 6) # kmax = 6 to 60
            computed_features.append(HFD_left/HFD_right)

            #------  Hjorth mobility and complexity -----------
  
            MOBILITY_left, COMPLEXITY_left = features.hjorth(bands_left[band])
            MOBILITY_right, COMPLEXITY_right = features.hjorth(bands_right[band])
            computed_features.append(MOBILITY_left/MOBILITY_right)
            computed_features.append(COMPLEXITY_left/COMPLEXITY_right)
   
            #-------- Fisher info ----------

            FISHER_left = features.fisher_info(bands_left[band],2,2)
            FISHER_right = features.fisher_info(bands_right[band],2,2)
            computed_features.append(FISHER_left/FISHER_right)

            #-------- detrended fluctuation analysis -----------

            DFA_left = features.dfa(bands_left[band])
            DFA_right = features.dfa(bands_right[band])
            computed_features.append(DFA_left/DFA_right)

         
    return np.asarray(computed_features)