# -*- coding: utf-8 -*-
"""
@author: %(Mikel Val Calvo)s
@email: %(mikel1982mail@gmail.com)
@institution: %(Dpto. de Inteligencia Artificial, Universidad Nacional de Educación a Distancia (UNED))
@DOI: 10.5281/zenodo.3759306 
"""
#%%
from FILTERS.filter_bank_manager import filter_bank_class
from FEATURES import features
from ENTROPY import entropy

import pandas as pd
import numpy as np

def compute_online_features(sample, constants):    
    fb = filter_bank_class(constants)
    fb.update_filters()
    
    FILTER_RANGES = constants.FILTER_RANGES
    all_channels = list(np.array(constants.CHANNEL_IDS)[constants.AVAILABLE_CHANNELS])
    Fs = constants.SAMPLE_RATE
    
    reco = pd.DataFrame(sample.T)    
    reco.columns = all_channels
   
    computed_features = []
    for channel in range(len(all_channels)):
        ch_data = np.asarray(reco[all_channels[channel]])

        ## band pass
        bands = fb.filter_bank(ch_data, constants.SAMPLE_RATE, FILTER_RANGES, order=5)
        
        for band in range(len(bands)):
            # -- diferential entropy
            DE = entropy.differential_entropy(bands[band])
            computed_features.append( DE )     
            # -- amplitude envelope
            AMP = features.amplitude_envelope(bands[band], Fs, FILTER_RANGES[band])
            computed_features.append( AMP ) 
            # -- Petrosian fractal dimension
            PFD = features.pfd(bands[band] )
            computed_features.append( PFD )
            # -- Hjorth fractal dimension
            HJ = features.hjorth_fd(bands[band],5 )
            computed_features.append( HJ )
            # -- Fisher info
            FI = features.fisher_info(bands[band],2,2)
            computed_features.append( FI )

    return np.asarray(computed_features)
