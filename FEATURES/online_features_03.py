# -*- coding: utf-8 -*-
"""
@author: %(Mikel Val Calvo)s
@email: %(mikel1982mail@gmail.com)
@institution: %(Dpto. de Inteligencia Artificial, Universidad Nacional de Educación a Distancia (UNED))
@DOI: 10.5281/zenodo.3759306 
"""
#%%
from FILTERS.filter_bank_manager import filter_bank_class
from scipy import fftpack
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
import pandas as pd
import numpy as np
    
def compute_online_features(sample, constants):
    fb = filter_bank_class(constants)
    fb.update_filters()
    
    FILTER_RANGES = [[4,8],[8,16],[16,32],[32,45]]
    all_channels = list(np.array(constants.CHANNEL_IDS)[constants.AVAILABLE_CHANNELS])
    
    reco = pd.DataFrame(sample.T)    
    reco.columns = all_channels

    features = np.zeros((60,))
    for channel in range(len(all_channels)):
        ch_data = np.asarray(reco[all_channels[channel]])
    
        ## band pass
        bands = fb.filter_bank(ch_data, constants.SAMPLE_RATE, FILTER_RANGES, order=5)
        
        bands_peaks = np.zeros((5,))
        for band in range(len(bands)):
            N = 1500

            yf = fftpack.fft(bands[band])
            yff = 2.0/N * np.abs(yf[:N//2])

            smooth = savgol_filter(yff, 41, 3)
            peaks = find_peaks(smooth)[0]
            
            bands_peaks[band] = smooth[peaks].max()
           
               
        ini = (channel%8)  * 5
        end = ini + 5
        features[ini:end] = bands_peaks
    
    features[-20:] = features[:20] - features[20:40]
    
    return features
 
