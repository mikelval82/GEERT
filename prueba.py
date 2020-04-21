#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: %(Mikel Val Calvo)s
@email: %(mikel1982mail@gmail.com)
@institution: %(Dpto. de Inteligencia Artificial, Universidad Nacional de Educación a Distancia (UNED))
"""

from FILMS_ANALYSIS_ESCENAS import utils_general
from GENERAL.constants_02 import constants
from FEATURES.online_features_02 import compute_online_features
from FILTERS.EAWICA import eawica
import numpy as np
import pandas as pd
from FILTERS.filter_bank_manager import filter_bank_class
from FEATURES import features
from ENTROPY import entropy

from scipy import signal, fftpack
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

subject = ['RICARDO']
film = 'cienmetros'
constants = constants()
constants.AVAILABLE_CHANNELS = [True,True,True,True,True,True,True,True]

arousal, valence, label_times_eeg, label_times_bvp, label_times_gsr = utils_general.get_labels(subject, film)
#% load film data and split into scenes
start = 10
end = 10

signal = 'eeg'
fs = 250
EEG_scenes = utils_general.load_film2scenes(subject, film, signal, label_times_eeg, start, end, fs)    
    
indx = 21
print('emotion: ', valence[indx])
EEG_data = EEG_scenes[indx]
fs = 250
seconds = 6
sliding = 3

step = fs*sliding# seconds of sliding step * fs
sample_length = fs*seconds
mask = constants.AVAILABLE_CHANNELS

n_samples = int(EEG_data.shape[1]/step - sample_length/step)
print(n_samples)

list_features_pos = np.zeros((20,40))
list_features_neg = np.zeros((20,40))

#%%
n = 19



ini = n*step
end = ini+sample_length
sample = EEG_data[:,ini:end] 
#from scipy.stats import zscore
#fig,ax = plt.subplots(2,1)
#for i in range(8):
#    ax[0].plot(zscore(sample[i,:])/5 + i)
sample = eawica(sample[mask,:],constants)
#for i in range(8):
#    ax[1].plot(zscore(sample[i,:])/5 + i)
    
    
fb = filter_bank_class(constants)
fb.update_filters()

FILTER_RANGES = constants.FILTER_RANGES
all_channels = list(np.array(constants.CHANNEL_IDS)[constants.AVAILABLE_CHANNELS])
Fs = constants.SAMPLE_RATE

reco = pd.DataFrame(sample.T)    
reco.columns = all_channels

#fig,axes = plt.subplots(4,2)
features = np.zeros((60,))
for channel in range(len(all_channels)):
    ch_data = np.asarray(reco[all_channels[channel]])

    ## band pass
    bands = fb.filter_bank(ch_data, constants.SAMPLE_RATE, FILTER_RANGES, order=5)
    
    bands_peaks = np.zeros((5,))
    for band in range(len(bands)):
        f_range = FILTER_RANGES[band]
        N = 1500
        fs = 250.0
        T = 1.0 / fs
        
        time_series = bands[band]

        yf = fftpack.fft(time_series)
        yff = 2.0/N * np.abs(yf[:N//2])
        xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
        
        
        
        computed_fft = np.array([ xf,yff ])
        smooth = savgol_filter(yff, 41, 3)
        peaks = find_peaks(smooth)[0]
        
        bands_peaks[band] = smooth[peaks].max()
        print(bands_peaks[band])
        
        
#        limit = int(len(xf)/2)
#        axes[(channel%4),int(channel/4)].plot(xf[:limit],yff[:limit])
#        axes[(channel%4),int(channel/4)].plot(xf[:limit],smooth[:limit])
#        axes[(channel%4),int(channel/4)].plot(xf[peaks], smooth[peaks], "x")
        
        
    ini = (channel%8)  * 5
    end = ini + 5
    features[ini:end] = bands_peaks

features[-20:] = features[:20] - features[20:40]

    
    