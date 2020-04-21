#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: %(Mikel Val Calvo)s
@email: %(mikel1982mail@gmail.com)
@institution: %(Dpto. de Inteligencia Artificial, Universidad Nacional de Educación a Distancia (UNED))
"""


EEG_data = EEG_scenes[13]


n = 30


fs=250
sliding = 2
step = fs*sliding# seconds of sliding step * fs
ini = n*step
end = ini+sample_length
sample = EEG_data[:,ini:end] 
sample = eawica(sample[:,:],constants)

import matplotlib.pyplot as plt
import scipy.fftpack

fb = filter_bank_class(constants)
fb.update_filters()

FILTER_RANGES = constants.FILTER_RANGES
all_channels = list(np.array(constants.CHANNEL_IDS)[constants.AVAILABLE_CHANNELS])
Fs = constants.SAMPLE_RATE

reco = pd.DataFrame(sample.T)    
reco.columns = all_channels
   
fig, ax = plt.subplots()
for channel in range(len(all_channels)):
    ch_data = np.asarray(reco[all_channels[channel]])

    ## band pass
    bands = fb.filter_bank(ch_data, constants.SAMPLE_RATE, FILTER_RANGES, order=5)
     # Number of samplepoints
    N = 1500
    # sample spacing
    T = 1.0 / 250.0
    x = np.linspace(0.0, N*T, N)
    
    for band in range(len(bands)):
        y = bands[band]
        yf = scipy.fftpack.fft(y)
        xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
        
    
        ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
        plt.show()
        
FILTER_RANGES = [[1,4],[4,8],[8,16],[16,32],[32,45]]

computed_fft = np.array([ np.linspace(0.0, 1.0/(2.0*T), N/2),2.0/N * np.abs(yf[:N//2])])
indx1  = computed_fft[0,:] > 0 
indx2 = computed_fft[0,:] < 4
both = indx1*indx2
gamma = computed_fft[1,both]

from scipy.signal import savgol_filter
from scipy.signal import find_peaks

savgol_filter(gamma, 11, 3).mean()
gamma.mean()
smooth = savgol_filter(gamma, 11, 3)

peaks, _ = find_peaks(smooth, width=5)
print(peaks)
plt.plot(gamma)
plt.plot(smooth)
plt.plot(peaks, smooth[peaks], "x")
plt.show()










