# -*- coding: utf-8 -*-
"""
@author: %(Mikel Val Calvo)s
@email: %(mikel1982mail@gmail.com)
@institution: %(Dpto. de Inteligencia Artificial, Universidad Nacional de Educación a Distancia (UNED))
@DOI: 10.5281/zenodo.3759306 
"""
#%%
import numpy as np
#from scipy import signal
from neurodsp import spectral
from lspopt.lsp import spectrogram_lspopt

class spectrum():
    def __init__(self, constants):
        self.constants = constants
        
    def get_spectrum(self, samples):
        spectrums = []
        for i in range(self.constants.NDIMS):
             freqs, spectre  = spectral.compute_spectrum(samples[i,:], self.constants.SAMPLE_RATE)
             spectrums.append(spectre)
        return freqs, np.asarray(spectrums)
    
    def get_spectrogram(self, samples):
#        _, _, Sxx = signal.spectrogram(samples, self.constants.SAMPLE_RATE)
        _,_, Sxx = spectrogram_lspopt(samples, self.constants.SAMPLE_RATE, c_parameter=20.0)
        return Sxx
        
