# -*- coding: utf-8 -*-
"""
@author: %(Mikel Val Calvo)s
@email: %(mikel1982mail@gmail.com)
@institution: %(Dpto. de Inteligencia Artificial, Universidad Nacional de Educación a Distancia (UNED))
@DOI: 10.5281/zenodo.3759306 
"""
#%%
import sys 
import os

class constants():
    def __init__(self, seconds=6, sample_rate=250, baud=115200, channels=8, ndims=8, signal='eeg', lowcut=1, highcut=45, order=5):
        ############### CONSTANTS ######################
        self.ADDRESS = '10.1.28.117'
        self.PORT = 10000
        self.SECONDS = seconds
        self.SAMPLE_RATE = sample_rate
        self.NOTCH = 50#Hz
        self.BAUD = baud
        self.WINDOW = self.SAMPLE_RATE * self.SECONDS
        self.LARGE_WINDOW = self.SAMPLE_RATE * 60# 1 minute of visualization        
        self.CHANNELS = channels
        self.NDIMS = ndims
        self.SIGNAL = signal
        self.LOWCUT = lowcut
        self.HIGHCUT = highcut
        self.ORDER = order
        self.METHOD = 'Butterworth'
        self.FILTER_RANGES = [[1,4],[4,8],[8,16],[16,32],[32,45]]
        self.CHANNEL_IDS = ['P7','T7', 'F7', 'F3', 'P8', 'T8', 'F8', 'F4']
        self.AVAILABLE_CHANNELS = [True,True,True,True,True,True,True,True]
        self.PATH = sys.path.append(os.path.realpath('./RESULTS/'))
        # dinamic variables
        self.last_action = 5
        self.pos_ini = self.LARGE_WINDOW - self.WINDOW #- 2*self.SAMPLE_RATE
        self.pos_end = self.LARGE_WINDOW #- 2*self.SAMPLE_RATE
        self.running_trial = 0
        self.running_window = 0    
        self.ispath = False
        self.refresh_rate = 1/sample_rate
        self.short_refresh_rate = 1/sample_rate
       
    def update(self, name, value):
        if name == 'seconds':
            self.SECONDS = value   
            self.WINDOW = self.SAMPLE_RATE * self.SECONDS
            self.pos_ini = self.LARGE_WINDOW - self.WINDOW - self.SAMPLE_RATE/2
            self.pos_end = self.LARGE_WINDOW - self.SAMPLE_RATE/2
        elif name == 'order':
            self.ORDER = value
        elif name == 'method':
            self.METHOD = value
            if value != 'Butterworth':
                self.short_refresh_rate = 0.05#50 ms for refresh plots
            else:
                self.short_refresh_rate = self.refresh_rate
        
    def set_filter_range(self, activated):
        if activated == 'Full':
            self.LOWCUT, self.HIGHCUT = 1, 45 
        elif activated == 'Delta':
            self.LOWCUT, self.HIGHCUT = self.FILTER_RANGES[0]    
        elif activated == 'Theta':
            self.LOWCUT, self.HIGHCUT = self.FILTER_RANGES[1]   
        elif activated == 'Alpha':
            self.LOWCUT, self.HIGHCUT = self.FILTER_RANGES[2]    
        elif activated == 'Beta':
            self.LOWCUT, self.HIGHCUT = self.FILTER_RANGES[3]   
        elif activated == 'Gamma':
            self.LOWCUT, self.HIGHCUT = self.FILTER_RANGES[4]    
          
        
