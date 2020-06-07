# -*- coding: utf-8 -*-
"""
@author: %(Mikel Val Calvo)s
@email: %(mikel1982mail@gmail.com)
@institution: %(Dpto. de Inteligencia Artificial, Universidad Nacional de Educación a Distancia (UNED))
@DOI: 10.5281/zenodo.3759306 
"""
#%%
from FILTERS.filter_bank_manager import filter_bank_class
from FILTERS.spectrum import spectrum 
from FILTERS import EAWICA

from threading import Thread, Lock
import time 
import numpy as np

class data_manager_openBCI(Thread):
    
    def __init__(self, app):
        Thread.__init__(self) 
        ### APP REFERENCE ################
        self.app = app
        ### data ###########
        self.all_data_store = np.empty(shape=(self.app.constants.CHANNELS, 0))
        ########### SHARED QUEUE ###########
        self.app.slots.append(self.append_to_store)
        self.filter_bank = filter_bank_class(self.app.constants)
        self.filter_bank.update_filters()
        self.spectrum = spectrum(self.app.constants)
        ##### Mutex #####
        self.muttex = Lock()
     
    def run(self):     
        while True:  
            time.sleep(0.0001)
            while not self.app.queue.empty(): 
                self.muttex.acquire()
                sample = self.app.queue.get()
                self.app.buffer.append(sample)
                self.muttex.release()
                
    def init_filters(self):
        self.filter_bank.update_filters()
        
    def get_sample(self): 
        self.muttex.acquire()
        filtered = self.filter_bank.pre_process( self.app.buffer.get() )
        self.muttex.release()
        return filtered
    
    def get_short_sample(self, method): 
        self.muttex.acquire()
        filtered = self.filter_bank.pre_process( self.app.buffer.get() )
        filtered = filtered[:,int(self.app.constants.pos_ini):int(self.app.constants.pos_end)]  
        if method == 'EAWICA':
            try:
                filtered = EAWICA.eawica(filtered, self.app.constants)
            except:
                pass
        self.muttex.release()
        return filtered

    def get_powerSpectrum(self, method):
        self.muttex.acquire()
        filtered = self.filter_bank.pre_process( self.app.buffer.get() )
        freqs, spectra = self.spectrum.get_spectrum( filtered )
        self.muttex.release()
        return freqs, spectra
    
    def get_powerSpectrogram(self, method, channel):
        self.muttex.acquire()
        filtered = self.filter_bank.pre_process( self.app.buffer.get() )
        spectrogram = self.spectrum.get_spectrogram( filtered[channel,:])
        self.muttex.release()
        return spectrogram
    
    def append_to_store(self):
        sample_data = self.get_short_sample(self.app.constants.METHOD)
        self.all_data_store = np.hstack((self.all_data_store, sample_data))  
        self.app.constants.running_window += 1
    
    def reset_data_store(self):       
        self.all_data_store = np.empty(shape=(self.app.constants.CHANNELS, 0))
        self.app.constants.running_trial += 1
 

        
    
        
        
        
        
        
        
