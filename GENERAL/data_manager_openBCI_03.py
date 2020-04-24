# -*- coding: utf-8 -*-
"""
@author: %(Mikel Val Calvo)s
@email: %(mikel1982mail@gmail.com)
@institution: %(Dpto. de Inteligencia Artificial, Universidad Nacional de Educación a Distancia (UNED))
@DOI: 10.5281/zenodo.3759306 
"""
#%%
from EDF.writeEDFFile import edf_writter
from FILTERS.filter_bank_manager import filter_bank_class
from FILTERS.spectrum import spectrum 
from FILTERS import AICAW, EAWICA
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
        self.io = edf_writter(self.app.constants)
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
            filtered = EAWICA.eawica(filtered, self.app.constants)
        elif method == 'AICAW':
            filtered = AICAW.aicaw(filtered, self.app.constants)
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
        
    def create_file(self):
        self.io.new_file(self.app.constants.PATH + '_trial_' + str(self.app.constants.running_trial) + '.edf')
        self.app.log.update_text('* -- USER ' + self.app.constants.PATH + ' CREATED -- *')
        
    def close_file(self):
        self.io.close_file()
        self.app.log.update_text('* -- USER ' + self.app.constants.PATH + ' CLOSED -- *')
        
    def reset_data_store(self):
        self.all_data_store = np.empty(shape=(self.app.constants.CHANNELS, 0))
        
    def online_annotation(self, notation):
        instant = self.app.constants.running_window*self.app.constants.SECONDS + (self.app.buffer.cur % self.app.buffer.size_short)/self.app.constants.SAMPLE_RATE
        duration = -1
        event = notation
        self.io.annotation(instant, duration, event)
        
    def append_to_store(self):
        sample_data = self.get_short_sample(self.app.constants.METHOD)
        self.all_data_store = np.hstack((self.all_data_store, sample_data))  
#        instant = self.app.constants.running_window*self.app.constants.SECONDS
#        duration = -1
#        self.app.log.update_text('* -- last action in eeg_dmg: ' + str(self.app.constants.last_action) + ' -- *')
#        event = self.app.constants.last_action
#        self.io.annotation(instant, duration, event)
        self.app.constants.running_window += 1
        
    def append_to_file(self):# tarda mucho en guardar, probar hilos o guardar en variable allData hasta terminar registro y luego guardar en archivo
        if self.app.constants.ispath:
            # save EDF trial file
            self.io.append(self.all_data_store)
            self.io.writeToEDF()
            # re-initialize
            self.all_data_store = np.empty(shape=(self.app.constants.CHANNELS, 0))
            # update metadata
            self.app.constants.running_trial += 1
            if self.app.constants.isstored:
                self.app.constants.isstored = False
        else:
            self.app.log.update_text('* -- EDF file path is needed -- *')
        
        
        
        
        
        
