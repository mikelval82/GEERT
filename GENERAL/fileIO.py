# -*- coding: utf-8 -*-
"""
@author: %(Mikel Val Calvo)s
@email: %(mikel1982mail@gmail.com)
@institution: %(Dpto. de Inteligencia Artificial, Universidad Nacional de Educación a Distancia (UNED))
@DOI: 10.5281/zenodo.3759306 
"""
#%%
from EDF.writeEDFFile import edf_writter
from multiprocessing import Process

class io_manager():
    
    def __init__(self, constants, buffer, log):
        self.constants = constants
        self.buffer = buffer
        self.log = log
        self.edf = edf_writter(self.constants)
        
    # -- EDF files -- #
    def create_file(self):
        self.edf.new_file(self.constants.PATH + '_trial_' + str(self.constants.running_trial) + '.edf')
        self.log.update_text('* -- USER ' + self.constants.PATH + ' CREATED -- *')
        
    def close_file(self):
        self.edf.close_file()
        self.log.update_text('* -- USER ' + self.constants.PATH + ' CLOSED -- *')
        
    def append_to_file(self, all_data_store):# tarda mucho en guardar, probar hilos o guardar en variable allData hasta terminar registro y luego guardar en archivo
        if self.constants.ispath:
            # save EDF trial file
            self.edf.append(all_data_store)
            
            p = Process(target=self.edf.writeToEDF())
            p.start()
            
        else:
            print('* -- EDF file path is needed -- *')

    
    def online_annotation(self, notation):
        instant = self.constants.running_window*self.constants.SECONDS + (self.buffer.cur % self.buffer.size_short)/self.constants.SAMPLE_RATE
        duration = -1
        event = notation
        self.edf.annotation(instant, duration, event)
