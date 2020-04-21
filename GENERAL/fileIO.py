# -*- coding: utf-8 -*-
"""
@author: %(Mikel Val Calvo)s
@email: %(mikel1982mail@gmail.com)
@institution: %(Dpto. de Inteligencia Artificial, Universidad Nacional de Educación a Distancia (UNED))
"""
from EDF.writeEDFFile import edf_writter
from multiprocessing import Process

class io_manager():
    
    def __init__(self, app):
        self.app = app
        self.edf = edf_writter(self.app.constants)
        
    # -- EDF files -- #
    def create_file(self):
        self.edf.new_file(self.app.constants.PATH + '_trial_' + str(self.app.constants.running_trial) + '.edf')
        self.app.log.update_text('* -- USER ' + self.app.constants.PATH + ' CREATED -- *')
        
    def close_file(self):
        self.edf.close_file()
        self.app.log.update_text('* -- USER ' + self.app.constants.PATH + ' CLOSED -- *')
        
    def append_to_file(self, all_data_store):# tarda mucho en guardar, probar hilos o guardar en variable allData hasta terminar registro y luego guardar en archivo
        if self.app.constants.ispath:
            # save EDF trial file
            self.edf.append(all_data_store)
            
            p = Process(target=self.edf.writeToEDF())
            p.start()
            
        else:
            print('* -- EDF file path is needed -- *')

    
    def online_annotation(self, notation):
        instant = self.app.constants.running_window*self.app.constants.SECONDS + (self.app.buffer.cur % self.app.buffer.size_short)/self.app.constants.SAMPLE_RATE
        duration = -1
        event = notation
        self.edf.annotation(instant, duration, event)