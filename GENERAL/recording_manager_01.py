# -*- coding: utf-8 -*-
"""
@author: %(Mikel Val Calvo)s
@email: %(mikel1982mail@gmail.com)
@institution: %(Dpto. de Inteligencia Artificial, Universidad Nacional de Educación a Distancia (UNED))
@DOI: 10.5281/zenodo.3759306 
"""
#%%
from multiprocessing import Value
from GENERAL.fileIO import io_manager 

class recording_manager:
    
    def __init__(self, driver, eeg_dmg, gui, constants, buffer, log):
        self.streaming = Value('b',0)
        self.driver = driver
        self.constants = constants
        self.gui = gui
        self.log = log
        self.eeg_dmg = eeg_dmg
        self.io = io_manager(constants, buffer, log)
        
    def test_acquisition(self):
        if not self.streaming.value:
            # init driver and gui updating
            self.driver.send_start()
            self.gui.eeg_timer.start(self.constants.refresh_rate) 
            self.gui.eeg_short_timer.start(self.constants.short_refresh_rate) 
            self.gui.freq_timer.start(self.constants.refresh_rate) 
        else:
            # stop driver and gui updating
            self.driver.send_stop()
            self.gui.eeg_timer.stop()   
            self.gui.eeg_short_timer.stop() 
            self.gui.freq_timer.stop()  
       
    def update_state(self, action):
        if not self.streaming.value and action == 'start':
            self.log.update_text('Start recording trial: ' + str(self.constants.running_trial))
            # update metadata
            self.constants.running_window = 0
            # new file for new trial
            if self.constants.ispath:
                self.io.create_file()
                self.io.online_annotation(action)
            else:
                self.log.update_text('No user filename has been defined!!')
            # init driver and gui updating
            self.eeg_dmg.reset_data_store()
            self.driver.send_start()
            self.gui.eeg_timer.start(self.constants.refresh_rate) 
            self.gui.eeg_short_timer.start(self.constants.short_refresh_rate) 
            self.gui.freq_timer.start(self.constants.refresh_rate) 
        elif action == 'stop':
            self.log.update_text('Stop recording trial: ' + str(self.constants.running_trial))
            # stop driver and gui updating
            self.driver.send_stop()
            self.gui.eeg_timer.stop()    
            self.gui.eeg_short_timer.stop() 
            self.gui.freq_timer.stop()  
            # update file
            self.eeg_dmg.append_to_store()
            if self.constants.ispath:
                self.io.online_annotation(action)
                self.io.append_to_file( self.eeg_dmg.all_data_store )
                self.io.close_file()
            else:
                self.log.update_text('No user filename was defined, so no data has been stored!!')
        else:
            self.eeg_dmg.online_annotation(action)
            self.constants.last_action = action
        
    
        
