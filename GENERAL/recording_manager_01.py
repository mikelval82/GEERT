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
    
    def __init__(self, app):
        self.streaming = Value('b',0)
        self.app = app
        self.io = io_manager(self.app)
        
    def test_acquisition(self):
        if not self.streaming.value:
            # init driver and gui updating
            self.app.driver.send_start()
            self.app.gui.eeg_timer.start(self.app.constants.refresh_rate) 
            self.app.gui.eeg_short_timer.start(self.app.constants.short_refresh_rate) 
            self.app.gui.freq_timer.start(self.app.constants.refresh_rate) 
        else:
            # stop driver and gui updating
            self.app.driver.send_stop()
            self.app.gui.eeg_timer.stop()   
            self.app.gui.eeg_short_timer.stop() 
            self.app.gui.freq_timer.stop()  
       
    def update_state(self, action):
        if not self.streaming.value and action == 'start':
            self.app.log.update_text('Start recording trial: ' + str(self.app.constants.running_trial))
            # update metadata
            self.app.constants.running_window = 0
            # new file for new trial
            if self.app.constants.ispath:
                self.io.create_file()
                self.io.online_annotation(action)
            else:
                self.app.log.update_text('No user filename has been defined!!')
            # init driver and gui updating
            self.app.eeg_dmg.reset_data_store()
            self.app.driver.send_start()
            self.app.gui.eeg_timer.start(self.app.constants.refresh_rate) 
            self.app.gui.eeg_short_timer.start(self.app.constants.short_refresh_rate) 
            self.app.gui.freq_timer.start(self.app.constants.refresh_rate) 
        elif action == 'stop':
            self.app.log.update_text('Stop recording trial: ' + str(self.app.constants.running_trial))
            # stop driver and gui updating
            self.app.driver.send_stop()
            self.app.gui.eeg_timer.stop()    
            self.app.gui.eeg_short_timer.stop() 
            self.app.gui.freq_timer.stop()  
            # update file
            self.app.eeg_dmg.append_to_store()
            if self.app.constants.ispath:
                self.io.online_annotation(action)
                self.io.append_to_file( self.app.eeg_dmg.all_data_store )
                self.io.close_file()
            else:
                self.app.log.update_text('No user filename was defined, so no data has been stored!!')
        else:
            self.app.eeg_dmg.online_annotation(action)
            self.app.constants.last_action = action
        
    
        
