#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: %(Mikel Val Calvo)s
@email: %(mikel1982mail@gmail.com)
@institution: %(Dpto. de Inteligencia Artificial, Universidad Nacional de Educación a Distancia (UNED))
"""
from multiprocessing import Value

class recording_manager:
    
    def __init__(self, app):
        self.streaming = Value('b',0)
        self.app = app
        
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
            self.app.eeg_dmg.create_file()
            self.app.eeg_dmg.reset_data_store()
            self.app.eeg_dmg.online_annotation(action)
            # init driver and gui updating
            self.app.driver.send_start()
            self.app.gui.eeg_timer.start(self.app.constants.refresh_rate) 
            self.app.gui.eeg_short_timer.start(self.app.constants.short_refresh_rate) 
            self.app.gui.freq_timer.start(self.app.constants.refresh_rate) 
        elif action == 'stop':
            self.app.log.update_text('Stop recording trial: ' + str(self.app.constants.running_trial))
            self.app.eeg_dmg.online_annotation(action)
            # stop driver and gui updating
            self.app.driver.send_stop()
            self.app.gui.eeg_timer.stop()    
            self.app.gui.eeg_short_timer.stop() 
            self.app.gui.freq_timer.stop()  
            # update file
            self.app.eeg_dmg.append_to_store()
            self.app.eeg_dmg.append_to_file()
            self.app.eeg_dmg.close_file()
        else:
            self.app.eeg_dmg.online_annotation(action)
            self.app.constants.last_action = action
        
    
        