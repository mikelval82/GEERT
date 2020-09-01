# -*- coding: utf-8 -*-
"""
@author: %(Mikel Val Calvo)s
@email: %(mikel1982mail@gmail.com)
@institution: %(Dpto. de Inteligencia Artificial, Universidad Nacional de Educación a Distancia (UNED))
@DOI: 10.5281/zenodo.3759306 
"""
#%%
#%%############## MY IMPORTS #############################################
from COM.trigger_server_2 import trigger_server
from DYNAMIC import dynamic as Dyn_import
from GENERAL.data_manager_openBCI_04 import data_manager_openBCI 
from LOGGING import logger as log
from GENERAL.ring_buffer_02 import RingBuffer as buffer
from GENERAL.constants_02 import constants
from COM.open_bci_GCPDS_02 import OpenBCIBoard as openBCI
from GUI.GUI_bci_03 import GUI 
from GENERAL.slots_manager import SlotsManager
from GENERAL.fileIO import io_manager 
############# EXTERNAL LIBRARIES ######################################
from PyQt5 import QtWidgets
from multiprocessing import Queue, Value
############# APP DEFINITION #########################################
class MyApp(QtWidgets.QApplication):
    
    def __init__(self):     
        QtWidgets.QApplication.__init__(self,[''])     
        ############# LOGIC CONTROL ##################
        self.isconnected = Value('b',1) 
        self.trigger_server_activated = False
        self.streaming = Value('b',0)
        ############### INIT CONSTANTS DEFINITION ###########################
        self.constants = constants()
        ######### slots manager for multiple callbacks settings #############                                          
        self.slots = SlotsManager()
        ########### queue ###########
        self.queue = Queue()
        ################ BUFFER  ####################     
        self.buffer = buffer(self.constants)
        self.buffer.emitter.connect(self.slots.trigger)
        ################ INIT DATA MANAGER #####################
        self.eeg_dmg = data_manager_openBCI(self.constants, self.queue, self.buffer, self.slots)  
        self.eeg_dmg.start()
       ################# INIT GUI ################################
        self.gui = GUI(self, callbacks = [self.connection_manager, self.test_acquisition, self.launch_trigger_server, self.saveFileDialog, self.openFileNameDialog])
        ########## LOGGER ####################
        self.log = log.logger(self.gui.bci_graph.logger)
        ####### INIT DRIVER ###########
        self.driver = openBCI(self.queue, self.streaming, self.isconnected, self.log)
        self.driver.start()
#        ###### INIT tcp/ip INTERFACE FOR DATA ACQUISITION
#        self.recording_manager = recording_manager(self.driver, self.eeg_dmg, self.gui, self.constants, self.buffer, self.log)
#       
    def connection_manager(self):        
        if not self.isconnected.value:
            self.driver.connect() 
            self.driver.enable_filters()
        else:
            self.driver.disconnect()
            
    def launch_trigger_server(self):
        if self.trigger_server_activated:
            self.trigger_server.close_socket()
            del self.trigger_server
        else:
            self.trigger_server = trigger_server(self.constants)
            self.trigger_server.socket_emitter.connect(self.update_state)
            self.trigger_server.log_emitter.connect(self.log.update_text)
            self.trigger_server.create_socket()   
            self.trigger_server.start()  
            
        self.trigger_server_activated = not self.trigger_server_activated
    
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
            
    def saveFileDialog(self):    
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self.gui.MainWindow,"QFileDialog.getSaveFileName()","","EDF Files (*.edf)", options=options)
        if fileName:
            self.constants.PATH = fileName
            self.constants.ispath = True
            
    def openFileNameDialog(self, btn):    
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileType = "PYTHON Files (*.py)"
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self.gui.MainWindow,"QFileDialog.getOpenFileName()","",fileType, options=options)       
        #----------------- LOAD AND EXECUTE THE MODULE -----#
        Dyn_import.load_module(fileName, self)
              
    def execute_gui(self):
        self.exec()

## -- RUN THE APP -- ##
if __name__ == "__main__":
    main = MyApp()
    main.execute_gui()

