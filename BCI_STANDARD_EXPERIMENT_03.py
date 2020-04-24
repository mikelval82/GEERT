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
from GENERAL.recording_manager_01 import recording_manager
############# EXTERNAL LIBRARIES ######################################
from PyQt5 import QtWidgets
from multiprocessing import Queue, Value
############# APP DEFINITION #########################################
class MyApp(QtWidgets.QApplication):
    
    def __init__(self):     
        QtWidgets.QApplication.__init__(self,[''])     
        ############# LOGIC CONTROL ##################
        self.isconnected = Value('b',1) 
        ############### INIT CONSTANTS DEFINITION ###########################
        self.constants = constants()
        ######### slots manager for multiple callbacks settings #############                                          
        self.slots = SlotsManager()
        ########### queue ###########
        self.queue = Queue()
        ##### TRIGGER SERVER ############
        self.trigger_server = trigger_server(self.constants.ADDRESS, self.constants.PORT)
        ################ BUFFER  ####################     
        self.buffer = buffer(self.constants)
        self.buffer.emitter.connect(self.slots.trigger)
        ################ INIT DATA MANAGER #####################
        self.eeg_dmg = data_manager_openBCI(self)  
        self.eeg_dmg.start()
        ###### INIT tcp/ip INTERFACE FOR DATA ACQUISITION
        self.recording_manager = recording_manager(self)
        ################# INIT GUI ################################
        self.gui = GUI(self, callbacks = [self.connection_manager, self.recording_manager.test_acquisition, self.recording_manager.update_state, self.saveFileDialog, self.openFileNameDialog])
        ########## LOGGER ####################
        self.log = log.logger(self.gui)
        ####### INIT DRIVER ###########
        self.driver = openBCI(self.queue, self.recording_manager.streaming, self.isconnected, self.log)
        self.driver.start()

    def connection_manager(self):        
        if not self.isconnected.value:
            self.driver.connect() 
            self.driver.enable_filters()
        else:
            self.driver.disconnect()
             
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

