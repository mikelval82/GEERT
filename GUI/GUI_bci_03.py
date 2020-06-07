# -*- coding: utf-8 -*-
"""
@author: %(Mikel Val Calvo)s
@email: %(mikel1982mail@gmail.com)
@institution: %(Dpto. de Inteligencia Artificial, Universidad Nacional de Educación a Distancia (UNED))
@DOI: 10.5281/zenodo.3759306 
"""
#%%
from QTDesigner.bci_biosignals_01 import Ui_MainWindow as ui
from PyQt5 import QtWidgets, QtCore
from qwt.qt.QtGui import QFont
from qwt import  QwtText
import pyqtgraph as pg
import numpy as np

class GUI():
    
    def __init__(self, app, callbacks):
        
        self.app = app
        ################    data managers  #############################
        self.curves_EEG = []
        self.lr = None
        self.spectrogram_Img = None
        self.curves_Freq = []
        self.curves_EEG_short = []
        ############## EEG gui design ##############################
        self.MainWindow = QtWidgets.QMainWindow()
        self.bci_graph = ui() 
        self.bci_graph.setupUi(self.MainWindow)
        self.bci_graph.WindowsSize_spinBox.setRange(0,12)
        self.initLongTermViewCurves()
        self.initShortTermViewCurves()
        self.initFrequencyView()
        self.set_plots()
        self.initFrequencyComboBox()
        self.initFilteringComboBox()
        self.initSpectrogramComboBox()
        self.load_style()
        self.MainWindow.show()
        # Enable antialiasing for prettier plots
        pg.setConfigOptions(antialias=True)
        ########### set callbacks ###########
        self.bci_graph.btn_connect.clicked.connect(callbacks[0])
        self.bci_graph.btn_start.clicked.connect(callbacks[1])
        self.bci_graph.btn_trigger.clicked.connect(lambda: self.launch_trigger_server(callbacks[2]))
        self.bci_graph.btn_user.clicked.connect(callbacks[3])
        self.bci_graph.btn_loadScript.clicked.connect(callbacks[4])
        self.bci_graph.frequency_comboBox.currentIndexChanged.connect(lambda: self.set_frequency())
        self.bci_graph.filtering_comboBox.currentIndexChanged.connect(lambda: self.set_filtering())
        self.bci_graph.WindowsSize_spinBox.valueChanged.connect(lambda: self.set_sampleSize())
        self.bci_graph.butterOrder_spinBox.valueChanged.connect(lambda: self.set_order())
        self.bci_graph.Spectrogram_radioButton.toggled.connect(lambda: self.set_channel_spectrogram())
        self.bci_graph.Spectrogram_comboBox.currentIndexChanged.connect(lambda: self.set_channel_spectrogram())
        ############# set timers for updating plots ############3
        self.eeg_timer = QtCore.QTimer()
        self.eeg_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.eeg_timer.timeout.connect(self.eeg_update)
        
        self.eeg_short_timer = QtCore.QTimer()
        self.eeg_short_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.eeg_short_timer.timeout.connect(self.eeg_short_update)
        
        self.freq_timer = QtCore.QTimer()
        self.freq_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.freq_timer.timeout.connect(self.freq_update)        
        
######### SIGNAL VISUALIZATION UPDATE MANAGERS ########################################    
    def eeg_update(self):       
        sample = self.app.eeg_dmg.get_sample()
        
        for i in range(self.app.constants.CHANNELS):
            self.curves_EEG[i].setData(sample[i,:])
        
    def freq_update(self):
        freqs, spectra = self.app.eeg_dmg.get_powerSpectrum(self.app.constants.METHOD)  
        if self.bci_graph.Spectrogram_radioButton.isChecked():
            channel = self.app.constants.CHANNEL_IDS.index(self.bci_graph.Spectrogram_comboBox.currentText())
            spectrogram = self.app.eeg_dmg.get_powerSpectrogram(self.app.constants.METHOD, channel) 
            ini = int(self.app.constants.pos_ini/self.app.constants.SAMPLE_RATE)
            # end = int(self.app.constants.pos_end/self.app.constants.SAMPLE_RATE)
            self.spectrogram_Img.setImage(spectrogram[:,:].T, autoLevels=True)
            #self.spectrogram_Img.setImage(spectrogram[:,ini:end].T, autoLevels=True)
        else:
            freqs, spectra = self.app.eeg_dmg.get_powerSpectrum(self.app.constants.METHOD)  
            for i in range(self.app.constants.CHANNELS):
                self.curves_Freq[i].setData(freqs,np.log10(spectra[i,:])) 

    def eeg_short_update(self):
        sample = self.app.eeg_dmg.get_short_sample(self.app.constants.METHOD)
        for i in range(self.app.constants.CHANNELS):
            self.curves_EEG_short[i].setData(sample[i,:])#
            
################## TRIGGER MANAGER ###########################################################
    def launch_trigger_server(self, callback):
        if self.app.trigger_server.activated:
            self.app.trigger_server.close_socket()
        else:
            self.app.trigger_server.create_socket()   
            self.app.trigger_server.start()
            self.app.trigger_server.new_COM1.connect(callback) 
            
############### BUTTON LISTENERS ###########################################################          
    def set_channel_spectrogram(self):
        self.initFrequencyView()
        
    def set_frequency(self):
        self.app.constants.set_filter_range(self.bci_graph.frequency_comboBox.currentText())  
        if self.app.constants.LOWCUT != None:
            self.app.eeg_dmg.init_filters()
            
    def set_filtering(self):
        if self.app.streaming.value:
            self.eeg_timer.stop()      
            self.freq_timer.stop()
            self.eeg_short_timer.stop()
        self.app.constants.update('method', self.bci_graph.filtering_comboBox.currentText())  
        if self.app.streaming.value:
            self.eeg_timer.start(self.app.constants.refresh_rate) 
            self.freq_timer.start(self.app.constants.refresh_rate) 
            self.eeg_short_timer.start(self.app.constants.short_refresh_rate) 

    def set_sampleSize(self):
        self.app.constants.update('seconds', int(self.bci_graph.WindowsSize_spinBox.value()))
        self.app.eeg_dmg.buffer.reset(self.app.constants.WINDOW)
        self.set_plots(reset = True)
    
    def set_order(self):
        self.app.constants.update('order', int(self.bci_graph.butterOrder_spinBox.value()))
        
 ###################### GUI SETTINGS ##################################################   
    def eeg_short_view(self):
        self.app.constants.pos_ini, self.app.constants.pos_end = self.lr.getRegion() 
        self.bci_graph.Emotions_plot.setXRange(0, int(self.app.constants.pos_end - self.app.constants.pos_ini))
        self.bci_graph.Emotions_plot.setLimits(xMin=0, xMax=int(self.app.constants.pos_end - self.app.constants.pos_ini))

    def set_plots(self, reset = False):
        channels = self.app.constants.CHANNEL_IDS
        ### EEG plot settings ###
        self.bci_graph.EEG_plot.setLabel('bottom', 'Samples', units='n')
        self.bci_graph.EEG_plot.getAxis('left').setTicks([[(100, channels[0]), (200, channels[1]), (300, channels[2]), (400, channels[3]), (500, channels[4]), (600, channels[5]), (700, channels[6]), (800, channels[7])]])
        self.bci_graph.EEG_plot.setYRange(0, 900)
        self.bci_graph.EEG_plot.setXRange(0, self.app.constants.LARGE_WINDOW)
        self.bci_graph.EEG_plot.showGrid(True, True, alpha = 0.3)
        self.bci_graph.EEG_plot.setLimits(xMin=0, xMax=self.app.constants.LARGE_WINDOW)
        # Linear region settings #
        if not reset:
            self.lr = pg.LinearRegionItem([self.app.constants.pos_ini,self.app.constants.pos_end])
            self.bci_graph.EEG_plot.addItem(self.lr)
            self.lr.sigRegionChanged.connect(self.eeg_short_view)
            self.eeg_short_view()
        else:
            self.lr.setRegion([self.app.constants.pos_ini,self.app.constants.pos_end])
        ### EEG short view Plot settings ###
        self.bci_graph.Emotions_plot.setLabel('bottom', 'Samples', units='n')
        self.bci_graph.Emotions_plot.getAxis('left').setTicks([[(100, channels[0]), (200, channels[1]), (300, channels[2]), (400, channels[3]), (500, channels[4]), (600, channels[5]), (700, channels[6]), (800, channels[7])]])
        self.bci_graph.Emotions_plot.setYRange(0, 900)
        self.bci_graph.Emotions_plot.setXRange(0, int(self.app.constants.pos_end - self.app.constants.pos_ini))
        self.bci_graph.Emotions_plot.showGrid(True, True, alpha = 0.3)
        self.bci_graph.Emotions_plot.setLimits(xMin=0, xMax=int(self.app.constants.pos_end - self.app.constants.pos_ini))

    def load_style(self):        
        self.styleQwtPlot('EEG', self.bci_graph.EEG_plot)
        self.styleQwtPlot('Frequency', self.bci_graph.Frequency_plot)
        self.styleQwtPlot('Emotion estimation', self.bci_graph.Emotions_plot)
        
        with open("QTDesigner/style.css") as f:
            self.app.setStyleSheet(f.read())
            
    def styleQwtPlot(self, name, elem):
        font = QFont()
        font.setPixelSize(24)
        title = QwtText(name)
        title.setFont(font)
        elem.setTitle(title)
        
    def initFrequencyComboBox(self):
        self.bci_graph.frequency_comboBox.addItems(['Full','Delta','Theta','Alpha','Beta','Gamma'])
        
    def initSpectrogramComboBox(self):
        self.bci_graph.Spectrogram_comboBox.addItems(self.app.constants.CHANNEL_IDS)
       
    def initFilteringComboBox(self):
        self.bci_graph.filtering_comboBox.addItems(['Butterworth','EAWICA','AICAW'])
       
    def initLongTermViewCurves(self):
        ########################### EEG #####################################
        for i in range(self.app.constants.CHANNELS):
            c = pg.PlotCurveItem(pen=(i,self.app.constants.CHANNELS*1.3))
            c.setPos(0,0)
            self.bci_graph.EEG_plot.addItem(c)
            self.curves_EEG.append(c)
            
    def initShortTermViewCurves(self):
        ############ EEG short view ##########################
        for i in range(self.app.constants.CHANNELS):
            c = pg.PlotCurveItem(pen=(i,self.app.constants.CHANNELS*1.3))
            c.setPos(0,0)
            self.bci_graph.Emotions_plot.addItem(c)
            self.curves_EEG_short.append(c)
            
    def initFrequencyView(self):
        self.curves_Freq = []
        self.bci_graph.Frequency_plot.clear()
        
        if self.bci_graph.Spectrogram_radioButton.isChecked():
            self.bci_graph.Frequency_plot.showGrid(True, True, alpha = 0)
            self.bci_graph.Frequency_plot.setLogMode(False, False)
            self.bci_graph.Frequency_plot.setLabel('left', 'Frequency', units='Hz')
            self.bci_graph.Frequency_plot.setLabel('bottom', "Samples", units='n')
            
            self.spectrogram_Img = pg.ImageItem()     
            self.bci_graph.Frequency_plot.addItem(self.spectrogram_Img)

            pos = np.array([0.0, 0.5, 1.0])
            color = np.array([[0,0,0,255], [255,128,0,255], [255,255,0,255]], dtype=np.ubyte)
            map = pg.ColorMap(pos, color)
            lut = map.getLookupTable(0.0, 1.0, 256)           
            self.spectrogram_Img.setLookupTable(lut)
            
        else:   
            ### FREQUENCY Plot settings ###
            self.bci_graph.Frequency_plot.showGrid(True, True, alpha = 0.3)
            self.bci_graph.Frequency_plot.setLogMode(False, True)
            self.bci_graph.Frequency_plot.setLabel('left', 'Amplitude', units='dB')
            self.bci_graph.Frequency_plot.setLabel('bottom', "Frequency", units='Hz')
            
            for i in range(self.app.constants.CHANNELS):
                c = pg.PlotCurveItem(pen=(i,self.app.constants.CHANNELS*1.3))
                self.bci_graph.Frequency_plot.addItem(c)
                self.curves_Freq.append(c)
    
            
        
        

