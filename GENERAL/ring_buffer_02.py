# -*- coding: utf-8 -*-
"""
@author: %(Mikel Val Calvo)s
@email: %(mikel1982mail@gmail.com)
@institution: %(Dpto. de Inteligencia Artificial, Universidad Nacional de Educación a Distancia (UNED))
@DOI: 10.5281/zenodo.3759306 
"""
#%%
from PyQt5 import QtCore 
import numpy as np

class RingBuffer(QtCore.QThread):
    """ class that implements a not-yet-full buffer """
    emitter = QtCore.pyqtSignal()
    
    def __init__(self, constants, parent=None):
        super(RingBuffer, self).__init__(parent)
        self.constants = constants
        self.channels = self.constants.CHANNELS
        self.max = self.constants.LARGE_WINDOW 
        self.size_short = self.constants.WINDOW 
        self.data = np.zeros((self.channels,self.max))
        self.cur = self.max
        self.full = False
         
    def reset(self, size_short):
        self.size_short = size_short
        self.data = np.zeros((self.channels,self.max))
        self.cur = self.max
        
    def append(self,x):
        """append an element at the end of the buffer"""  
        self.cur = self.cur % self.max
        self.data[:,self.cur] = np.asarray(x).transpose()
        self.cur = self.cur+1
            
        if (self.cur % self.size_short) == 0:
            self.emitter.emit()  
            print('full myfriend: ', self.cur, 'short window size: ', self.size_short)              

    def get(self):
        """ Return a list of elements from the oldest to the newest. """
        return np.hstack((self.data[:,self.cur:], self.data[:,:self.cur]))
    
    
