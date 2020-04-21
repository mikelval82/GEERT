#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 10:42:31 2019

@author: anaiak
"""

from __future__ import division, print_function, absolute_import

import os
import pyedflib

class edf_writter:
    
    def __init__(self, constants):
        self.constants = constants
        
    def new_file(self,path):
        data_file = os.path.join('.', path)
        self.file = pyedflib.EdfWriter(data_file, self.constants.CHANNELS, file_type=pyedflib.FILETYPE_EDFPLUS)
        
        self.channel_info = []
        self.data_list = []
        
    def append(self, all_data_store):
        for channel in range(self.constants.CHANNELS):
            ch_dict = {'label': self.constants.CHANNEL_IDS[channel], 'dimension': 'uV', 'sample_rate': self.constants.SAMPLE_RATE, 'physical_max': all_data_store[channel,:].max(), 'physical_min': all_data_store[channel,:].min(), 'digital_max': 32767, 'digital_min': -32768, 'transducer': '', 'prefilter':''}
            self.channel_info.append(ch_dict)
            self.data_list.append(all_data_store[channel,:])

    def writeToEDF(self):
        self.file.setSignalHeaders(self.channel_info)
        self.file.writeSamples(self.data_list)
        
    def annotation(self, instant, duration, event):
        self.file.writeAnnotation(instant, duration, event)
        
    def close_file(self):
        self.file.close()
        del self.file
        
    def __del__(self):
        print("deleted")