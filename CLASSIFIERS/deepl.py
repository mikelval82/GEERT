#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: %(Mikel Val Calvo)s
@email: %(mikel1982mail@gmail.com)
@institution: %(Dpto. de Inteligencia Artificial, Universidad Nacional de Educación a Distancia (UNED))
"""

from FEATURES.online_features_02 import compute_online_features
from FILTERS.EAWICA import eawica
from GENERAL import csv_fileIO as io
import numpy as np
from mne.io import read_raw_edf
from GENERAL.constants_02 import constants

#%% LOAD DATA
fs = 250
seconds = 6
sample_length = fs*seconds
constants = constants()

#%%
SUBJECT = 'INES'

path_labels = './data/' + SUBJECT + '/labels_' + SUBJECT.lower() + '.csv'
labels_byTrial = io.open_csvFile(path_labels)
labels_byTrial = np.asarray(labels_byTrial)[:,0]
trials = np.arange(1,len(labels_byTrial)+1)

EEG_trials = []

for trial in trials:
    path_training = './data/' + SUBJECT + '/test_1_trial_' + str(trial) + '.edf'
    
    raw_train = read_raw_edf(path_training)
    EEG_data = raw_train.get_data()
    EEG_trials.append( EEG_data[:,sample_length:-sample_length] )
    

#%%



    
import matplotlib.pyplot as plt
from scipy.io import wavfile
import resin
import seaborn
seaborn.set_style('white')

trial = 1
sr = 250
sample = EEG_trials[trial-1][:,1500:-1500]    
constants.WINDOW = sample.shape[1]
data = eawica(sample,constants)

data = data[0,:]

spa2 = resin.Spectra(sr,
                     NFFT=1024,
                     noverlap=1000,
                     data_window=data.shape[0],
                     n_tapers=5,
                     NW=1.8,
                     freq_range=(0, 250))
spa2.signal(data)
spa2.spectrogram()


#%%
features = []
labels = []
trials_labels_map = []

step = fs*1# seconds of sliding step * fs
fs = 250
seconds = 6
sample_length = fs*seconds


for trial in np.arange(0,len(labels_byTrial)):
    if labels_byTrial[trial] == 4 or labels_byTrial[trial] == 6:
        n_samples = int(EEG_trials[trial].shape[1]/step - sample_length/step)
        for n in np.arange(0,n_samples):
            ini = n*step
            end = ini+sample_length
            
            sample = EEG_trials[trial][:,ini:end] 
            sample = eawica(sample,constants)
            
            features.append( sample)
            labels.append( labels_byTrial[trial] )  
            trials_labels_map.append( trial )
        
train_features = np.asarray(features)
train_labels = np.asarray(labels)
train_trials_labels_map = trials_labels_map

#%%
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
 
#%% conv1d model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(8,1500)))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#%%
from FEATURES.features_train_test_split import get_aleatory_k_trials_out
import pandas as pd

emotions = [4,6]
all_labels = pd.DataFrame(labels)
all_features = pd.DataFrame(features)
old_indx = pd.DataFrame(trials_labels_map)

print(all_labels.shape)
print(all_features.shape)
print(old_indx.shape)


X_train,y_train,X_test,y_test = get_aleatory_k_trials_out(all_features,all_labels,emotions,old_indx,size=1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# fit network
model.fit(X_train, y_train, epochs=100, batch_size=100, verbose=1)
# evaluate model
_, accuracy = model.evaluate(X_test, y_test, batch_size=100, verbose=0)
print('accuracy: ', accuracy * 100.0)

#%% load data

 

