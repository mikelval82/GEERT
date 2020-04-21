#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: %(Mikel Val Calvo)s
@email: %(mikel1982mail@gmail.com)
@institution: %(Dpto. de Inteligencia Artificial, Universidad Nacional de Educación a Distancia (UNED))
"""
#%%
from __future__ import print_function

import matplotlib.pyplot as plt
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
import numpy as np
from FEATURES.features_train_test_split import reorder_lstm, get_in_order_train_test

batch_size=16

all_features = np.load('./data/ALBA/peliculas/cienmetros/all_valence_features_normalized.npy')
all_labels = np.load('./data/ALBA/peliculas/cienmetros/all_valence_labels.npy')
old_indx = np.load('./data/ALBA/peliculas/cienmetros/old_indx_valence.npy')
print(all_features.shape)
print(all_labels.shape)
print(old_indx.shape)

num_group = 5
step = 1
all_features_temporal, all_labels_new, old_indx_new = reorder_lstm(all_features,all_labels,old_indx,num_group,step)
print(all_features_temporal.shape)
print(all_labels_new.shape)
print(old_indx_new.shape)
size=1

emotions = [4,6]
which_scene = 0
test_set, X_train, y_train, X_test, y_test = get_in_order_train_test(all_features_temporal,all_labels_new.squeeze(),emotions,old_indx_new.squeeze(),which_scene=which_scene)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

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
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(5,200)))
model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# fit network
model.fit(X_train, y_train, epochs=100, batch_size=100, verbose=1)
# evaluate model
_, accuracy = model.evaluate(X_test, y_test, batch_size=100, verbose=0)
print('accuracy: ', accuracy * 100.0)

