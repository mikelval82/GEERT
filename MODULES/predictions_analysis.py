#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: %(Mikel Val Calvo)s
@email: %(mikel1982mail@gmail.com)
@institution: %(Dpto. de Inteligencia Artificial, Universidad Nacional de Educación a Distancia (UNED))
"""

import numpy as np
from FEATURES.feature_selection import rfe_selection
from GENERAL.constants_02 import constants
from FILTERS.EAWICA import eawica

from FEATURES.online_features_02 import compute_online_features
from GENERAL import csv_fileIO as io
import matplotlib.pyplot as plt
#%% TRAINNING DATA
# -- set paths --
constants = constants(seconds=4)
selected_features_indx = None
numBasicFeatures = 5
numBands = 5
numFeatures = numBasicFeatures*numBands*8
#%%
path_training = './data/NESA/nesa.csv'
path_labels = './data/NESA/labels_nesa.csv'

#%%############  load trainning data #############################
dataframe = io.open_csvFile(path_training)
labels_byTrial = io.open_csvFile(path_labels)
#%%
############## compute features ###################################
numSamples = int(dataframe.shape[0]/constants.CHANNELS)
# -- init training data
#features = np.zeros((numSamples, numFeatures))
#labels = np.zeros((numSamples,))

features = []
labels = []
anterior = 0
contador = 0
for i in range(numSamples):
    # -- indexing training data
    ini = i*constants.CHANNELS
    end = ini+constants.CHANNELS
    
    
    if anterior == dataframe['trial'].iloc[ini] and contador >= 2:
        sample = dataframe.iloc[ini:end,3:]
#        sample = eawica(np.asarray(sample),constants)
        features.append(compute_online_features(sample,constants,numFeatures))
        labels.append( labels_byTrial['label'].iloc[ dataframe['trial'].iloc[i*constants.CHANNELS] ]     )
        contador+=1
    
    elif anterior == dataframe['trial'].iloc[ini] and contador < 2:
        contador+=1
    else:
        anterior = dataframe['trial'].iloc[ini]
        contador = 0
    
features = np.asarray(features)
labels = np.asarray(labels)
  
print(features.shape)
print(labels.shape)
#%%
from FEATURES.features_train_test_split import ByTrials_train_test_split
_, _, features, labels = ByTrials_train_test_split(features, labels, 15, 1)
#%% PROCESSO 1 - COMPROBAR QUE SE CONSIGURE ENTRENAR UN MODELO INTERTRIALS
#%%####### DATASET PREPROCESSING

from sklearn.preprocessing import QuantileTransformer
scaler1 = QuantileTransformer(output_distribution='normal')

features_quantile = scaler1.fit_transform(features)

from FEATURES.feature_smoothing import smoothing
features_smooth = np.zeros(features_quantile.shape)
for i in range(features.shape[1]):
    features_smooth[:,i] = smoothing(features_quantile[:,i])
    
plt.close('all')
plt.figure()
plt.subplot(311)
plt.plot(labels)
plt.xlim([0,features.shape[0]])
plt.subplot(312)
plt.imshow(np.squeeze(features_quantile).T,interpolation='nearest', aspect='auto',cmap = 'seismic')
plt.subplot(313)
plt.imshow(np.squeeze(features_smooth).T, interpolation='nearest', aspect='auto',cmap = 'seismic')


#%%############# feature selecion ###########
from FEATURES.features_train_test_split import ByTrials_train_test_split
X_train, y_train, X_test, y_test = ByTrials_train_test_split(features_smooth, labels, 15, 9)

from CLASSIFIERS import models_trainer 

label_names = ['POS', 'NEU','NEG']
scores = []
models = []
best_model = {'model':[],'name':[],'predictions':[],'score':0,'selected_features':[],'report':[]}
for i in range(features.shape[1]):
    _, selected_features_indx = rfe_selection(X_train, y_train, (i+1))
    classifiers, names, predictions, scores_list = models_trainer.classify(X_train[:,selected_features_indx], y_train, X_test[:,selected_features_indx], y_test)
    winner = np.asarray(scores_list).argmax()
    print(np.asarray(scores_list).max())
    report = models_trainer.get_report(classifiers[winner], X_test[:,selected_features_indx], y_test,label_names)
    print(report)
    scores.append(scores_list[winner])
    models.append(names[winner])
    if scores_list[winner] > best_model['score']:
        best_model['model'] = classifiers[winner]
        best_model['name'] = names[winner]
        best_model['predictions'] = predictions[winner]
        best_model['score'] = scores_list[winner]
        best_model['selected_features'] = selected_features_indx
        best_model['report'] = models_trainer.get_report(classifiers[winner], X_test[:,selected_features_indx], y_test, label_names)
        
print(best_model['report'])  
#%%
from scipy.signal import find_peaks

peaks, _ = find_peaks(np.asarray(scores)*100, height=60)

plt.close('all')
plt.figure()
plt.style.use('seaborn')
plt.plot(np.asarray(scores)*100)
plt.ylim([40,80])
plt.xlim([1,200])
plt.xlabel('Number of features (n)',size=30)
plt.ylabel('Accuracy (%)',size=30)
plt.title('Best model and feature selection process',size=40)
plt.tick_params(labelsize=20)

from matplotlib.lines import Line2D

legend_elements = [Line2D([0], [0], marker='^', color='sienna', lw=4, label='Linear SVM',markerfacecolor='sienna', markersize=15),
                   Line2D([0], [0], marker='s', color='gold', lw=4, label='RBF SVM',markerfacecolor='gold', markersize=15),
                   Line2D([0], [0], marker='X', color='olivedrab', lw=4, label='Decision Tree',markerfacecolor='olivedrab', markersize=15),
                   Line2D([0], [0], marker='D', color='darkcyan', lw=4, label='Random Forest',markerfacecolor='darkcyan', markersize=15),
             ]
for peak in peaks:
    if models[peak] == 'Linear SVM':
        plt.scatter(peak, scores[peak]*100, s=200,marker='^',c='sienna')
    if models[peak] == 'RBF SVM':
        plt.scatter(peak, scores[peak]*100, s=200,marker='s',c='gold')
    if models[peak] == 'Decision Tree':
        plt.scatter(peak, scores[peak]*100, s=200,marker='X',c='olivedrab')
    if models[peak] == 'Random Forest':
        plt.scatter(peak, scores[peak]*100, s=200,marker='D',c='darkcyan')
   
plt.legend(handles=legend_elements, fontsize =20)

print(best_model['report'])  

model,name = models_trainer.train_models(features_smooth[:,best_model['selected_features']],labels, best_model['name'])
   
#%% PROCESSO 3: PROBANDO CON LOS TRIALS DEL TEST REAL

path_training = './data/FANI/fani_test.csv'
path_labels = './data/FANI/labels_fani_test_modified.csv'


#%%############  load trainning data #############################
dataframe_test = io.open_csvFile(path_training)
labels_byTrial_test = io.open_csvFile(path_labels)
#%% 

numSamples = int(dataframe_test.shape[0]/constants.CHANNELS)
# -- init training data
features_test = []
labels_test = []
anterior = 0
contador = 0
for i in range(numSamples):
    # -- indexing training data
    ini = i*constants.CHANNELS
    end = ini+constants.CHANNELS
    
    
    if anterior == dataframe_test['trial'].iloc[ini] and contador > 1:
        sample = dataframe_test.iloc[ini:end,3:]
        sample = eawica(np.asarray(sample),constants)
        features_test.append(compute_online_features(sample,constants,numFeatures))
        labels_test.append( labels_byTrial_test['label'].iloc[ dataframe_test['trial'].iloc[i*constants.CHANNELS] ]     )
        contador+=1
    elif anterior == dataframe_test['trial'].iloc[ini] and contador <= 1:
        contador+=1
    else:
        anterior = dataframe_test['trial'].iloc[ini]
        contador = 0

features_test = np.asarray(features_test)
labels_test = np.asarray(labels_test)
  
print(features_test.shape)
print(labels_test.shape)
#%%####### DATASET PREPROCESSING

from sklearn.preprocessing import QuantileTransformer
scaler2 = QuantileTransformer(output_distribution='normal')

features_test_quantile = scaler2.fit_transform(features_test)

from FEATURES.feature_smoothing import smoothing
features_test_smooth = np.zeros(features_test_quantile.shape)
for i in range(features_test_quantile.shape[1]):
    features_test_smooth[:,i] = smoothing(features_test_quantile[:,i])
    
plt.close('all')
plt.figure()
plt.subplot(311)
plt.plot(labels_test)
plt.xlim([0,features_test.shape[0]])
plt.subplot(312)
plt.imshow(np.squeeze(features_test_quantile).T,interpolation='nearest', aspect='auto',cmap = 'seismic')
plt.subplot(313)
plt.imshow(np.squeeze(features_test_smooth).T, interpolation='nearest', aspect='auto',cmap = 'seismic')

#%% 

predictions = np.zeros((labels_test.shape[0], 3))

X_train_toSmooth = np.copy(features_quantile)

cont = 0
for label in labels_test:
    sample = scaler1.transform(features_test[cont,:].reshape(1, -1))
    
    aux = np.vstack((X_train_toSmooth,sample))
    X_train_toSmooth = np.vstack((X_train_toSmooth,sample))
    for i in range(features_test.shape[1]):
        aux[:,i] = smoothing(aux[:,i])
    sample = aux[-1,:]
    
    predictions[cont,:] = model.predict_proba(sample[best_model['selected_features'] ].reshape(1, -1))
    cont+=1

#%%

features_quantile = scaler1.fit_transform(features_test)
   
features_smooth = np.zeros(features_quantile.shape)
for i in range(features.shape[1]):
    features_smooth[:,i] = smoothing(features_quantile[:,i])
    
report = models_trainer.get_report(model, features_smooth[:,best_model['selected_features']], labels_test, label_names)
print(report) 
#%%
plt.close('all')
from FEATURES.feature_smoothing import smoothing

fig, axes = plt.subplots(2,1,constrained_layout=True, sharex=True, sharey=True)
plt.style.use('seaborn')

alpha = .8
axes[0].fill_between(np.arange(0.0, features_test.shape[0]),predictions[:,2],color='c',alpha=alpha)
axes[0].fill_between(np.arange(0.0, features_test.shape[0]),predictions[:,1],color='grey',alpha=alpha)
axes[0].fill_between(np.arange(0.0, features_test.shape[0]),predictions[:,0],color='orange',alpha=alpha)

axes[0].set_ylabel('Accuracy (%)', color='k', size=20)
axes[0].tick_params(axis='y', labelcolor='k')
axes[0].set_ylim([0,2])
axes[0].set_yticks( (0, 0.5, 1))
axes[0].tick_params(axis='both', which='major', labelsize=20)

legend = axes[0].legend(('negative', 'neutral', 'positive'))  
legend.get_frame().set_edgecolor('w')
axes[0].set_title('Instantaneous prediction', size=20)
ax0 = axes[0].twinx()
ax0.set_ylabel('label', color='k', size=20)
ax0.plot(labels_test,linewidth=7.0, color='k')
ax0.tick_params(axis='y', labelcolor='k')
ax0.set_ylim([-0.01,2.02])
ax0.set_yticklabels(['','positive','','','','neutral','','','','negative'],fontsize=15)


axes[1].fill_between(np.arange(0.0, features_test.shape[0]),smoothing(predictions[:,2],19,3),color='c',alpha=alpha)
axes[1].fill_between(np.arange(0.0, features_test.shape[0]),smoothing(predictions[:,1],19,3),color='grey',alpha=alpha)
axes[1].fill_between(np.arange(0.0, features_test.shape[0]),smoothing(predictions[:,0],19,3),color='orange',alpha=alpha)

axes[1].set_xlabel('Samples (s)', size=20)
axes[1].set_ylabel('Accuracy (%)', color='k', size=20)
axes[1].set_ylim([0,2])
axes[1].set_yticks( (0, 0.5, 1))
axes[1].tick_params(axis='y', labelcolor='k')
axes[1].tick_params(axis='both', which='major', labelsize=20)
legend = axes[1].legend(('negative', 'neutral','positive'))  

ax1 = axes[1].twinx()
ax1.set_ylabel('label', color='k', size=20)
ax1.plot(labels_test,linewidth=7.0, color='k')
ax1.tick_params(axis='y', labelcolor='k')
ax1.set_ylim([-0.01,2.02])
ax1.set_yticklabels(['','positive','','','','neutral','','','','negative'],fontsize=15)


legend.get_frame().set_edgecolor('w')
ax1.set_title('Smoothed prediction', size=20)

fig.suptitle("Online prediction", fontsize=30)
#%%




