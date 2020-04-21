#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: %(Mikel Val Calvo)s
@email: %(mikel1982mail@gmail.com)
@institution: %(Dpto. de Inteligencia Artificial, Universidad Nacional de Educación a Distancia (UNED))
"""
from FEATURES.online_features_02 import compute_online_features
from FEATURES.feature_selection import rfe_selection
from FILTERS.EAWICA import eawica
from CLASSIFIERS import models_trainer 
from GENERAL import csv_fileIO as io
import numpy as np
from mne.io import read_raw_edf
from GENERAL.constants_02 import constants
import matplotlib.pyplot as plt

#%% LOAD DATA
fs = 250
seconds = 6
sample_length = fs*seconds
constants = constants()

#%%
SUBJECT = 'PAQUI'

path_labels = './data/' + SUBJECT + '/labels_' + SUBJECT + '_modified.csv'
labels_byTrial = io.open_csvFile(path_labels)
labels_byTrial = np.asarray(labels_byTrial)[:,0]
trials = np.arange(1,len(labels_byTrial)+1)

EEG_trials = []

for trial in trials:
    path_training = './data/' + SUBJECT + '/short_films_trial_' + str(trial) + '.edf'
    
    raw_train = read_raw_edf(path_training)
    EEG_data = raw_train.get_data()
    EEG_trials.append( EEG_data[:,sample_length:-sample_length] )

#%%############## COMPUTE FEATURES ###################################
# -- init training data     
numFeatures = 5
  
features = []
labels = []
trials_labels_map = []

step = fs*1# seconds of sliding step * fs
fs = 250
seconds = 6
sample_length = fs*seconds


for trial in np.arange(0,len(labels_byTrial)):
    
    n_samples = int(EEG_trials[trial].shape[1]/step - sample_length/step)
    
    for n in np.arange(0,n_samples):
        ini = n*step
        end = ini+sample_length
    
        sample = EEG_trials[trial][:,ini:end] 
        sample = eawica(sample,constants)
        
        features.append( compute_online_features(sample,constants) )
        labels.append( labels_byTrial[trial] )  
        trials_labels_map.append( trial )
        

features = np.asarray(features)
labels = np.asarray(labels)

#%%####### DATASET PREPROCESSING

from sklearn.preprocessing import QuantileTransformer
scaler1 = QuantileTransformer(output_distribution='normal')
features_quantile = scaler1.fit_transform(features)

from sklearn.preprocessing import MinMaxScaler
scaler2 = MinMaxScaler()
features_minmax = scaler2.fit_transform(features_quantile)

from FEATURES.feature_smoothing import smoothing
features_smooth = np.zeros(features_quantile.shape)
for i in range(features.shape[1]):
    features_smooth[:,i] = smoothing(features_minmax[:,i])
    
   
plt.close('all')
plt.figure()
plt.subplot(311)
plt.plot(labels)
plt.xlim([0,features.shape[0]])
plt.subplot(312)
plt.imshow(np.squeeze(features_quantile).T,interpolation='nearest', aspect='auto',cmap = 'seismic')
plt.subplot(313)
plt.imshow(np.squeeze(features_smooth).T, interpolation='nearest', aspect='auto',cmap = 'seismic')
#%% SAVE DATA
np.save('./data/' + SUBJECT + '/features_smooth.npy', features_smooth)
np.save('./data/' + SUBJECT + '/labels.npy', labels)
np.save('./data/' + SUBJECT + '/trials_labels_map.npy', trials_labels_map)
#%% LOAD DATA
features_smooth = np.load('./data/' + SUBJECT + '/features_smooth.npy')
labels = np.load('./data/' + SUBJECT + '/labels.npy')
trials_labels_map = np.load('./data/' + SUBJECT + '/trials_labels_map.npy')

#%%########## DATASET SMOOTHING #######################
label_names = ['POSITIVE','NEGATIVE']
emotions = [4,6]

from FEATURES.features_train_test_split import train_test_split, get_aleatory_one_trial_out

reordered_data = train_test_split(features_smooth, labels, emotions, trials_labels_map)  

feature_evolution = []
for i in range(19,20):
    OneTrialOut = []
    for indx in range(20):
        X_train,y_train,X_test,y_test = get_aleatory_one_trial_out(reordered_data,emotions,indx)
        
        select_features = i
        _, selected_features_indx = rfe_selection(X_train, y_train, select_features)  
        # -- model training -- #  
        classifiers, names, predictions, scores_list = models_trainer.classify(X_train[:,selected_features_indx,], y_train, X_test[:,selected_features_indx], y_test)
        winner = np.asarray(scores_list).argmax()
        print(models_trainer.get_report(classifiers[winner], X_test[:,selected_features_indx], y_test, label_names))
        f1_score = models_trainer.get_f1_score(classifiers[winner], X_test[:,selected_features_indx], y_test)
        
        OneTrialOut.append(f1_score)
        
    feature_evolution.append(np.mean(OneTrialOut))

#plt.figure()
#plt.plot(feature_evolution)
#plt.title('positive against negative')
#%%########## DATASET SMOOTHING #######################
label_names = ['RELAX','INTENSE']
emotions = [2,8]

from FEATURES.features_train_test_split import train_test_split, get_aleatory_one_trial_out

reordered_data = train_test_split(features_smooth, labels, emotions, trials_labels_map)  

feature_evolution = []
for i in range(19,20):
    OneTrialOut = []
    for indx in range(20):
        X_train,y_train,X_test,y_test = get_aleatory_one_trial_out(reordered_data,emotions,indx)
        
        select_features = i
        _, selected_features_indx = rfe_selection(X_train, y_train, select_features)  
        # -- model training -- #  
        classifiers, names, predictions, scores_list = models_trainer.classify(X_train[:,selected_features_indx,], y_train, X_test[:,selected_features_indx], y_test)
        winner = np.asarray(scores_list).argmax()
        print(models_trainer.get_report(classifiers[winner], X_test[:,selected_features_indx], y_test, label_names))
        f1_score = models_trainer.get_f1_score(classifiers[winner], X_test[:,selected_features_indx], y_test)
        
        OneTrialOut.append(f1_score)
        
    feature_evolution.append(np.mean(OneTrialOut))
    
plt.figure()
plt.plot(feature_evolution)
plt.title('relax against intense')
#%%########## FEATURE SELECTION AND MODEL TRAINING ##############
label_names = ['NEGATIVE', 'POSITIVE', 'RELAX']
emotions = [4,6,2]

from FEATURES.features_train_test_split import train_test_split, get_one_trial_out, get_aleatory_one_trial_out

reordered_data = train_test_split(features_smooth, labels, emotions, trials_labels_map)  

feature_evolution = []
for i in range(19,20):
    OneTrialOut = []
    for indx in range(20):
        X_train,y_train,X_test,y_test = get_aleatory_one_trial_out(reordered_data,emotions,indx)
        
        select_features = i
        _, selected_features_indx = rfe_selection(X_train, y_train, select_features)  
        # -- model training -- #  
        classifiers, names, predictions, scores_list = models_trainer.classify(X_train[:,selected_features_indx,], y_train, X_test[:,selected_features_indx], y_test)
        winner = np.asarray(scores_list).argmax()
        
        print(models_trainer.get_report(classifiers[winner], X_test[:,selected_features_indx], y_test, label_names))
        f1_score = models_trainer.get_f1_score(classifiers[winner], X_test[:,selected_features_indx], y_test)
        
        OneTrialOut.append(f1_score)
        
    feature_evolution.append(np.mean(OneTrialOut))
    
plt.figure()
plt.plot(feature_evolution)
plt.title('all mental states')
#%% DATASET TO PANDAS DATAFRAME FOR EDA ANALYSIS
import pandas as pd

variables = ['DE','AMP','PFD','HJ','FI']
bandas = ['delta','theta','alpha','beta','gamma']
electrodos = ['P7', 'T7', 'F7', 'F3', 'P8', 'T8', 'F8', 'F4']
columns =[]

for i in range(200):
    columns.append( variables[i%5] + '_' +  bandas[int(i/5)%5] + '_' + electrodos[int(i/25)] )
    
features_smooth_pd = pd.DataFrame(features_smooth,columns=columns)

features_smooth_pd.head()

#%% BOXPLOT FOR STATISTICAL ANALYSIS --- SIN RESOLVER !!!!
import seaborn as sns
#from statannot_master.statannot import add_stat_annotation
from scipy.stats import wilcoxon, shapiro
from collections import OrderedDict

sns.set(style="white")

features_ch = ['_delta_','_theta_','_alpha_','_beta_','_gamma_']

feature_names = ['DE','AMP','PFD','HJ','FI']
electrode_names = ['P7', 'T7', 'F7', 'F3', 'P8', 'T8', 'F8', 'F4']

columns = ['pos','neg','rel','ten']

combinations = [('positive','negative'),('relax','tense')]
indx_positive = np.where(labels == 6)[0]
indx_negative = np.where(labels == 4)[0]
indx_relax = np.where(labels == 2)[0]
indx_tense = np.where(labels == 8)[0]

plt.close('all')
for electrode in range(8):
    for f_range in range(5):
        for feature in range(5):
            variables  = []
            variable = feature_names[feature] + features_ch[f_range] + electrode_names[electrode] 
    
            variables.append( features_smooth_pd[variable].iloc[indx_positive] )
            variables.append( features_smooth_pd[variable].iloc[indx_negative]) 
            variables.append( features_smooth_pd[variable].iloc[indx_relax] )
            variables.append( features_smooth_pd[variable].iloc[indx_tense] )
            
            aux = {'positive' : variables[0].values ,'negative' : variables[1].values, 'relax':variables[2].values,'tense':variables[3].values}
            df = pd.DataFrame.from_dict(aux, orient='index').transpose() 


#%% CORRELATION MATRIX FOR EDA ANALYSIS

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="white")

def compute_matrix_corr(pd,title):
    corr = pd.corr()
    
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    f.suptitle(title)
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
features_ch = ['_delta_','_theta_','_alpha_','_beta_','_gamma_']

feature_names = ['DE','AMP','PFD','HJ','FI']
electrode_names = ['P7', 'T7', 'F7', 'F3', 'P8', 'T8', 'F8', 'F4']
plt.close('all')
for i in range(len(feature_names)-1):
    for j in range(len(electrode_names)-1):
        c1 = features_smooth_pd.loc[:, [feature_names[i] + s + electrode_names[j] for s in features_ch]]     
        c2 = features_smooth_pd.loc[:, [feature_names[i] + s + electrode_names[j+1] for s in features_ch]]
        title = feature_names[i] + '_' + electrode_names[j]  + '_VS_' + feature_names[i] + '_' + electrode_names[j+1]
        compute_matrix_corr( pd.concat([c1,c2], axis=1), title)
#%% RANDOM FOREST VARIABLE IMPORTANCE
plt.close('all')
from sklearn.ensemble import ExtraTreesClassifier

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(X_train, y_train)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
xticks_columns = []
for f in range(X_train.shape[1]):
    print("%d. feature: %s (%f)" % (f + 1, columns[indices[f]], importances[indices[f]]))
    xticks_columns.append(columns[indices[f]])


# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), xticks_columns, rotation=45)
plt.xlim([-1, 50])
plt.show()
#%% DECISION TREE VISUALIZATION 
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(classifiers[3], out_file=dot_data, feature_names=columns, 
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())

## Create PDF
#graph.write_pdf("INES.pdf")
#
## Create PNG
graph.write_png("INES.png")
