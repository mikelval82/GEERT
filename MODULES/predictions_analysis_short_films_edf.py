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
SUBJECT = 'RUBEN'

from data.RUBEN import labels_metadata
labels_byTrial = labels_metadata.labels['mylabels']
trials = np.arange(1,len(labels_byTrial)+1)

EEG_trials = []

for trial in trials:
    path_training = './data/' + SUBJECT + '/short_films_trial_' + str(trial) + '.edf'
    
    raw_train = read_raw_edf(path_training)
    EEG_data = raw_train.get_data()
    EEG_trials.append( EEG_data[:,sample_length:-sample_length] )
    
    print(EEG_data[:,sample_length:-sample_length].shape)
    print(EEG_trials[trial-1].shape)
#%%############## COMPUTE FEATURES ###################################
# -- init training data     
numFeatures = 5
  
#features = []
labels = []
trials_labels_map = []

step = fs*3# seconds of sliding step * fs
fs = 250
seconds = 6
sample_length = fs*seconds


for trial in np.arange(0,len(labels_byTrial)):
    
    n_samples = int(EEG_trials[trial].shape[1]/step - sample_length/step)
    print('trial ', trial, ' label ', labels_byTrial[trial], ' samples ', n_samples)
    if n_samples > 0:
        for n in np.arange(0,n_samples):
            ini = n*step
            end = ini+sample_length
#        
#            sample = EEG_trials[trial][:,ini:end] 
#            sample = eawica(sample,constants)
#            
#            features.append( compute_online_features(sample,constants) )
            labels.append( labels_byTrial[trial] )  
            trials_labels_map.append( trial )
            

features = np.asarray(features)
labels = np.asarray(labels)

print(features.shape)
print(labels.shape)
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
import numpy as np
features_smooth = np.load('./data/' + SUBJECT + '/features_smooth.npy')
labels = np.load('./data/' + SUBJECT + '/labels.npy')
trials_labels_map = np.load('./data/' + SUBJECT + '/trials_labels_map.npy')

print(features_smooth.shape)
#%%########## DATASET SMOOTHING #######################
from CLASSIFIERS import models_trainer 
from FEATURES.feature_selection import kbest_selection
from FEATURES.features_train_test_split import get_aleatory_k_trials_out
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


emotions = [2,4,6,8]
label_names = [str(i) for i in emotions]

scaler2 = MinMaxScaler()
features_minmax = scaler2.fit_transform(features_smooth)

all_labels = pd.DataFrame(labels)
all_features = pd.DataFrame(features_minmax)
old_indx = pd.DataFrame(trials_labels_map)

print(all_labels.shape)
print(all_features.shape)
print(old_indx.shape)

feature_evolution = {'mean':[],'std':[]}
best = (0,0)
check = [5,20,50,100,150]
for i in check:
    OneTrialOut = []
    for indx in range(20):
        print('numfeatures: ', i, ' iteration: ', indx)
        X_train,y_train,X_test,y_test = get_aleatory_k_trials_out(all_features,all_labels,emotions,old_indx,size=1)
        select_features = i
        _, selected_features_indx = kbest_selection(X_train, y_train, select_features)  
        # -- model training -- #  
        classifiers, names, predictions, scores_list = models_trainer.classify(X_train[:,selected_features_indx,], y_train, X_test[:,selected_features_indx], y_test)
        winner = np.asarray(scores_list).argmax()
        print(models_trainer.get_report(classifiers[winner], X_test[:,selected_features_indx], y_test, label_names))
        f1_score = models_trainer.get_f1_score(classifiers[winner], X_test[:,selected_features_indx], y_test)
        
        OneTrialOut.append(f1_score)
    print('f1 score mean: ', np.mean(OneTrialOut), ' std: ', np.std(OneTrialOut))
    
    if best[0] < np.mean(OneTrialOut):
        best = (np.mean(OneTrialOut), np.std(OneTrialOut))
        
    feature_evolution['mean'].append(np.mean(OneTrialOut))
    feature_evolution['std'].append(np.std(OneTrialOut))

#
numfeatures = [str(value) for value in list(check)]
x_pos = np.arange(len(numfeatures))
CTEs = feature_evolution['mean']
error = feature_evolution['std']

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_xticks(x_pos)
ax.set_xticklabels(numfeatures)
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.show()
#%%########## DATASET SMOOTHING #######################
from CLASSIFIERS import models_trainer 

label_names = ['RELAX','INTENSE']
emotions = [2,8]

from FEATURES.features_train_test_split import train_test_split, get_aleatory_k_trials_out

reordered_data = train_test_split(features_smooth, labels, emotions, trials_labels_map)  

feature_evolution = []
for i in [50,100,150]:
    OneTrialOut = []
    for indx in range(5):
        X_train,y_train,X_test,y_test = get_aleatory_k_trials_out(reordered_data,emotions,size=3)
        
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
from CLASSIFIERS import models_trainer 

label_names = ['NEGATIVE', 'POSITIVE', 'RELAX','TENSE']
emotions = [4,6,2,8]

from FEATURES.features_train_test_split import train_test_split, get_aleatory_k_trials_out

reordered_data = train_test_split(features_smooth, labels, emotions, trials_labels_map)  

feature_evolution = []
for i in [50,100,150]:
    OneTrialOut = []
    for indx in range(5):
        X_train,y_train,X_test,y_test = get_aleatory_k_trials_out(reordered_data,emotions,size=2)
        
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
#%%
from CLASSIFIERS import models_trainer as mt1
from CLASSIFIERS import models_trainer as mt2

emotions_arousal = [2,8]
emotions_valence = [4,6]

from FEATURES.features_train_test_split import train_test_split, get_aleatory_k_trials_out

reordered_data_arousal = train_test_split(features_smooth, labels, emotions_arousal, trials_labels_map)  
reordered_data_valence = train_test_split(features_smooth, labels, emotions_valence, trials_labels_map)  

for i in [50,100,150]:
    select_features = i
    
    for indx in range(5):    
        # -- arousal modelling -- #
        X_train_arousal,y_train_arousal,X_test_arousal,y_test_arousal = get_aleatory_k_trials_out(reordered_data_arousal,emotions_arousal,size=2)
        _, selected_features_indx_arousal = rfe_selection(X_train_arousal, y_train_arousal, select_features)  
        classifiers_arousal, names_arousal, predictions_arousal, scores_list_arousal = mt1.classify(X_train_arousal[:,selected_features_indx_arousal,], y_train_arousal, X_test_arousal[:,selected_features_indx_arousal], y_test_arousal)
        winner_arousal = np.asarray(scores_list_arousal).argmax()
        print(names_arousal[winner])
        # -- valence modelling -- #
        X_train_valence,y_train_valence,X_test_valence,y_test_valence = get_aleatory_k_trials_out(reordered_data_valence,emotions_valence,size=2)
        _, selected_features_indx_valence = rfe_selection(X_train_valence, y_train_valence, select_features)  
        classifiers_valence, names_valence, predictions_valence, scores_list_valence = mt2.classify(X_train_valence[:,selected_features_indx_valence,], y_train_valence, X_test_valence[:,selected_features_indx_valence], y_test_valence)
        winner_valence = np.asarray(scores_list_valence).argmax()
        print(names_valence[winner])
        # -- validation report -- #
        X_test = np.vstack((X_test_arousal[:,selected_features_indx_arousal], X_test_valence[:,selected_features_indx_valence]))
        Y_test = np.hstack((y_test_arousal,y_test_valence))
 
        predictions = np.zeros((Y_test.shape[0],4))
        cont = 0
        for sample in X_test:
            arousal_prediction = classifiers_arousal[winner_arousal].predict_proba(sample.reshape(1, -1))
            valence_prediction = classifiers_valence[winner_valence].predict_proba(sample.reshape(1, -1))
            
            predictions[cont] = np.hstack((arousal_prediction, valence_prediction))
            cont+=1
        
        plt.figure()
        plt.subplot(211)
        plt.plot(predictions[:,0],'c')
        plt.plot(predictions[:,1],'m')
        plt.subplot(212)
        plt.plot(predictions[:,2],'g')
        plt.plot(predictions[:,3],'r')
        
        


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
