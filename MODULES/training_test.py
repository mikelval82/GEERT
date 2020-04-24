# -*- coding: utf-8 -*-
"""
@author: %(Mikel Val Calvo)s
@email: %(mikel1982mail@gmail.com)
@institution: %(Dpto. de Inteligencia Artificial, Universidad Nacional de Educación a Distancia (UNED))
@DOI: 10.5281/zenodo.3759306 
"""
#%%
#%% General imports
from FILMS_ANALYSIS_ESCENAS import utils_general_prueba
from GENERAL.constants_02 import constants
import pandas as pd
import numpy as np


constants = constants()
constants.AVAILABLE_CHANNELS = [True,True,True,True,True,True,True,False]

#%% set subject 
subject = 'test'
full_module_name = 'prueba.labels_metadata'
arousal, valence, label_times_eeg, label_times_bvp, label_times_gsr = utils_general_prueba.get_labels(full_module_name)
baseline_eeg_features, baseline_bvp_features, baseline_gsr_phasic_features, baseline_gsr_tonic_features = utils_general_prueba.load_baseline_data(subject, constants)

print(baseline_eeg_features.head())
print(baseline_bvp_features.head())
print(baseline_gsr_phasic_features.head())
print(baseline_gsr_tonic_features.head())

#% load film data and split into scenes
start = 10
end = 10

signal = 'eeg'
fs = 250
EEG_scenes = utils_general_prueba.load_film2scenes(subject, signal, label_times_eeg, start, end, fs)    

signal = 'bvp'
fs = 64
BVP_scenes = utils_general_prueba.load_film2scenes(subject, signal, label_times_bvp, start, end, fs)   

signal = 'gsr'
fs = 4
GSR_scenes = utils_general_prueba.load_film2scenes(subject, signal, label_times_gsr, start, end, fs)  

print(len(EEG_scenes),len(BVP_scenes),len(GSR_scenes))

for i in range(len(valence)):
    print('i: ', i+1, ' GSR seconds: ', GSR_scenes[i].shape[1]/4)
    print('i: ', i+1, ' BVP seconds: ', BVP_scenes[i].shape[1]/64)
    print('i: ', i+1, ' EEG seconds: ', EEG_scenes[i].shape[1]/250)

    
all_arousal_labels, all_arousal_features, old_indx_arousal, all_valence_labels, all_valence_features, old_indx_valence = utils_general_prueba.compute_features(constants, baseline_eeg_features, EEG_scenes, GSR_scenes, BVP_scenes, valence, arousal)
np.save('./prueba/allFeatures.npy',np.asarray(all_valence_features))

#import numpy as np
#import pandas as pd
#all_valence_features = pd.DataFrame(np.load('./prueba/allFeatures.npy'))



#% CLASSIFICATION BY EMOTION FOR A SPECIFIC SUBJECT
from FEATURES import feature_smoothing
from sklearn.preprocessing import QuantileTransformer

qt_arousal = QuantileTransformer(output_distribution='normal')
all_arousal_features_normalized = pd.DataFrame(qt_arousal.fit_transform(all_arousal_features) , columns=list(all_arousal_features.keys()))
qt_valence = QuantileTransformer(output_distribution='normal')
all_valence_features_normalized = pd.DataFrame(qt_valence.fit_transform(all_valence_features) , columns=list(all_valence_features.keys()))

# -- smoothing the feature space -- #
for i in range(all_arousal_labels.shape[1]):
    all_arousal_features_normalized.iloc[:,i] = feature_smoothing.smoothing(all_arousal_features_normalized.iloc[:,i])
    all_valence_features_normalized.iloc[:,i] = feature_smoothing.smoothing(all_valence_features_normalized.iloc[:,i])
 
np.sum(np.isnan(all_arousal_features_normalized))
all_arousal_features_normalized = all_arousal_features_normalized.drop(['sdsd'], axis=1)

print(all_arousal_features_normalized.shape)
print(all_valence_features_normalized.shape)


#%########## DATASET SMOOTHING #######################
from CLASSIFIERS.models_trainer import model_trainer 
#from FILMS_ANALYSIS_ESCENAS.utils_general import get_aleatory_k_trials_out

model_trainer_valence = model_trainer('valence')
model_trainer_arousal = model_trainer('arousal')

emotions_valence = [4,5,6]
label_names_valence = [str(i) for i in emotions_valence]

emotions_arousal = [2,5,8]
label_names_arousal = [str(i) for i in emotions_arousal]

scores = {'test':[],'valence':[], 'arousal':[]}

from sklearn.model_selection import train_test_split
X_train_valence, X_test_valence, y_train_valence, y_test_valence = train_test_split(all_valence_features_normalized, all_valence_labels, test_size=0.33, random_state=42)
#size=1
#test, X_train_arousal,X_test_arousal,y_train_arousal,y_test_arousal,X_train_valence,X_test_valence,y_train_valence,y_test_valence = get_aleatory_k_trials_out(all_arousal_features_normalized,all_valence_features_normalized,all_arousal_labels,all_valence_labels,emotions_arousal,emotions_valence,old_indx_arousal,old_indx_valence,size=size)

#scores['test'].append(test)
# -- model training -- #  
classifiers_valence, _, predictions_valence, scores_list_valence = model_trainer_valence.classify(X_train_valence, y_train_valence, X_test_valence, y_test_valence)
#print('test: ', test, ' valence label: ', np.unique(y_test_valence), ' arousal label: ', np.unique(y_test_arousal))

scores['valence'].append(np.max(scores_list_valence))
print('valence: ', np.max(scores_list_valence))
#        
#classifiers_arousal, _, predictions_arousal, scores_list_arousal = model_trainer_arousal.classify(X_train_arousal,y_train_arousal,X_test_arousal,y_test_arousal)
 
#
#scores['arousal'].append(np.max(scores_list_arousal))
#print('arousal: ', np.max(scores_list_arousal))
    
    
print('valence mean: ', np.mean(scores['valence']), ' std: ', np.std(scores['valence']))  
#print('arousal mean: ', np.mean(scores['arousal']), ' std: ', np.std(scores['arousal']))  

import pickle
pickle.dump(classifiers_valence[np.argmax(scores_list_valence)], open('./prueba/final_model.sav', 'wb'))
from sklearn.externals import joblib 
joblib.dump(qt_valence, './prueba/quantile_scaler.pkl') 
