# -*- coding: utf-8 -*-
"""
@author: %(Mikel Val Calvo)s
@email: %(mikel1982mail@gmail.com)
@institution: %(Dpto. de Inteligencia Artificial, Universidad Nacional de Educación a Distancia (UNED))
@DOI: 10.5281/zenodo.3759306 
"""
#%%
from FEATURES.online_features_02 import compute_online_features
from FEATURES.feature_selection import rfe_selection
from FILTERS.EAWICA_new import eawica
from CLASSIFIERS import models_trainer 
from GENERAL import csv_fileIO as io
import numpy as np
import pickle
from sklearn.externals import joblib 

class pipeline():
    '''
        A class for online feature extraction and estimation.
    '''
    def __init__(self, app):
        self.constants = app.constants
        self.log = app.log
        self.numBasicFeatures = 5
        self.numBands = 5
        self.numFeatures = self.numBasicFeatures*self.numBands*self.constants.CHANNELS
        self.numTrials = 15
        self.numTestTrials = 9
        print('training pipeline is initialized')
        
    def run(self):
        # -- set paths --
        user = 'dani'
        path_training = './data/DANI/' + user +'.csv'
        path_labels = './data/DANI/labels_' + user + '.csv'
        path_best_model = './RESULTS/DANI/best_model.npy'
        path_best_model_furnitures = './RESULTS/DANI/best_model_furnitures.npy'
        path_final_model = './RESULTS/DANI/final_model.npy'
        path_final_model_furnitures = './RESULTS/DANI/final_model_furnitures.npy'
        path_allFeatures = './RESULTS/DANI/allFeatures.npy'
        path_allLabels = './RESULTS/DANI/allLabels.npy'
        path_quantile_scaler = './RESULTS/DANI/quantile_scaler.pkl'
        ############  LOAD TRAINING DATA #############################
        self.log.update_text('Data loading')
        dataframe = io.open_csvFile(path_training)
        labels_byTrial = io.open_csvFile(path_labels)
        self.log.update_text('Data have been loaded')
        dataframe = dataframe.iloc[8:,:]
        ############## COMPUTE FEATURES ###################################
        numSamples = int(dataframe.shape[0]/self.constants.CHANNELS)
        # -- init training data           
        features = []
        labels = []
        anterior = 0
        contador = 0
        self.log.update_text('Start computing features')
        for i in range(numSamples):
            # -- indexing training data
            ini = i*self.constants.CHANNELS
            end = ini+self.constants.CHANNELS
            
            
            if anterior == dataframe['trial'].iloc[ini] and contador > 2:
                sample = dataframe.iloc[ini:end,3:]
                sample = eawica(np.asarray(sample),self.constants)
                features.append(compute_online_features(sample,self.constants,self.numFeatures))
                labels.append( labels_byTrial['label'].iloc[ dataframe['trial'].iloc[i*self.constants.CHANNELS] ]     )
                contador+=1
            elif anterior == dataframe['trial'].iloc[ini] and contador <=2:
                contador+=1
            else:
                anterior = dataframe['trial'].iloc[ini]
                contador = 0
            
        features = np.asarray(features)
        labels = np.asarray(labels)
        
        self.log.update_text('Features have been computed')
        
        ####### DATASET PREPROCESSING #######################
        self.log.update_text('Features preprocessing')
        from sklearn.preprocessing import QuantileTransformer
        Quantile_scaler = QuantileTransformer(output_distribution='normal')
        features = Quantile_scaler.fit_transform(features)
        
        np.save(path_allFeatures, features)
        np.save(path_allLabels, labels)  
        joblib.dump(Quantile_scaler, path_quantile_scaler) #
        
        ########## DATASET SMOOTHING #######################
        from FEATURES.feature_smoothing import smoothing
        features = smoothing(features)

        ######## TRAIN AND TEST SETS SPLIT ##############
        self.log.update_text('Train test split')
        from FEATURES.features_train_test_split import ByTrials_train_test_split
        X_train, y_train, X_test, y_test = ByTrials_train_test_split(features, labels, self.numTrials, self.numTestTrials)
        
        ########## FEATURE SELECTION AND MODEL TRAINING ##############
        self.log.update_text('Model trainning')
        
        label_names = ['POS', 'NEU', 'NEG']
        best_model = {'model':[],'name':[],'predictions':[],'score':0,'selected_features':[],'report':[]}
        for i in range(1,self.numFeatures):
            # -- feature selection -- #
            select_features = i
            _, selected_features_indx = rfe_selection(X_train, y_train, select_features)  
            # -- model training -- #  
            classifiers, names, predictions, scores_list = models_trainer.classify(X_train[:,selected_features_indx,], y_train, X_test[:,selected_features_indx], y_test)
            winner = np.asarray(scores_list).argmax()
            if scores_list[winner] > best_model['score']:
                best_model['model'] = classifiers[winner]
                best_model['name'] = names[winner]
                best_model['predictions'] = predictions[winner]
                best_model['score'] = scores_list[winner]
                best_model['selected_features'] = selected_features_indx
                best_model['report'] = models_trainer.get_report(classifiers[winner], X_test[:,selected_features_indx], y_test, label_names)
                # -- logging report --
                self.log.update_text('Report: ' + best_model['report'])
                # save the model to disk     
                pickle.dump(best_model['model'], open(path_best_model, 'wb'))
                # -- save furnitures to disk --
                self.log.update_text('Saving furnitures')
                np.save(path_best_model_furnitures, best_model)
                
        model,name = models_trainer.train_models(features[:,best_model['selected_features']],labels, best_model['name'])   
        final_model = {'model':[],'name':[],'selected_features':[]}
        final_model['model'] = model
        final_model['name'] = name
        final_model['selected_features'] = best_model['selected_features']
        self.log.update_text('Saving sklearn model')
        pickle.dump(final_model['model'], open(path_final_model, 'wb'))
        np.save(path_final_model_furnitures, final_model)
        self.log.update_text('Training finished!')
               
            
        
