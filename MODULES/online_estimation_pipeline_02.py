# -*- coding: utf-8 -*-
"""
@author: %(Mikel Val Calvo)s
@email: %(mikel1982mail@gmail.com)
@institution: %(Dpto. de Inteligencia Artificial, Universidad Nacional de Educación a Distancia (UNED))
@DOI: 10.5281/zenodo.3759306 
"""
#%%
from FEATURES.online_features_02 import compute_online_features
from FEATURES.feature_smoothing import smoothing
from FILTERS.EAWICA import eawica
from multiprocessing import Process, Array
from sklearn.externals import joblib
import numpy as np
import pickle
from sklearn.externals import joblib 
from threading import Thread

class pipeline(Thread):
    '''
        A class for online feature extraction and estimation.
    '''
    def __init__(self, app):  
        Thread.__init__(self)
        self.app = app
        self.trigger_server = app.trigger_server
        self.eeg_dmg = app.eeg_dmg
        self.constants = app.constants
        self.log = app.log
        self.slots = app.slots
        # callbacks
        self.trigger_server.new_COM1.connect(self.save_predictions)
        self.isstored = True
        self.trial = 0
        # -- features settings --
        numBasicFeatures = 5
        numBands = 5
        self.numFeatures = numBasicFeatures*numBands*self.constants.CHANNELS
        # -- path settings --
        path_model = './prueba/final_model.sav'
#        path_best_model_furnitures = './prueba/'+ subject +'/final_model_furnitures.npy'
        self.path_allFeatures = './prueba/allFeatures.npy'
        path_quantile_scaler = './prueba/quantile_scaler.pkl'
        self.path_predictions = './prueba/predictions'
        # -- load machine-learning model --
        self.model = pickle.load(open(path_model, 'rb'))
        # -- load furnitures --
        self.quantile_scaler = joblib.load(path_quantile_scaler)
#        best_model_furnitures = np.load(path_best_model_furnitures)
#        self.selected_features_indx = best_model_furnitures.item().get('selected_features')
        self.allPredictions = []
        # -- set shared array
        self.predictions = Array('f', range(3))
        
        
        
    def run(self):
        print('################# start online estimation pipeline ################3')
        self.app.slots.append(self.start_process)     
        self.app.slots.append(self.append_to_store_predictions)
        print('##############3 slots appened ##################')
            
    def start_process(self):
        process = Process(target=self.online_estimation, args=(self.predictions,))
        process.start()
        
    def online_estimation(self, predictions):
        # -- get sample --
        sample = self.eeg_dmg.get_short_sample(self.constants.METHOD)   
     
        # -- artifact removal --
        sample = eawica(sample,self.constants)
        # -- compute features --
#        feature = compute_online_features(sample,self.constants, self.numFeatures)
        feature = compute_online_features(sample,self.constants)
        # -- load train data --
        self.training_data = np.load(self.path_allFeatures)
        
        # -- feature smoothing --
        aux = np.vstack((self.training_data,feature[:self.training_data.shape[1]]))
        # -- feature scaling --
        aux  = self.quantile_scaler.transform( aux )
        
        
        for i in range(aux.shape[1]):
            aux[:,i] = smoothing(aux[:,i])
        feature = aux[-1,:]

        # -- estimate emotion --
#        probabilities = self.model.predict_proba(feature[self.selected_features_indx].reshape(1,-1))
        probabilities = self.model.predict_proba(feature.reshape(1,-1))
        predictions[:] = probabilities.squeeze().tolist()
        
        print('after processing predictions: ', predictions[:])

        
    def append_to_store_predictions(self):
        print(self.predictions[:])
        self.allPredictions.append(self.predictions[:])
        
    
    def save_predictions(self):
        print('save predictions: ', self.isstored)
        if not self.isstored:
            np.save(self.path_predictions + '_trial_' + str(self.trial), self.allPredictions)
            self.trial += 1
        self.isstored = not self.isstored
        
        
