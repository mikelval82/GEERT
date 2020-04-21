#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: %(Mikel Val Calvo)s
@email: %(mikel1982mail@gmail.com)
@institution: %(Dpto. de Inteligencia Artificial, Universidad Nacional de Educación a Distancia (UNED))
"""

#%%
import matplotlib.pyplot as plt
from keras import optimizers
from keras.layers import Input, Dense
from keras.models import Model

def train(X_train, X_test):
    # this is the size of our encoded representations    
    input_img = Input(shape=(200,))
    encoded = Dense(64, activation='relu')(input_img)
    encoded = Dense(32, activation='relu')(encoded)
    encoded = Dense(8, activation='relu')(encoded)
    
    decoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(200, activation='sigmoid')(decoded)
    
    autoencoder = Model(input_img, decoded)
    
    sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    autoencoder.compile(loss='mean_squared_error', optimizer=sgd)
    
    autoencoder.fit(X_train, X_train,
                    epochs=100,
                    batch_size=256,
                    shuffle=False,
                    validation_data=(X_test, X_test))
    
    # this model maps an input to its encoded representation
    encoder = Model(input_img, encoded)
    
    return encoder

#%%########## DATASET SMOOTHING #######################
from CLASSIFIERS import models_trainer 

label_names = ['POSITIVE','NEGATIVE']
emotions = [6,4]

from FEATURES.features_train_test_split import train_test_split, get_aleatory_k_trials_out

from sklearn.preprocessing import MinMaxScaler
scaler2 = MinMaxScaler()
features_minmax = scaler2.fit_transform(features_smooth)

reordered_data = train_test_split(features_minmax, labels, emotions, trials_labels_map)  

OneTrialOut = []
for indx in range(10):
    X_train,y_train,X_test,y_test = get_aleatory_k_trials_out(reordered_data,emotions,size=2)
    
    encoder = train(X_train, X_test)
    
    z_train = encoder.predict(X_train)
    z_test = encoder.predict(X_test)
    
    classifiers, names, predictions, scores_list = models_trainer.classify(z_train, y_train, z_test, y_test)
    winner = np.asarray(scores_list).argmax()
    print(models_trainer.get_report(classifiers[winner], z_test, y_test, label_names))

    f1_score = models_trainer.get_f1_score(classifiers[winner], z_test, y_test)  
    OneTrialOut.append(f1_score)
    
print('f1 mean: ', np.mean(OneTrialOut), ' std: ', np.std(OneTrialOut))  