#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: %(Mikel Val Calvo)s
@email: %(mikel1982mail@gmail.com)
@institution: %(Dpto. de Inteligencia Artificial, Universidad Nacional de Educación a Distancia (UNED))
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras import backend as K

from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply
from keras.models import Model, Sequential

        
original_dim = 200
intermediate_dim = 100
latent_dim = 10
batch_size = 50
epochs = 150
epsilon_std = 0.01


def nll(y_true, y_pred):
    """ Negative log likelihood (Bernoulli). """

    # keras.losses.binary_crossentropy gives the mean
    # over the last axis. we require the sum
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)


class KLDivergenceLayer(Layer):

    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs

def train(X_train, X_test):
    decoder = Sequential([
        Dense(intermediate_dim, input_dim=latent_dim, activation='relu'),
        Dense(original_dim, activation='relu')
    ])
    
    x = Input(shape=(original_dim,))
    h = Dense(intermediate_dim, activation='relu')(x)
    
    z_mu = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)
    
    z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
    z_sigma = Lambda(lambda t: K.exp(.5*t))(z_log_var)
    
    eps = Input(tensor=K.random_normal(stddev=epsilon_std,
                                       shape=(K.shape(x)[0], latent_dim)))
    z_eps = Multiply()([z_sigma, eps])
    z = Add()([z_mu, z_eps])
    
    x_pred = decoder(z)
    
    vae = Model(inputs=[x, eps], outputs=x_pred)
    vae.compile(optimizer='rmsprop', loss=nll)
    
    # train the VAE on MNIST digits
    #(X_train, y_train), (X_test, y_test) = mnist.load_data()
    #X_train = X_train.reshape(-1, original_dim) / 255.
    #X_test = X_test.reshape(-1, original_dim) / 255.
    
    vae.fit(X_train,
            X_train,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, X_test))
    
    encoder = Model(x, z_mu)

    return encoder

def plot_latent(encoder, X_test, X_train):
    # display a 2D plot of the digit classes in the latent space
    z_test = encoder.predict(X_test, batch_size=batch_size)
    plt.figure(figsize=(6, 6))
    plt.scatter(z_test[:, 0], z_test[:, 1], c=y_test,
                alpha=.4, s=3**2, cmap='viridis')
    plt.colorbar()
    plt.show()
    
    z_train = encoder.predict(X_train, batch_size=batch_size)
    plt.figure(figsize=(6, 6))
    plt.scatter(z_train[:, 0], z_train[:, 1], c=y_train,
                alpha=.4, s=3**2, cmap='viridis')
    plt.colorbar()
    plt.show()
#%% LOAD DATA

SUBJECT = 'LIDIA'

features_smooth = np.load('./data/' + SUBJECT + '/peliculas/features_smooth.npy')
labels = np.load('./data/' + SUBJECT + '/peliculas/labels.npy')
trials_labels_map = np.load('./data/' + SUBJECT + '/peliculas/trials_labels_map.npy')
print(features_smooth.shape)
print(labels.shape)
print(trials_labels_map.shape)
#%%########## DATASET SMOOTHING #######################
from CLASSIFIERS import models_trainer 

label_names = ['POSITIVE','NEGATIVE']
emotions = [6,4]

from FEATURES.features_train_test_split import train_test_split, get_aleatory_k_trials_out

reordered_data = train_test_split(features_smooth, labels, emotions, trials_labels_map)  

OneTrialOut = []
for indx in range(20):
    X_train,y_train,X_test,y_test = get_aleatory_k_trials_out(reordered_data,emotions,size=1)
    
    encoder = train(X_train, X_test)
    
    z_train = encoder.predict(X_train, batch_size=batch_size)
    z_test = encoder.predict(X_test, batch_size=batch_size)
    
    classifiers, names, predictions, scores_list = models_trainer.classify(z_train, y_train, z_test, y_test)
    winner = np.asarray(scores_list).argmax()
    print(models_trainer.get_report(classifiers[winner], z_test, y_test, label_names))

    f1_score = models_trainer.get_f1_score(classifiers[winner], z_test, y_test)  
    OneTrialOut.append(f1_score)
    
print('f1 mean: ', np.mean(OneTrialOut), ' std: ', np.std(OneTrialOut))  