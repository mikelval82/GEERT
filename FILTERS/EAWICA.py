# -*- coding: utf-8 -*-
"""
@author: %(Mikel Val Calvo)s
@email: %(mikel1982mail@gmail.com)
@institution: %(Dpto. de Inteligencia Artificial, Universidad Nacional de Educación a Distancia (UNED))
@DOI: 10.5281/zenodo.3759306 
"""
#%%
# entropy
from ENTROPY import entropy
# filters
from FILTERS.filter_bank_manager import filter_bank_class
# infomax based ICA method
from ica import ica1
# external common libraries
from scipy.stats import kurtosis
from sklearn import preprocessing
from scipy.stats import zscore
import numpy as np

#%% ENHANCED WAVELET-ICA 
def eawica(sample, constants, wavelet='db4', low_k=45, up_k=95, low_r=45, up_r=95, alpha=6):
    n_channels = sample.shape[0]
    n_epochs = constants.SECONDS
    n_samples = constants.WINDOW
    fb = filter_bank_class(constants)
    
    # COMPUTE WAVELET DECOMPOSED wcs_delta
    wcs, wcs_gamma, wcs_beta, wcs_alpha = [],[],[],[]
    for i in range(n_channels):
        GAMMA, BETA, ALPHA, THETA, DELTA = fb.eawica_wavelet_band_pass(sample[i,:], wavelet)
        pos = i*2
        wcs.append([THETA,pos])
        wcs.append([DELTA,pos+1])
        wcs_alpha.append(ALPHA)
        wcs_beta.append(BETA)
        wcs_gamma.append(GAMMA)
  
    # CHECKING FIRST CONDITION OVER ALL wcs_delta
    kurt_list = []
    renyi_list = []
    for i in range(len(wcs)):
        #  -- kurtosis --
        k = kurtosis(wcs[i][0])
        kurt_list.append(k)
        # -- renyi entropy --
        pdf = np.histogram(wcs[i][0], bins=10)[0]/wcs[i][0].shape[0]
        r = entropy.renyientropy(pdf,alpha=alpha,logbase=2,measure='R')
        renyi_list.append(r)
     
    # -- scaling --   
    kurt_list_scaled = zscore(kurt_list)
    renyi_list_scaled = zscore(renyi_list)
      
    # -- threshold --
    low_kurt_threshold, up_kurt_threshold = np.percentile(kurt_list_scaled, low_k), np.percentile(kurt_list_scaled, up_k)
    low_renyi_threshold, up_renyi_threshold = np.percentile(renyi_list_scaled, low_r), np.percentile(kurt_list_scaled, up_r)

    cond_11 = np.logical_or(kurt_list_scaled > up_kurt_threshold, kurt_list_scaled < low_kurt_threshold)
    cond_12 = np.logical_or(renyi_list_scaled > up_renyi_threshold, renyi_list_scaled < low_renyi_threshold)   
    cond_1 = cond_11 + cond_12 
    
    # SELECT wcs_delta MARKED AS CONTAINING ARTIFACTUAL INFORMATION
    signals2check = np.zeros((np.sum(cond_1), n_samples+1))
    
    indices = np.where(cond_1 == True)[0]
    for indx in range(len(indices)):
        if cond_1[indices[indx]]:
            signals2check[indx,:-1] = wcs[indices[indx]][0]
            signals2check[indx,-1] = wcs[indices[indx]][1]
            
    # ICA INFOMAX DECOMPOSITION OF MARKED signals TO OBTAIN ICs
    n_components = signals2check.shape[0]
    A,S,W = ica1(signals2check[:,:-1], n_components)

    # CHECK SECOND CONDITION OVER EACH EPOCH ON WICs
    control_k = np.zeros((S.shape[0], (n_epochs)))
    control_r = np.zeros((S.shape[0], (n_epochs)))
    
    for indx1 in range(S.shape[0]):        
        for indx2 in range((n_epochs)):
            ini = int((indx2*S.shape[1]/n_epochs))
            end = int(ini + S.shape[1]/n_epochs)
            if end+1==S.shape[1]:
                end+=1
            epoch = S[indx1,ini:end] 
            control_k[indx1,indx2] = kurtosis(epoch)
            pdf = np.histogram(epoch, bins=10)[0]/epoch.shape[0]
            r = entropy.renyientropy(pdf,alpha=alpha,logbase=2,measure='R')
            control_r[indx1,indx2] = r        
    
    table = np.zeros((S.shape[0], n_epochs))
    for indx1 in range(n_epochs):
        control_k[:,indx1] = preprocessing.scale(control_k[:,indx1])
        control_r[:,indx1] = preprocessing.scale(control_r[:,indx1])     
        
    table = np.logical_or(control_k > up_kurt_threshold, control_k < low_kurt_threshold) + np.logical_or(control_r > up_renyi_threshold, control_r < low_renyi_threshold)
    
    # ZEROING THOSE EPOCHS IN WICs MARKED AS ARTIFACTUAL EPOCHS
    for indx1 in range(S.shape[0]):        
        for indx2 in range(n_epochs):
            if table[indx1,indx2]:
                ini = indx2*int(n_samples/n_epochs)
                end = ini + int(n_samples/n_epochs)
                # epochs zeroing
                S[indx1,ini:end] = 0             
    
    # wcs_delta RECONSTRUCTION FROM WICs
    reconstructed = A.dot(S) 
    for i in range(reconstructed.shape[0]):   
        wcs[int(signals2check[i,-1])][0] = reconstructed[i,:]

    
    data_cleaned = np.zeros(sample.shape)   
    for i in range(n_channels):
        pos = i*2
        data_cleaned[i,:] = wcs[pos][0]+wcs[pos+1][0]+wcs_alpha[i]+wcs_beta[i]+wcs_gamma[i]
            
            
    return data_cleaned

