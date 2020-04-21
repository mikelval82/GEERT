#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:59:51 2019

@author: anaiak
"""
import numpy as np 
import pandas as pd
import random

def reorder(all_features,all_labels_modified,old_indx,num_group,step):
    step = step
    num_group = num_group
    
    new_data = []
    new_labels = []
    new_old_indx = []
    
    scenes = np.unique(old_indx)

    for which_scene in scenes:
        indx_scene = np.where(old_indx == which_scene)[0]
        
        all_features_to_reorder = all_features[indx_scene,:]
        all_labels_to_reorder = all_labels_modified[indx_scene]
        old_indx_to_reorder = old_indx[indx_scene]
        
        num_samples = int((len(indx_scene)-num_group)/step)+1
        if num_samples > 1:
            for i in range(num_samples):
                ini = step*i
                end = ini + num_group
                
                matrix_sample = all_features_to_reorder[ini:end,:]
                new_sample = matrix_sample.reshape((matrix_sample.shape[1]*num_group,))
                new_data.append(new_sample)
                new_labels.append(all_labels_to_reorder[ini])
                new_old_indx.append(old_indx_to_reorder[ini])
        else:
            matrix_sample = np.zeros((num_group,all_features_to_reorder.shape[1]))
            for i in range(num_group):
                if all_features_to_reorder.shape[0] == 1:
                    matrix_sample[i,:] = all_features_to_reorder
                else:
                    pos = i%all_features_to_reorder.shape[0]
                    matrix_sample[i,:] = all_features_to_reorder[pos,:]
            new_sample = matrix_sample.reshape((matrix_sample.shape[1]*num_group,))
            new_data.append(new_sample)
            new_labels.append(all_labels_to_reorder[0])
            new_old_indx.append(old_indx_to_reorder[0])
            
            
            
    return np.asarray(new_data), np.asarray(new_labels,dtype='int'), np.asarray(new_old_indx,dtype='int')


def get_in_order_train_test(all_features_temporal,all_labels_new,emotions,old_indx_new,which_scene=0):
    
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    
    list_test = []
    
    for emo in emotions:
        indx = np.where(all_labels_new == emo)[0]
        data_subset = all_features_temporal[indx,:]
        labels_subset = all_labels_new[indx]
        old_indx_subset = old_indx_new[indx]
        
        list_old = list(np.unique(old_indx_subset))
        selected = which_scene 
        
        
        list_test.append(list_old[selected])
        print('list_test: ', list_test)
        
        for selection in list_test:
            indx = np.where(old_indx_subset == selection)[0]
            X_test.append( data_subset[indx,:] )
            y_test.append( labels_subset[indx] )
        
        list_train = []
        for selection in list_old:       
            if not selection in list_test:
                list_train.append(selection)
        
        print('list_train: ', list_train)
        
        for selection in list_train:
            indx = np.where(old_indx_subset == selection)[0]
            X_train.append( data_subset[indx,:] )
            y_train.append( labels_subset[indx] )
            
    X_train_a = X_train[0]
    y_train_a = y_train[0]
    if len(X_train) > 1:
        for i in range(1,len(X_train)):
            X_train_a = np.vstack((X_train_a,X_train[i]))
            y_train_a = np.hstack((y_train_a,y_train[i]))
            
    X_test_a = X_test[0]
    y_test_a = y_test[0]
    if len(X_test) > 1:
        for i in range(1,len(X_test)):
            X_test_a = np.vstack((X_test_a,X_test[i]))
            y_test_a = np.hstack((y_test_a,y_test[i]))
        
    return list_test,X_train_a,y_train_a,X_test_a,y_test_a



def get_aleatory_k_trials_out(all_features,all_labels,emotions,old_indx,size=1):
    
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    
    list_test = []
    
    for emo in emotions:
        indx = np.where(all_labels == emo)[0]
        data_subset = all_features[indx,:]
        labels_subset = all_labels[indx]
        old_indx_subset = old_indx[indx]
        
        list_old = list(np.unique(old_indx_subset))
        selected = random.sample(range(len(list_old)), size) 
        
        
        for selection in selected:
            list_test.append(list_old[selection])
            
        
        for selection in list_test:
            indx = np.where(old_indx_subset == selection)[0]
            X_test.append( data_subset[indx,:] )
            y_test.append( labels_subset[indx] )
        
        list_train = []
        for selection in range(0,len(list_old)):
            if not selection in selected:
                list_train.append(list_old[selection])
        
        
        for selection in list_train:
            indx = np.where(old_indx_subset == selection)[0]
            X_train.append( data_subset[indx,:] )
            y_train.append( labels_subset[indx] )
            
    X_train_a = X_train[0]
    y_train_a = y_train[0]
    if len(X_train) > 1:
        for i in range(1,len(X_train)):
            X_train_a = np.vstack((X_train_a,X_train[i]))
            y_train_a = np.hstack((y_train_a,y_train[i]))
            
    X_test_a = X_test[0]
    y_test_a = y_test[0]
    if len(X_test) > 1:
        for i in range(1,len(X_test)):
            X_test_a = np.vstack((X_test_a,X_test[i]))
            y_test_a = np.hstack((y_test_a,y_test[i]))
        
    return list_test,X_train_a,y_train_a,X_test_a,y_test_a










