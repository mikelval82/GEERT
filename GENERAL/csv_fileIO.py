# -*- coding: utf-8 -*-
"""
@author: %(Mikel Val Calvo)s
@email: %(mikel1982mail@gmail.com)
@institution: %(Dpto. de Inteligencia Artificial, Universidad Nacional de Educación a Distancia (UNED))
@DOI: 10.5281/zenodo.3759306 
"""
#%%
import csv
import pandas as pd
import numpy as np

def create_csvFile(path):
    with open(path, 'wb') as csvfile:
        csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    
def append_to_csvFile(data, path, columns, control):
    over_control = True
    for sample_dict in data:
        print(sample_dict['data'].shape)
        data_df = pd.DataFrame(sample_dict['data']) 
        data_df.columns = [str(i) for i in range(data_df.shape[1])]
        dataframe = pd.DataFrame()   
        dataframe['channel_ID'] = columns            
        dataframe['trial'] = sample_dict['trial']
        dataframe['window'] = sample_dict['window']     
        dataframe = dataframe.join(data_df)
        
        with open(path, 'a') as f:
            dataframe.to_csv(f, index=False, header=np.logical_and(over_control,control))
        over_control = False
        
def open_csvFile(path):
    return pd.read_csv(path) 
    
