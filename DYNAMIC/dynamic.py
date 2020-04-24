# -*- coding: utf-8 -*-
"""
@author: %(Mikel Val Calvo)s
@email: %(mikel1982mail@gmail.com)
@institution: %(Dpto. de Inteligencia Artificial, Universidad Nacional de Educación a Distancia (UNED))
@DOI: 10.5281/zenodo.3759306 
"""
  
import sys 
import os
import importlib
        
def load_module(fileName, app):
    ########### separo path y nombre del modulo ##############
    aux = fileName.split("/")
    path = ''
    for i in range(1, len(aux)-1):
        path += aux[i] + '/'
    module_name = aux[-1][:-3]
    
    #---------------------------------------------------------
    sys.path.append(os.path.realpath('./MODULES/'))
    module = importlib.import_module(module_name)
    custom_object = module.pipeline(app)
    
    try:
#        custom_object.run()
        custom_object.start()
    except:
        app.log.update_text('* -- Loaded module must have a callable -> run() <- function -- *')
        
    #################################################################
    
      
        
