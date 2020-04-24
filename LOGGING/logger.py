# -*- coding: utf-8 -*-
"""
@author: %(Mikel Val Calvo)s
@email: %(mikel1982mail@gmail.com)
@institution: %(Dpto. de Inteligencia Artificial, Universidad Nacional de Educación a Distancia (UNED))
@DOI: 10.5281/zenodo.3759306 
"""
#%%
class logger():
    def __init__(self, gui):
        self.text = ''
        self.gui = gui
        
    def update_text(self, text):
        self.gui.bci_graph.logger.appendPlainText(text)
