# -*- coding: utf-8 -*-
"""
@author: %(Mikel Val Calvo)s
@email: %(mikel1982mail@gmail.com)
@institution: %(Dpto. de Inteligencia Artificial, Universidad Nacional de Educación a Distancia (UNED))
@DOI: 10.5281/zenodo.3759306 
"""

from __future__ import division, print_function, absolute_import

import numpy as np

import pyedflib


from stacklineplot import stackplot


if __name__ == '__main__':
    f = pyedflib.data.test_generator()
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    n_min = f.getNSamples()[0]
    sigbufs = [np.zeros(f.getNSamples()[i]) for i in np.arange(n)]
    for i in np.arange(n):
        sigbufs[i] = f.readSignal(i)
        if n_min < len(sigbufs[i]):
            n_min = len(sigbufs[i])
    f._close()
    del f

    n_plot = np.min((n_min, 2000))
    sigbufs_plot = np.zeros((n, n_plot))
    for i in np.arange(n):
        sigbufs_plot[i,:] = sigbufs[i][:n_plot]

    stackplot(sigbufs_plot[:, :n_plot], ylabels=signal_labels)
