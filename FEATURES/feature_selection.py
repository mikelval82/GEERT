#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: %(Mikel Val Calvo)s
@email: %(mikel1982mail@gmail.com)
@institution: %(Dpto. de Inteligencia Artificial, Universidad Nacional de Educación a Distancia (UNED))
"""

from sklearn.feature_selection import RFE
from sklearn.svm import SVC

def rfe_selection(features, labels, numFeatures=2, kernel='linear'):
    model = SVC(kernel="linear")
    rfe = RFE(model, numFeatures)
    fit = rfe.fit(features, labels)
    return features[:,fit.support_],fit.support_

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def kbest_selection(features, labels, numFeatures=2):
    selector = SelectKBest(chi2, k=numFeatures)
    features = selector.fit_transform(features, labels)
    return features, selector.get_support()

#from FEATURES.mifs.mifs import MutualInformationFeatureSelector as mrmr
#def maxRel_minRed(features, labels, numFeatures, method):
#    feat_selector = mrmr(n_features=numFeatures, method=method)
#    # find all relevant features
#    feat_selector.fit(features, labels)
#    # check selected features
#    print('check selected features: ',feat_selector.support_)
#    # check ranking of features
#    print('check ranking of features: ', feat_selector.ranking_)
#    # call transform() on X to filter it down to selected features
#    features_filtered = feat_selector.transform(features)
#    
#    return features_filtered, feat_selector.support_