# -*- coding: utf-8 -*-
"""
@author: %(Mikel Val Calvo)s
@email: %(mikel1982mail@gmail.com)
@institution: %(Dpto. de Inteligencia Artificial, Universidad Nacional de Educación a Distancia (UNED))
@DOI: 10.5281/zenodo.3759306 
"""

from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import f1_score

class model_trainer:
    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
                 "Decision Tree", "Random Forest", "AdaBoost",
                 "Naive Bayes", "QDA"]
        
    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.5, probability=True),
        SVC(gamma=2, C=0.5, probability=True),
        DecisionTreeClassifier(),#max_depth=5
        RandomForestClassifier(),#max_depth=5, n_estimators=10, max_features=5
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]
        
    def __init__(self,instance):
        print(instance)

    def classify(self, X_train, y_train, X_test, y_test):
        scores_list = []
        predictions = []
        # iterate over classifiers
        for name, clf in zip(self.names, self.classifiers):
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            scores_list.append(score)
            predictions.append(clf.predict(X_test))
            
        return self.classifiers, self.names, predictions, scores_list
    
    def train_models(self, X_train, y_train, selected_model):
        loc = list(self.names).index(selected_model)
        model = self.classifiers[loc]
        model.fit(X_train, y_train)
        return model,selected_model
    
    def get_report(self, model, X_test, y_test, label_names):
        predictions = model.predict(X_test)   
        return classification_report(y_test,predictions,target_names=label_names)
    
    def get_f1_score(self, model, X_test, y_test):
        predictions = model.predict(X_test)  
        return f1_score(y_test, predictions, average='macro')  

