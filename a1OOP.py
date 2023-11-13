
# A1 COMP472

# TEAM MEMBERS:
# Chris Anglin 40216346
# Stefan Codrean 40227929
# Daniele Comitogianni

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import tree
from sklearn import model_selection as ms
from sklearn.model_selection import GridSearchCV as gridsearch
from sklearn.neural_network import MLPClassifier as mlp

#############################
# CLASS DEFINITIONS [for models and metrics] (made them OOP-ish so we can call it multiple times for part 6)

# a class for Base-DT
class BaseDT:

    def __init__(self):
        self.dtc = tree.DecisionTreeClassifier()

    def train(self, x_train, y_train):
        self.dtc.fit(x_train, y_train)

    def predict(self, x_test):
        return self.dtc.predict(x_test)
    
    def plot(self, x_train, filename):
        tree.plot_tree(self.dtc, feature_names=x_train.columns)
        plt.savefig(filename + '.png')

# a class for Top-DT
class TopDT:

    def __init__(self, p_grid):
        self.dtc = tree.DecisionTreeClassifier()
        self.parameter_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20], # just using 10, 20 to see the difference in values
            'min_samples_split':[3, 5, 17]
            }

    def train(self, x_train, y_train):
        self.grid_search = gridsearch(self.dtc, self.parameter_grid, cv=5, scoring='accuracy')
        self.grid_search.fit(x_train, y_train)
        self.best = self.grid_search.best_estimator_

    def predict(self, x_test):
        return self.best.predict(x_test)
    
    def get_best_parameters(self):
        return self.best.get_params()

    def plot(self, x_train, filename):
        tree.plot_tree(self.best, feature_names=x_train.columns)
        plt.savefig(filename + '.png')

# a class for Base-MLP
class BaseMLP:

    def __init__(self, activ='logistic'): # added option here to pass in sigmoid (or other)
        self.mlp = mlp(hidden_layer_sizes=(100, 100), activation=activ, solver='sgd')

    def train(self, x_train, y_train):
        self.mlp.fit(x_train, y_train)

    def predict(self, x_test):
        return self.mlp.predict(x_test)

# a class for Top-MLP
class TopMLP:

    def __init__(self, p_grid):
        self.mlp = mlp(max_iter = 2000)
        self.parameter_grid = {
                'activation': ['sigmoid', 'tanh', 'relu'],
                'hidden_layer_sizes': [(77, 37), (37, 77), (120, 80, 120)], # put 3.. was curious to try a pair of 'inverse' and something larger to see what works best
                'solver': ['adam', 'sgd']  
            }

    def train(self, x_train, y_train):
        self.grid_search = gridsearch(self.mlp, self.parameter_grid, cv=5, scoring='accuracy')
        self.grid_search.fit(x_train, y_train)
        self.best = self.grid_search.best_estimator_

    def predict(self, x_test):
        return self.best.predict(x_test)
    
    def get_best_parameters(self):
        return self.best.get_params()


# a class for metric output stuff
class MetricsOutput:

    def __init__(self):
        self.file = open('metrics[COMP472 A1].txt', 'a')
    
    def metrics(self, model_name, labels, predictions):
        self.name = model_name
        self.confusion_matrix = metrics.confusion_matrix(labels, predictions)
        self.precision = metrics.precision_score(labels, predictions, average=None)
        self.recall = metrics.recall_score(labels, predictions, average=None)
        self.f1 = metrics.recall_score(labels, predictions, average=None)
        self.accuracy = metrics.accuracy_score(labels, predictions)
        self.macrof1 = metrics.f1_score(labels, predictions, average='macro')
        self.weightedf1 = metrics.f1_score(labels, predictions, average='weighted')

    # this is the method needed for part 5
    def output(self):
        self.file.write('\n\n!!! ******* !!!\n(A) ' + self.name)
        self.file.write('\n(B) confusion matrix:\n' + str(self.confusion_matrix))
        self.file.write('\n(C) precision: ' + str(self.precision) + '\n    recall: ' + str(self.recall) + '\n    f1: ' + str(self.f1))
        self.file.write('\n(D) accuracy: ' + str(self.accuracy) + '\n    macro-average-f1: ' + str(self.macrof1) + '\n    weighted-average-f1: ' + str(self.weightedf1))

#####################################

# we can move the scripting stuff over here to process the data 

# we can call the above methods to get the output for part 5

# call them some more to get the output for part 6
