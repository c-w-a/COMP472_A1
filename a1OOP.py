
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
from sklearn.neural_network import MLPClassifier

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
            'max_depth': [None, 10, 20], #just using 10, 20 to see the difference in values
            'min_samples_split':[5, 10, 20]
            }
        self.gridsearch = gridsearch(self.dtc, self.parameter_grid, cv=5, scoring='accuracy')
        self.optimized = False


    def optimize(self, x_train, y_train):
        self.gridsearch.fit(x_train, y_train)
        self.best = self.gridsearch.best_estimator_
        self.optimized = True

    def train(self, x_train, y_train):
        self.best.fit(x_train, y_train)

# a class for Base-MLP

# a class for Top-MLP

# a class for metric output stuff

#####################################

# we can move the scripting stuff over here to process the data 

# we can call the above methods to get the output for part 5

# call them some more to get the output for part 6
