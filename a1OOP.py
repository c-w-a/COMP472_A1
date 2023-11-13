
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

4.
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
    
    def plot(self, x_train, depth):
        tree.plot_tree(self.dtc, feature_names=x_train.columns, max_depth=depth)
        plt.show()

# a class for Top-DT
class TopDT:

    def __init__(self):
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

    def plot(self, x_train):
        tree.plot_tree(self.best, feature_names=x_train.columns)
        plt.show()

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

    def __init__(self):
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
        self.p_file = open('penguin-performance.txt', 'w')
        self.a_file = open('abalone-performance.txt', 'w')
    
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
    def output(self, file):
        file.write('\n\n ******* \n(A) ' + self.name)
        file.write('\n(B) confusion matrix:\n' + str(self.confusion_matrix))
        file.write('\n(C) precision: ' + str(self.precision) + '\n    recall: ' + str(self.recall) + '\n    f1: ' + str(self.f1))
        file.write('\n(D) accuracy: ' + str(self.accuracy) + '\n    macro-average-f1: ' + str(self.macrof1) + '\n    weighted-average-f1: ' + str(self.weightedf1))

    def close_files(self):
        self.p_file.close()
        self.a_file.close()


#####################################

# 1.
# read the .csv files in
penguins = pd.read_csv('COMP472_A1/penguins.csv')
abalone = pd.read_csv('COMP472_A1/abalone.csv')

# do the one-hot encoding for penguins
penguins_onehot = pd.get_dummies(penguins, columns=['island','sex'])

# do the manual categorization for penguins
penguins_categorized = penguins
penguins_categorized['island'] = pd.Categorical(penguins_categorized['island'])
penguins_categorized['sex'] = pd.Categorical(penguins_categorized['sex'])

penguins_categorized['island'] = penguins_categorized['island'].cat.codes
penguins_categorized['sex'] = penguins_categorized['sex'].cat.codes

# do the catergorization for abalone (looks like Type needs to be made numerical)
abalone_categorized = abalone

abalone_categorized['Type'] = pd.Categorical(abalone_categorized['Type']) # make string categorical

abalone_categorized['Type']  = abalone_categorized['Type'].cat.codes # make values numerical

# i am just going to check that the changes are looking okay
print('checking the encodings: ')
print(penguins_onehot.head())
print(penguins_categorized.head())
print(abalone_categorized.head())

# 2.
# plotting the species percentage distribution for penguins
penguins_species_counts = penguins_onehot['species'].value_counts(normalize=True)*100
plt.figure()
penguins_species_counts.plot(kind='bar')
plt.title("Penguin Species")
plt.xlabel('Species')
plt.ylabel('Percentages')
plt.title('Penguin Classes')
plt.savefig('penguin-classes.png')
plt.close()

# plotting the sex percentage distribution for abalone
abalone_sex_counts = abalone_categorized['Type'].value_counts(normalize=True)*100
plt.figure()
abalone_sex_counts.plot(kind='bar')
plt.title("Abalone Sex")
plt.xlabel('Sex')
plt.xticks(ticks=[0,1,2],labels=['Female','Infant','Male'])
plt.ylabel('Percentages')
plt.title('Abalone Classes')
plt.savefig('abalone-classes.png')
plt.close()

# 3.
# split datasets up for training and test:
# penguin 
penguin_features = penguins_onehot.drop('species', axis = 1)  
penguin_labels = penguins_onehot['species']
xtrain_penguin, xtest_penguin, ytrain_penguin, ytest_penguin = ms.train_test_split(penguin_features, penguin_labels)

# abalone
abalone_features = abalone_categorized.drop('Type', axis=1)  
abalone_labels = abalone_categorized['Type']
xtrain_abalone, xtest_abalone, ytrain_abalone, ytest_abalone = ms.train_test_split(abalone_features, abalone_labels)

# 4. (is above at the top, the model definitions)

# 5. output metrics

# create models 
basedt = BaseDT()
topdt = TopDT()
basemlp1 = BaseMLP(activ='logistic')
basemlp2 = BaseMLP(activ='sigmoid')
topmlp = TopMLP()

# create metric manager thing
brains = MetricsOutput()

# Base DT for penguin
basedt.train(xtrain_penguin, ytrain_penguin)
predictions = basedt.predict(xtest_penguin)
brains.metrics('Base-DT', ytest_penguin, predictions)
brains.output(brains.p_file)
basedt.plot(xtrain_penguin, 'default')

# Base DT for abalone
basedt.train(xtrain_abalone, ytrain_abalone)
predictions = basedt.predict(xtest_abalone)
brains.metrics('Base-DT', ytest_abalone, predictions)
brains.output(brains.a_file)
basedt.plot(xtrain_abalone, 7)

# Top DT for penguins
topdt.train(xtrain_penguin, ytrain_penguin)
predictions = topdt.predict(xtest_penguin)
brains.metrics('Top-DT    | best parameters: ' + topdt.get_best_parameters(), ytest_penguin, predictions)
brains.output(brains.p_file)
topdt.plot(xtrain_penguin, 'default')

# Top DT for abalone
topdt.train(xtrain_abalone, ytrain_abalone)
predictions = topdt.predict(xtest_abalone)
brains.metrics('Top-DT    | best parameters: ' + topdt.get_best_parameters(), ytest_abalone, predictions)
brains.output(brains.a_file)
topdt.plot(xtrain_abalone, 7)

# Base MLP for penguins LOGISTIC
basemlp1.train(xtrain_penguin, ytrain_penguin)
predictions = basemlp1.predict(xtest_penguin)
brains.metrics('Base-MLP [logistic]', ytest_penguin, predictions)
brains.output(brains.p_file)

# Base MLP for abalone LOGISTIC
basemlp1.train(xtrain_abalone, ytrain_abalone)
predictions = basemlp1.predict(xtest_abalone)
brains.metrics('Base-MLP [logistic]', ytest_abalone, predictions)
brains.output(brains.a_file)

# Base MLP for penguin SIGMOID
basemlp2.train(xtrain_penguin, ytrain_penguin)
predictions = basemlp2.predict(xtest_penguin)
brains.metrics('Base-MLP [sigmoid]', ytest_penguin, predictions)
brains.output(brains.p_file)

# Base MLP for abalone SIGMOID
basemlp2.train(xtrain_abalone, ytrain_abalone)
predictions = basemlp2.predict(xtest_abalone)
brains.metrics('Base-MLP [sigmoid]', ytest_abalone, predictions)
brains.output(brains.a_file)

# Top MLP for penguins
topmlp.train(xtrain_penguin, ytrain_penguin)
predictions = topmlp.predict(xtest_penguin)
brains.metrics('Top-MLP   | best parameters: '+ topmlp.get_best_parameters(), ytest_penguin, predictions)
brains.output(brains.p_file)

# Top MLP for abalone
topmlp.train(xtrain_abalone, ytrain_abalone)
predictions = topmlp.predict(xtest_abalone)
brains.metrics('Top-MLP   | best parameters: ' + topmlp.get_best_parameters(), ytest_abalone, predictions)
brains.output(brains.a_file)

brains.close_files()

# call them some more to get the output for part 6
