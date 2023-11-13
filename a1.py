# A1 COMP472

# TEAM MEMBERS:
# Chris Anglin 40216346
# Stefan Codrean 40227929
# Daniele Comitogianni

import numpy as np
import pandas as pd
from sklearn import metrics
import sklearn.model_selection as ms
import sklearn.tree as tree
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.neural_network import MLPClassifier


# 1.
# read the .csv files in
penguins = pd.read_csv('COMP472_A1/penguins.csv')
abalone = pd.read_csv('COMP472_A1/abalone.csv')

# do the one-hot encoding for penguins
penguins_onehot = pd.get_dummies(penguins, columns=['island','sex'])

# do the manual categorization for penguins
penguins_manual = penguins
penguins_manual['island'] = pd.Categorical(penguins_manual['island'])
penguins_manual['sex'] = pd.Categorical(penguins_manual['sex'])

penguins_manual['island'] = penguins_manual['island'].cat.codes
penguins_manual['sex'] = penguins_manual['sex'].cat.codes

# do the catergorization for abalone (looks like Type needs to be made numerical)
abalone_categorized = abalone

abalone_categorized['Type'] = pd.Categorical(abalone_categorized['Type']) # make string categorical

abalone_categorized['Type']  = abalone_categorized['Type'].cat.codes # make values numerical

# i am just going to save the .csv's and take a look that the changes are looking okay
penguins_onehot.to_csv('penguins_onehot.csv')
penguins_manual.to_csv('penguins_manual.csv')
abalone_categorized.to_csv('abelone_categorized.csv')

# 2.
# plotting the species percentage distribution for penguins
penguins_species_counts = penguins_onehot['species'].value_counts(normalize=True)*100
plt.figure()
penguins_species_counts.plot(kind='bar')
plt.title("Penguin Species")
plt.xlabel('Species')
plt.ylabel('Percentages')
plt.savefig('penguin-classes.png')

# plotting the sex percentage distribution for abalone
abalone_sex_counts = abalone_categorized['Type'].value_counts(normalize=True)*100
plt.figure()
abalone_sex_counts.plot(kind='bar')
plt.title("Abalone Sex")
plt.xlabel('Sex')
plt.xticks(ticks=[0,1,2],labels=['Female','Infant','Male'])
plt.ylabel('Percentages')
plt.savefig('abalone-classes.png')

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

# !!!!!
# create files to write to
p_file = open('penguin-performance.txt', 'a')
a_file = open('abalone-performance.txt', 'a')

def metrics_output(model, x_test, y_test, file, name):
    y_predictions = model.predict(x_test)

    confusion_matrix = metrics.confusion_matrix(y_test, y_predictions)
    precision = metrics.precision_score(y_test, y_predictions, average=None)
    recall = metrics.recall_score(y_test, y_predictions, average=None)
    f1 = metrics.recall_score(y_test, y_predictions, average=None)
    accuracy = metrics.accuracy_score(y_test, y_predictions)
    macrof1 = metrics.f1_score(y_test, y_predictions, average='macro')
    weightedf1 = metrics.f1_score(y_test, y_predictions, average='weighted')

    file.write('\n\n\n!!! ******* !!!\n(A) ' + name)
    file.write('\n(B) confusion matrix: \n   ' + str(confusion_matrix))
    file.write('\n(C) precision: ' + str(precision) + ' recall: ' + str(recall) + ' f1: ' + str(f1))
    file.write('\n(D) accuracy: ' + str(accuracy) + ' macro-average-f1: ' + str(macrof1) + ' weighted-average-f1: ' + str(weightedf1))

############################ 4a) Base-DT:#############################################
# penguin
# create the decision tree classifier
decision_tree_classifier_penguins = tree.DecisionTreeClassifier()
# fit the training data :)
decision_tree_classifier_penguins.fit(xtrain_penguin, ytrain_penguin)

metrics_output(decision_tree_classifier_penguins, xtest_penguin, ytest_penguin, p_file, 'BASE-DT (penguins)')

# plot the tree
tree.plot_tree(decision_tree_classifier_penguins, feature_names = xtrain_penguin.columns)
plt.savefig('penguin_basicDT.png')

# abalone
# create a decision tree classifier
decision_tree_classifier_abalone = tree.DecisionTreeClassifier()
# fit the training data :)
decision_tree_classifier_abalone.fit(xtrain_abalone, ytrain_abalone)

metrics_output(decision_tree_classifier_abalone, xtest_abalone, ytest_abalone, a_file, 'BASE-DT (abalone)')

# plot the tree (i tried some different max depths to get a legible looking tree, kind of cool just to see the full tree though)
tree.plot_tree(decision_tree_classifier_abalone, feature_names = xtrain_abalone.columns)
plt.savefig('abalone_basicDT.png')
##############################(4b) Top-DT #############################################

#setting up parameter Grid
parameter_grid={
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20], #just using 10, 20 to see the difference in values
    'min_samples_split':[5, 10, 20]
}

#initializes the decision tree classifier
dt = DecisionTreeClassifier()

grid_search_penguins = GridSearchCV(dt, parameter_grid, cv=5, scoring = 'accuracy')

grid_search_penguins.fit(xtrain_penguin, ytrain_penguin)

best_tree_penguins = grid_search_penguins.best_estimator_

optimized_top_dt_penguins = dt(best_tree_penguins)

op_base_penguins = optimized_top_dt_penguins.fit(xtrain_penguin, ytrain_penguin)


metrics_output(op_base_penguins, xtest_penguin, ytest_penguin, p_file, 'TOP-DT (penguin)')

plt.figure()
tree.plot_tree(best_tree_penguins, filled=True)

plt.savefig('best_decision_tree_penguin.png')
plt.close()

dt = DecisionTreeClassifier()

grid_search_abalone = GridSearchCV(dt, parameter_grid, cv=5, scoring = 'accuracy')

grid_search_abalone.fit(xtrain_abalone, ytrain_abalone)

best_tree_abalone = grid_search_abalone.best_estimator_

optimized_top_dt_abalone = dt(best_tree_abalone)

op_base_abalone = optimized_top_dt_abalone.fit(xtrain_abalone, ytrain_abalone)


metrics_output(op_base_abalone, xtest_abalone, ytest_abalone, a_file, 'TOP-DT (abalone)')

plt.figure()
tree.plot_tree(best_tree_abalone, filled=True)

plt.savefig('best_decision_tree_penguin.png')
plt.close()


##############################4c) BASE-MLP #############################################

mlp_penguin = MLPClassifier(hidden_layer_sizes=(100, 100), activation = 'logistic', solver = 'sgd')
mlp_penguin.fit(xtrain_penguin, ytrain_penguin)
prediction = mlp_penguin.predict(xtest_penguin)
score = np.round(metrics.accuracy_score(ytest_penguin, prediction), 2)    
print("Mean accuracy score: ", score)

metrics_output(mlp_penguin, xtest_penguin, ytest_penguin, p_file, 'BASE-MLP (penguins)')

mlp_abalone = MLPClassifier(hidden_layer_sizes=(100, 100), activation = 'logistic', solver = 'sgd') #add parameter verbose = True to see the training process, random_state = anyNumber to debug
mlp_abalone.fit(xtrain_abalone, ytrain_abalone)
predictions = mlp_abalone.predict(xtest_abalone)
score = np.round(metrics.accuracy_score(ytest_abalone, predictions), 2)
print("Mean accuracy score: ", score)

metrics_output(mlp_abalone, xtest_abalone, ytest_abalone, a_file, 'BASE-MLP (abalone)')

###############################4d) TOP-MLP#############################################
parameters = {
    'hidden_layer_sizes': [(30, 50), (10, 10, 10), (40,40)],
    'activation': ['relu', 'tanh', 'logistic'], 
    'solver': ['adam', 'sgd']  
}

# for penguin data set
mlp = MLPClassifier(max_iter = 2000)
grid_search_penguin = GridSearchCV(mlp, parameters, cv = 5)
grid_search_penguin.fit(xtrain_penguin, ytrain_penguin)
print('Best parameters found:\n', grid_search_penguin.best_params_)
best_model_penguins = grid_search_penguin.best_estimator_
print('Best model:\n', best_model_penguins)

optimized_top_mlp_penguin = MLPClassifier(best_model_penguins, max_iter = 2000)
op_top_penguins = optimized_top_mlp_penguin.fit(xtrain_penguin, ytrain_penguin)
# predictions_penguin_top = optimized_top_mlp_penguin.predict(xtest_penguin)

metrics_output(op_top_penguins, xtest_penguin, ytest_penguin, p_file, 'TOP-MLP (penguins)')

# abalone data set
mlp = MLPClassifier(max_iter = 2000)

# Apply GridSearchCV For Abalone
grid_search_abalone = GridSearchCV(mlp, parameters, cv = 5)
grid_search_abalone.fit(xtrain_abalone, ytrain_abalone)

# Best parameters found by GridSearchCV
print('Best parameters found:\n', grid_search_abalone.best_params_)

# Best model found by GridSearchCV
best_model_abalone = grid_search_abalone.best_estimator_
print('Best model:\n', best_model_abalone)
optimized_top_mlp_abalone = MLPClassifier(best_model_abalone, max_iter = 2000)
optimized_top_mlp_abalone.fit(xtrain_penguin, ytrain_penguin)
op_top_abalone = optimized_top_mlp_abalone.fit(xtrain_abalone, ytrain_abalone)


metrics_output(op_top_abalone, xtest_abalone, ytest_abalone, a_file, 'TOP-MLP (abalone)')

p_file.close()
a_file.close()
