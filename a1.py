# A1 COMP472

# TEAM MEMBERS:

import pandas as pd
import sklearn
import sklearn.model_selection as ms
import sklearn.tree as tree
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier


# read the .csv files in
penguins = pd.read_csv('penguins.csv')
abalone = pd.read_csv('abalone.csv')

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

# split datasets up for training and test:
# penguin 

penguin_features = penguins_onehot.drop('species', axis = 1)  

penguin_labels = penguins_onehot['species']

xtrain_penguin, xtest_penguin, ytrain_penguin, ytest_penguin = ms.train_test_split(penguin_features, penguin_labels)

# abalone
abalone_features = abalone_categorized.drop('Type', axis=1)  
abalone_labels = abalone_categorized['Type']

xtrain_abalone, xtest_abalone, ytrain_abalone, ytest_abalone = ms.train_test_split(abalone_features, abalone_labels)

# (4a) base-DT:
# penguin
# create the decision tree classifier
decision_tree_classifier_penguins = tree.DecisionTreeClassifier()
# fit the training data :)
decision_tree_classifier_penguins.fit(xtrain_penguin, ytrain_penguin)

# plot the tree
tree.plot_tree(decision_tree_classifier_penguins, feature_names = xtrain_penguin.columns)
plt.savefig('penguin_basicDT.png')
plt.show()

# abalone
# create a decision tree classifier
decision_tree_classifier_abalone = tree.DecisionTreeClassifier()
# fit the training data :)
decision_tree_classifier_abalone.fit(xtrain_abalone, ytrain_abalone)

# plot the tree (i tried some different max depths to get a legible looking tree, kind of cool just to see the full tree though)
tree.plot_tree(decision_tree_classifier_abalone, feature_names = xtrain_abalone.columns)
plt.savefig('abalone_basicDT.png')

# #BASE-MLP
mlp = MLPClassifier(hidden_layer_sizes=(100, 100), activation = 'logistic', solver = 'sgd')
mlp.fit(xtrain_abalone, ytrain_abalone)
prediction = mlp.predict(xtest_abalone)
