
# A1 COMP472

# TEAM MEMBERS:

import pandas as pd
import sklearn 
import matplotlib

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

# split datasets up for training and test:
# penguin 
penguin_features = penguins_onehot.drop('species')  
penguin_labels = penguins_onehot['species']

xtrain_penguin, xtest_penguin, ytrain_penguin, ytest_penguin = sklearn.model_selection.train_test_split(penguin_features, penguin_labels)

# abalone
abalone_features = abalone_categorized.drop('sex')  
abalone_labels = abalone_categorized['sex']

xtrain_abalone, xtest_abalone, ytrain_abalone, ytest_abalone = sklearn.model_selection.train_test_split(abalone_features, abalone_labels)









