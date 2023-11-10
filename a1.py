
# A1 COMP472

# TEAM MEMBERS:

import pandas as pd
import sklearn 
import matplotlib

# read the .csv files in
penguins = pd.read_csv('penguins.csv')
abalone = pd.read_csv('abalone.csv')

# do the one-hot encoding for penguins
penguins_oneHot = pd.get_dummies(penguins, columns=['island','sex'])

# do the manual categorization for penguins
penguins_manual = penguins
penguins_manual['island'] = pd.Categorical(penguins_manual['island'])
penguins_manual['sex'] = pd.Categorical(penguins_manual['sex'])

penguins_manual['island'] = penguins_manual['island'].cat.codes
penguins_manual['sex'] = penguins_manual['sex'].cat.codes

# do the catergorization for abalone (looks like Type needs to be made numerical)
abalone_categorized = abalone

abalone_categorized['Type'] = pd.Categorical(abalone_categorized['Type']) # make string categorical

abalone_categorized['island']  = abalone_categorized['island'].cat.codes # make values numerical









