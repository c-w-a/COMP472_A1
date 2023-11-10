
# A1 COMP472

# TEAM MEMBERS:

import pandas as pd
import sklearn 
import matplotlib

# read the .csv files in
penguins = pd.read_csv('penguins.csv')
abalone = pd.read_csv('abalone.csv')

penguins_oneHot = pd.get_dummies(penguins, columns=['island','sex'])

penguins_manual=penguins
penguins_manual['island']=pd.Categorical(penguins_manual['island'])
penguins_manual['sex']=pd.Categorical(penguins_manual['sex'])

penguins_manual['island']=penguins_manual['island'].cat.codes
penguins_manual['sex']=penguins_manual['sex'].cat.codes







