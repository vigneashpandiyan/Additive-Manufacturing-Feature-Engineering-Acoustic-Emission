# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 02:51:37 2023

@author: srpv
"""

import numpy as np
import pandas as pd
import os     
from sklearn.preprocessing import StandardScaler
from Classifiers.RF import *
from Classifiers.SVM import *
from Classifiers.NeuralNets import *
from Classifiers.kNN import *
from Classifiers.QDA import *
from Classifiers.NavieBayes import *
from Classifiers.Logistic_regression import *
from Classifiers.XGBoost import *


#%%
windowsize= 5000
Material = "Bronze"
path=r'C:\Users\srpv\Desktop\C4 Science\lpbf-acoustic-feature-engineering\Feature extraction'

#%%

featurefile = str(Material)+'_Featurespace'+'_'+ str(windowsize)+'.npy'
classfile = str(Material)+'_classspace'+'_'+ str(windowsize)+'.npy'
featurefile = os.path.join(path, featurefile)
classfile = os.path.join(path, classfile)
Featurespace = np.load(featurefile).astype(np.float64)
classspace= np.load(classfile).astype(np.float64)

#%%

#classspace=np.ravel(classspace)
Featurespace = pd.DataFrame(Featurespace)
num_cols = len(list(Featurespace))
rng = range(1, num_cols + 1)
Featurenames = ['Feature_' + str(i) for i in rng] 
Featurespace.columns = Featurenames
feature_cols=list(Featurespace.columns) 
Featurespace.info()
Featurespace.describe()
Featurespace.head()

#%%

df2 = pd.DataFrame(classspace) 
df2.columns = ['Categorical']
df2=df2['Categorical'].replace(1,0)
df2 = pd.DataFrame(df2)
df2=df2['Categorical'].replace(2,1)
df2 = pd.DataFrame(df2)
df2=df2['Categorical'].replace(3,2)
classspace = pd.DataFrame(df2) 

#%%
standard_scaler = StandardScaler()
Featurespace=standard_scaler.fit_transform(Featurespace)
#%%

Featurespace = pd.DataFrame(Featurespace)
num_cols = len(list(Featurespace))
rng = range(1, num_cols + 1)
Featurenames = ['Feature_' + str(i) for i in rng] 
Featurespace.columns = Featurenames
feature_cols=list(Featurespace.columns) 
Featurespace.info()
Featurespace.describe()
Featurespace.head()

#%% Training and Testing

from sklearn.model_selection import train_test_split# implementing train-test-split
X_train, X_test, y_train, y_test = train_test_split(Featurespace, classspace, test_size=0.25, random_state=66)


#%% Model Training and Testing   

RF(X_train, X_test, y_train, y_test,100,feature_cols)
SVM(X_train, X_test, y_train, y_test,Featurespace, classspace) 
LR(X_train, X_test, y_train, y_test)
NN(X_train, X_test, y_train, y_test)
XGBoost(X_train, X_test, y_train, y_test)
KNN(X_train, X_test, y_train, y_test,5, 'distance')
QDA(X_train, X_test, y_train, y_test)
NB(X_train, X_test, y_train, y_test)

