# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 20:55:13 2023

@author: srpv
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os
from tSNE import *
#%%

file = os.path.join(os.getcwd(), os.listdir(os.getcwd())[0])
total_path=os.path.dirname(file) 
print(total_path)

#%%
sns.set(font_scale = 1.5)
sns.set_style("whitegrid", {'axes.grid' : False})
sns.set_style("ticks", {"xtick.major.size":8,"ytick.major.size":8})
sample_rate=1000000
windowsize= 5000
t0=0
dt=1/sample_rate
time = np.arange(0, windowsize) * dt + t0
Material = "StainlessSteel"
path=r'C:\Users\srpv\Desktop\C4 Science\lpbf-acoustic-feature-engineering\Feature extraction'

#%%

featurefile = str(Material)+'_Featurespace'+'_'+ str(windowsize)+'.npy'
classfile = str(Material)+'_classspace'+'_'+ str(windowsize)+'.npy'
rawfile = str(Material)+'_rawspace'+'_'+ str(windowsize)+'.npy'
featurefile = os.path.join(path, featurefile)
classfile = os.path.join(path, classfile)
rawfile = os.path.join(path, rawfile)

Featurespace = np.load(featurefile).astype(np.float64)
classspace= np.load(classfile).astype(np.float64)

rawspace = np.load(rawfile).astype(np.float64)
rawspace = pd.DataFrame(rawspace)

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
df2=df2['Categorical'].replace(1,'LoF pores')
df2 = pd.DataFrame(df2)
df2=df2['Categorical'].replace(2,'Conduction mode')
df2 = pd.DataFrame(df2)
df2=df2['Categorical'].replace(3,'Keyhole pores')
classspace = pd.DataFrame(df2)

#%% Training and Testing
standard_scaler = StandardScaler()
Featurespace=standard_scaler.fit_transform(Featurespace)

from sklearn.model_selection import train_test_split# implementing train-test-split
X_train, X_test, y_train, y_test = train_test_split(Featurespace, classspace, test_size=0.05, random_state=66)

#%%
graph_name= str(Material)+'_'+'.png'
#graph_name= str(Material)+'_'+'.png'
ax,fig=TSNEplot(X_train,y_train,graph_name,str(Material),10)
graph_name= str(Material)+'.gif'

#%%
def rotate(angle):
      ax.view_init(azim=angle)
      
angle = 3
ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
ani.save(graph_name, writer=animation.PillowWriter(fps=20))

#%%

