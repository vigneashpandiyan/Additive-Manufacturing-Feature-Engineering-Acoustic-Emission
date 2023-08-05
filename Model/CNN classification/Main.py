# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 20:55:13 2023

@author: srpv
"""



import torch
from torch.utils.data import Dataset, DataLoader,TensorDataset
from Utils import *
from Network import *
from Trainer import *
import os
import pandas as pd
#%%

windowsize= 5000
Material = "Bronze"
path=r'C:\Users\srpv\Desktop\C4 Science\lpbf-acoustic-feature-engineering\Feature extraction'

#%%

featurefile = str(Material)+'_rawspace'+'_'+ str(windowsize)+'.npy'
classfile = str(Material)+'_classspace'+'_'+ str(windowsize)+'.npy'
featurefile = os.path.join(path, featurefile)
classfile = os.path.join(path, classfile)
Featurespace = np.load(featurefile).astype(np.float64)
classspace= np.load(classfile).astype(np.float64)


df2 = pd.DataFrame(classspace)
df2.columns = ['Categorical']


df2=df2['Categorical'].replace(1,0)
df2 = pd.DataFrame(df2)
df2=df2['Categorical'].replace(2,1)
df2 = pd.DataFrame(df2)
df2=df2['Categorical'].replace(3,2)
df2 = pd.DataFrame(df2)

classspace = df2.to_numpy().astype(float)

#classspace = df2.to_numpy()

#%%


from sklearn.model_selection import train_test_split# implementing train-test-split
X_train, X_test, Y_train, Y_test = train_test_split(Featurespace, classspace, test_size=0.25, random_state=66)

   
X_train=covert_tensor(X_train)
X_test=covert_tensor(X_test)
Y_train=covert_tensor(Y_train)
Y_test=covert_tensor(Y_test)

print("Train input dim: ", X_train.shape)
print("Train target dim: ", Y_train.shape)
print("Test input dim: ", X_test.shape)
print("Test target dim: ", Y_test.shape)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
print(device)


#%% 

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  net= CNN(dropout_rate=0.5)
  # net = nn.DataParallel(net)


#%%
if __name__ == '__main__':
    
    testset,net,classes,iteration,Loss_value,Total_Epoch,Accuracy,Learning_rate,Training_loss_mean,Training_loss_std = test(net,device,X_train, Y_train,X_test, Y_test,Material,windowsize)
    
    iteration = np.array(iteration)
    Loss_value = np.array(Loss_value)
    Total_Epoch = np.array(Total_Epoch)
    Accuracy = np.array(Accuracy)
    
    Learning_rate = np.array(Learning_rate)
    Training_loss_mean = np.array(Training_loss_mean)
    Training_loss_std = np.array(Training_loss_std)
    
    plots(iteration,Loss_value,Total_Epoch,Accuracy,Learning_rate,Training_loss_mean,Training_loss_std,Material,windowsize)
    confusion_matrix = confusionmatrix(testset,device,net,classes)
    matrixplot(confusion_matrix,classes,Material,windowsize)
     
#%%
count_parameters(net)  
