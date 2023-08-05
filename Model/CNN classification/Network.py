# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 03:48:34 2023

@author: srpv
"""
from torch import nn, optim
from torch.nn import functional as F



#%%


class CNN(nn.Module): 
    def __init__(self,dropout_rate):
        super(CNN, self).__init__()
        
        self.dropout_rate = dropout_rate
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=16) 
        self.bn1 = nn.BatchNorm1d(4) #4001
        
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=16) 
        self.bn2 = nn.BatchNorm1d(8) #4001
        
        self.conv3 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=16) 
        self.bn3 = nn.BatchNorm1d(16) #4001
        
        self.conv4 = nn.Conv1d(16, 32, kernel_size=16)
        self.bn4 = nn.BatchNorm1d(32) #1000
        
        self.conv5 = nn.Conv1d(32, 64, kernel_size=16)
        self.bn5 = nn.BatchNorm1d(64) #1000
                       
        self.fc1 = nn.Linear(832, 3)
       
        self.pool = nn.MaxPool1d(3)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x) 
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x) 
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x) 
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout(x) 
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        
        
        x = F.relu(self.bn5(self.conv5(x)))
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        
        # x = self.conv4(x)
        # print(x.shape)
        x = x.view(-1, 832)
        # print(x.shape)
        
        x = self.fc1(x)
        # print(x.shape)
        
        return x