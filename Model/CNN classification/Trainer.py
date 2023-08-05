# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 03:56:24 2023

@author: srpv
"""

from torchsummary import summary
from torch import nn, optim
import torch
from torch.optim.lr_scheduler import StepLR
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
#%%
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        print('Learning rate =')
        print(param_group['lr'])
        return param_group['lr']
    
def covert_tensor(data):
    data_tensor=[]
    global dataTorch
    first = True
    for p in data:
        
        if first:
            p = torch.from_numpy(p)
            dataTorch = p.view(1, 1, -1)
            first = False
            print(dataTorch.shape)
        else:
            p = torch.from_numpy(p)
            dataTorch = torch.cat((dataTorch, p.view(1, 1, -1)), 0)
            dataTorch.shape
    
    data_tensor.append(dataTorch)
    data_tensor = torch.cat(data_tensor, 1)
    return data_tensor
    
def test(net,device,X_train, Y_train,X_test, Y_test,Material,windowsize):    
    
    
    train = TensorDataset(X_train, Y_train)
    test = TensorDataset(X_test, Y_test)
    
    trainset = torch.utils.data.DataLoader(train, batch_size=100, num_workers=0,
                                                    shuffle=True)
    testset = torch.utils.data.DataLoader(test, batch_size=100, num_workers=0,
                                                     shuffle=True)
    
    #net= CNN()
    net.to(device)
    summary(net, (1 ,5000))
    
    costFunc = torch.nn.CrossEntropyLoss()
    optimizer =  torch.optim.SGD(net.parameters(),lr=0.01,momentum=0.9)
    
    scheduler = StepLR(optimizer, step_size = 25, gamma= 0.5 )
    
    Loss_value =[]
    Train_loss =[]
    Iteration_count=0
    iteration=[]
    Epoch_count=0
    Total_Epoch =[]
    Accuracy=[]
    Learning_rate=[]
    
    Training_loss_mean = []
    Training_loss_std = []
    
    for epoch in range(50):
        epoch_smoothing=[]
        learingrate_value = get_lr(optimizer)
        Learning_rate.append(learingrate_value)
        closs = 0
        scheduler.step()
        for i,batch in enumerate(trainset,0):
            data,output = batch
            data,output = data.to(device,dtype=torch.float),output.to(device,dtype=torch.long)

            prediction = net(data)
            
            loss = costFunc(prediction,output.squeeze())
            
            closs = loss.item()
            
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # epoch_smoothing.append(closs.cpu().detach().numpy())
            epoch_smoothing.append(closs)
            
            if i%10 == 0:
                print('[%d  %d] loss: %.4f'% (epoch+1,i+1,closs))
                
                Iteration_count = Iteration_count + 20
                Train_loss.append(closs)
                iteration.append(Iteration_count)
                closs = 0
        correctHits=0
        total=0
        
        loss_train = closs / len(trainset)
        Loss_value.append(loss_train)
        
        Training_loss_mean.append(np.mean(epoch_smoothing))
        Training_loss_std.append(np.std(epoch_smoothing))
        
        
        for batches in testset:
            data,output = batches
            data,output =data.to(device,dtype=torch.float),output.to(device,dtype=torch.long)
            
            output=output.squeeze()
            #print(output)
            prediction = net(data)
            _,prediction = torch.max(prediction.data,1)
            #print(prediction)
            total += output.size(0)
            correctHits += (prediction==output).sum().item()
        
        
        Epoch_count = epoch+1
        Total_Epoch.append (Epoch_count)
        Epoch_accuracy = (correctHits/total)*100
        Accuracy.append(Epoch_accuracy)
        print('Accuracy on epoch ',epoch+1,'= ',str((correctHits/total)*100))
    
    correctHits=0
    total=0
    for batches in testset:
        data,output = batches
        data,output =data.to(device,dtype=torch.float),output.to(device,dtype=torch.long)
        output=output.squeeze()
        prediction = net(data)
        _,prediction = torch.max(prediction.data,1)  #returns max as well as its index
        total += output.size(0)
        correctHits += (prediction==output).sum().item()
    print('Accuracy = '+str((correctHits/total)*100))
    
    print('Finished Training')
    
    PATH = './model_'+ str(Material)+'_'+str(windowsize)+'.pth'
    torch.save(net.state_dict(), PATH)
    torch.save(net, PATH)
    #Trained_model = torch.load(PATH)
    classes = ('0', '1', '2')
    return testset,net,classes,iteration,Loss_value,Total_Epoch,Accuracy,Learning_rate,Training_loss_mean,Training_loss_std
