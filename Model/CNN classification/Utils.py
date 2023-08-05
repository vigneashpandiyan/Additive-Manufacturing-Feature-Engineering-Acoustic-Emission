# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 03:41:01 2023

@author: srpv
"""
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import numpy as np
import torch

from Heatmap import heatmap , annotate_heatmap

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

#%%
def matrixplot(confusion_matrix,classes,Material,windowsize):


    matrix=confusion_matrix/confusion_matrix.sum(axis=1)*100
    
    #fig, ax = plt.figure(figsize=(20,8))
    fig, ax = plt.subplots(figsize=(15,8))
    ax.set_title("Confusion Matrix",fontsize=25)
    ax.set_xlabel('Actual', fontsize=25)
    ax.set_ylabel('Predicted', fontsize=25)
    #ax.text(fontsize=15)
    
    im, cbar = heatmap(matrix, classes, classes, ax=ax,
                   cmap="coolwarm", cbarlabel="Classification accuracy")
    texts = annotate_heatmap(im, size=16,fontweight="bold",valfmt="{x:.2f} %")
    
    #fig.tight_layout()
    
    ax.tick_params(axis = 'both', which = 'major', labelsize = 15)
    ax.tick_params(axis = 'both', which = 'minor', labelsize = 15)
    
    
    cbar.ax.tick_params(labelsize=20)
    plot_0=  'Confusion Matrix_'+ str(Material)+'_'+str(windowsize)+'.png'
    plt.savefig(plot_0, dpi=600,bbox_inches='tight')
    plt.show()

#%%
def plots(iteration,Loss_value,Total_Epoch,Accuracy,Learning_rate,Training_loss_mean,Training_loss_std,Material,windowsize):
    
    
    Accuracyfile = 'Accuracy'+'_'+ str(Material)+'.npy'
    Lossfile = 'Loss_value'+'_'+ str(Material)+'.npy'

    np.save(Accuracyfile,Accuracy,allow_pickle=True)
    np.save(Lossfile,Loss_value, allow_pickle=True)
    
    
    fig, ax = plt.subplots()
    plt.plot(Loss_value,'r',linewidth =2.0)
    # ax.fill_between(Loss_value, Training_loss_mean - Training_loss_std, Training_loss_mean + Training_loss_std, alpha=0.9)
    plt.title('Iteration vs Loss_Value')
    plt.xlabel('Iteration')
    plt.ylabel('Loss_Value')
    plot_1=  'Loss_value_'+ str(Material)+'_'+str(windowsize)+'.png'
    plt.savefig(plot_1, dpi=600,bbox_inches='tight')
    plt.show()
    plt.clf()
    
    plt.figure(2)
    plt.plot(Total_Epoch,Accuracy,'g',linewidth =2.0)
    plt.title('Total_Epoch vs Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plot_2=  'Accuracy_'+ str(Material)+'_'+str(windowsize)+'.png'
    plt.savefig(plot_2, dpi=600,bbox_inches='tight')
    plt.show()
    
    plt.figure(3)
    plt.plot(Total_Epoch,Learning_rate,'b',linewidth =2.0)
    plt.title('Total_Epoch vs Learning_Rate')
    plt.xlabel('Epochs')
    plt.ylabel('Learning_Rate')
    plot_3=  'Learning_rate_'+ str(Material)+'_'+str(windowsize)+'.png'
    plt.savefig(plot_3, dpi=600,bbox_inches='tight')
    plt.show()

#%%

def confusionmatrix(testset,device,net,classes):

    total_correct = 0
    total_images = 0
    confusion_matrix = np.zeros([3,3], int)
    with torch.no_grad():
        for data in testset:
            images, labels = data
            images = images.to(device,dtype=torch.float)
            labels = labels.to(device,dtype=torch.long)
            labels=labels.squeeze()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            for i, l in enumerate(labels):
                confusion_matrix[l.item(), predicted[i].item()] += 1 
    
    model_accuracy = total_correct / total_images * 100
    print('Model accuracy on {0} test images: {1:.2f}%'.format(total_images, model_accuracy))
    
    confusion_matrix=confusion_matrix.transpose()
    print('{0:10s} - {1}'.format('Category','Accuracy'))
    for i, r in enumerate(confusion_matrix):
        print('{0:10s} - {1:.1f}'.format(classes[i], r[i]/np.sum(r)*100))
    
    
    
    print('actual/pred'.ljust(9), end='')
    for i,c in enumerate(classes):
        print(c.ljust(3), end='')
    print()
    for i,r in enumerate(confusion_matrix):
        print(classes[i].ljust(9), end='')
        for idx, p in enumerate(r):
            print(str(p).ljust(3), end='')
        print()
    
        r = r/np.sum(r)
        print(''.ljust(9), end='')
        for idx, p in enumerate(r):
            print(str(p).ljust(3), end='')
        print()
        
    return confusion_matrix
