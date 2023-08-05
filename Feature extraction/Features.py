# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 17:18:58 2023

@author: srpv
"""
import numpy as np
from Feature_extraction import *
print(np.__version__)
import os

#%%

windowsize= 5000
sample_rate = 1000000
t0=0
dt=1/sample_rate
time = np.arange(0, windowsize) * dt + t0

band_size = 6 
peaks_to_count = 7  
count = 0

path= r'C:\Users\srpv\Desktop\C4 Science\lpbf-acoustic-feature-engineering\Data'
Materials="Inconel"
classes=['LoF','Nopores','Keyhole']

#%%

def Timeseries_feature(classes,sample_rate,band_size,peaks_to_count):
    
    featurelist=[]
    classlist=[]
    rawlist =[]
    class_label=0
    
    for class_name in classes:
        
        print(class_name)
        
        path_=str(Materials)+'_'+str(class_name)+'.npy'
        path_ = os.path.join(path, path_)
        data = np.load(path_).astype(np.float64)
        columns = np.atleast_2d(data).shape[1]
        
        #for row in loop:
        for k in range(columns):
            
            val= data[:,k]
            rows,d = divmod(val.size,windowsize)
            val = val[d:]
            window=round(val.size/windowsize)
            
            Feature_vectors,classes,rawdataset=feature_extraction(val,window,class_label,sample_rate,band_size,peaks_to_count)
            
            print(class_label)
            
            for item in Feature_vectors:
                
                featurelist.append(item)
                
            for item in rawdataset:
                
                rawlist.append(item)
                
            for item in classes:
                
                classlist.append(item)
            
        class_label=class_label+1   
        
        return featurelist,classlist,rawlist


featurelist,classlist,rawlist = Timeseries_feature(classes,sample_rate,band_size,peaks_to_count)

#%%

Featurespace=np.asarray(featurelist)
Featurespace=Featurespace.astype(np.float64)

rawspace=np.asarray(rawlist)
rawspace=rawspace.astype(np.float64)

classspace=np.asarray(classlist)

featurefile = str(Materials)+'_Featurespace'+'_'+ str(windowsize)+'.npy'
classfile = str(Materials)+'_classspace'+'_'+ str(windowsize)+'.npy'
rawfile = str(Materials)+'_rawspace'+'_'+ str(windowsize)+'.npy'

np.save(featurefile,Featurespace, allow_pickle=True)
np.save(classfile,classspace,allow_pickle=True)
np.save(rawfile,rawspace, allow_pickle=True)

#%%
