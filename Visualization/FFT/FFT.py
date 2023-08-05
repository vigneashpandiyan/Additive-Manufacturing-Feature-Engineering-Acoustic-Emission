# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 20:55:13 2023

@author: srpv
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from scipy import signal
import matplotlib.patches as mpatches
import os

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
Featurespace = pd.DataFrame(Featurespace)

classspace= np.load(classfile).astype(np.float64)
classspace = pd.DataFrame(classspace)

rawspace = np.load(rawfile).astype(np.float64)
rawspace = pd.DataFrame(rawspace)


#%%

def filter(signal_window):
    
    lowpass = 100000  # Cut-off frequency of the filter
    lowpass_freq = lowpass / (sample_rate / 2) # Normalize the frequency
    b, a = signal.butter(5, lowpass_freq, 'low')
    signal_window = signal.filtfilt(b, a, signal_window)
    
    return signal_window

#%%

def Frequencyplot(rawspace,classspace):
    
    
    classspace.columns = ['Categorical']
    data = pd.concat([rawspace, classspace], axis=1)
    print("Respective windows per category",data.Categorical.value_counts())
    # minval = min(data.Categorical.value_counts())  
    minval = 1
    print("windows of the class: ",minval)
    data = pd.concat([data[data.Categorical == cat].head(minval) for cat in data.Categorical.unique() ])  
    print("Balanced dataset: ",data.Categorical.value_counts())
    rawspace=data.iloc[:,:-1]
    rawspace = rawspace.to_numpy() 
    classspace=data.iloc[:,-1]
    classspace = classspace.to_numpy() 
    
    for i in range(len(classspace)):
        
        print(i)
        
        data=rawspace[i]
        data= filter(data)
        category = int(classspace[i])
        print(category)
        
        
        # Define window length (4 seconds)
        win = 0.1 * sample_rate
        freqs, psd = signal.welch(data, sample_rate, nperseg=win)
        
        # Plot the power spectrum
        sns.set(font_scale=1.5, style='white')
        plt.figure(figsize=(7, 4),dpi=200)
        plt.plot(freqs, psd, color='k', lw=0.3)
        sec1, sec2 = 0, 20000
        sec3, sec4 = 20000, 40000
        sec5, sec6 = 40000, 60000
        sec7, sec8 = 60000, 80000
        sec9, sec10 = 80000, 100000
        
    
    # Find intersecting values in frequency vector
        idx_delta1 = np.logical_and(freqs >= sec1, freqs <= sec2)
        idx_delta2 = np.logical_and(freqs >= sec3, freqs <= sec4)
        idx_delta3 = np.logical_and(freqs >= sec5, freqs <= sec6)
        idx_delta4 = np.logical_and(freqs >= sec7, freqs <= sec8)
        idx_delta5 = np.logical_and(freqs >= sec9, freqs <= sec10)
        
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power spectral density')
        
        ynumber= max(psd)+ (0.1*max(psd))
        
        plt.ylim([0, ynumber])
        plt.fill_between(freqs, psd, where=idx_delta1, color='#657e39')
        plt.fill_between(freqs, psd, where=idx_delta2, color='#aa332d')
        plt.fill_between(freqs, psd, where=idx_delta3, color='#f0a334')
        plt.fill_between(freqs, psd, where=idx_delta4, color='#0080ff')
        plt.fill_between(freqs, psd, where=idx_delta5, color='#b05aac')
        #plottitle='Tribo'+'_'+'Welchs periodogram'+'_'+State
        plottitle=str(Material)+'_'+str(category)
        plt.title(plottitle)
        plt.xlim([0, freqs.max()])
        plt.xlim([0, 100000])
        
        skyblue = mpatches.Patch(color='#657e39', label='0-20 kHz')
        plt.legend(handles=[skyblue])
        red = mpatches.Patch(color='#aa332d', label='20-40 kHz')
        plt.legend(handles=[red])
        yellow = mpatches.Patch(color='#f0a334', label='40-60 kHz')
        plt.legend(handles=[yellow])
        green = mpatches.Patch(color='#0080ff', label='60-80 kHz')
        plt.legend(handles=[green])
        cyan = mpatches.Patch(color='#b05aac', label='80-100 kHz')
        plt.legend(handles=[skyblue,red,yellow,green,cyan])
        
        
        plt.ticklabel_format(axis='x', style='sci',scilimits=(0,0))
        plt.ticklabel_format(axis='y', style='sci',scilimits=(0,0))
        graph_1= str(Material)+'_'+str(category)+'.png'
        plt.savefig(os.path.join(total_path, graph_1),  bbox_inches='tight',dpi=800)
        plt.show()
    
    
#%%

Frequencyplot(rawspace,classspace)