# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 19:18:18 2023

@author: srpv
"""

import pandas as pd
import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
import ntpath
import re
from scipy import signal
# import pywt
import seaborn as sns
from sklearn import preprocessing
import os

def filter(signal_window,sample_rate):
    
    lowpass = 100000  # Cut-off frequency of the filter
    lowpass_freq = lowpass / (sample_rate / 2) # Normalize the frequency
    b, a = signal.butter(5, lowpass_freq, 'low')
    signal_window = signal.filtfilt(b, a, signal_window)
    
    #plt.plot(t, lowpassfilter, label='Low-Pass')
    return signal_window
    
def STFT(Material,sample_rate,rawspace,classspace,time,total_path):
    
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
        data= filter(data,sample_rate)
        category = int(classspace[i])
        print(category)
    
        
        plt.rcParams.update(plt.rcParamsDefault)
        plt.rcParams['agg.path.chunksize'] = len(data)
        
        # NFFT = 5000
        # dt = time[1] - time[0]
        Fs = 1.0 / sample_rate
        
        fig, ax = plt.subplots(figsize=(12, 7))
        # Pxx, freqs, bins, im = ax.specgram(signal_window, NFFT=NFFT, Fs=Fs, noverlap=4500)
        Pxx, freqs, bins, im = ax.specgram(data, Fs=sample_rate,cmap="rainbow")
        fig.patch.set_visible(True)
        
        ax.axis('on')
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=20)
        ax.ticklabel_format(axis='x', style='sci',scilimits=(0,0))
        ax.ticklabel_format(axis='y', style='sci',scilimits=(0,0))
        ax.yaxis.offsetText.set_fontsize(20)
        ax.xaxis.offsetText.set_fontsize(20)
        ax.set_ylim(0, sample_rate/2)
        cb=plt.colorbar(im)
        cb.set_label(label='Intensity',fontsize=20)
        cb.ax.tick_params(labelsize=20) 
        
        plottitle=('Spectrogram')
        plt.suptitle(plottitle, fontsize=20)
        
        plt.xlabel('Time(sec)',fontsize=20)
        plt.ylabel('Frequency(Hz)',fontsize=20)
        
        graphname=str(Material)+'_'+str(category)+'.png'
        plt.savefig(os.path.join(total_path, graphname), bbox_inches='tight',dpi=800)
       
        plt.show()
        plt.clf()
    
    