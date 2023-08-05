# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 19:18:18 2023

@author: srpv
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
# import pywt
import pywt
import os

def filter(signal_window,sample_rate):
    
    lowpass = 100000  # Cut-off frequency of the filter
    lowpass_freq = lowpass / (sample_rate / 2) # Normalize the frequency
    b, a = signal.butter(5, lowpass_freq, 'low')
    signal_window = signal.filtfilt(b, a, signal_window)
    
    #plt.plot(t, lowpassfilter, label='Low-Pass')
    return signal_window
    
def Wavelet2D(Material,sample_rate,rawspace,classspace,time,total_path):
    
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
    
    waveletname= 'morl'
    dt = time[1] - time[0]
    scales = np.arange(1, 100)
    
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
        [coefficients, frequencies] = pywt.cwt(data, scales, waveletname, dt)
        
        
        power = (abs(coefficients))    
        lenthA= len(frequencies)
        # frequencies= frequencies[frequencies < lowpass]
        lenthB = len(frequencies)
        trimlenth = lenthA - lenthB
        power=np.delete(power, np.s_[0:trimlenth],axis=0)
        # power=np.log2(power)
        
        
        print(np.min(power))
        print(np.max(power))
        
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_visible(True)
        im = plt.contourf(time, frequencies, power,cmap=plt.cm.rainbow)
        

        
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
        
        plottitle=str(Material)+'_'+str(category)
        plt.suptitle(plottitle, fontsize=20)
        
        plt.xlabel('Time(sec)',fontsize=20)
        plt.ylabel('Frequency(Hz)',fontsize=20)
        
     
        graphname=str(Material)+'_'+str(category)+'_2D_Wavelet.png'
        plt.savefig(os.path.join(total_path, graphname), bbox_inches='tight',dpi=800)
       
        plt.show()
        plt.clf()
    

 