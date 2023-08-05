# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 19:18:18 2023

@author: srpv
"""
import numpy as np
import matplotlib.pyplot as plt
import pywt
import pandas as pd
from matplotlib import cm
import os
from scipy import signal


def filter(signal_window,sample_rate):
    
    lowpass = 100000  # Cut-off frequency of the filter
    lowpass_freq = lowpass / (sample_rate / 2) # Normalize the frequency
    b, a = signal.butter(5, lowpass_freq, 'low')
    signal_window = signal.filtfilt(b, a, signal_window)
    
    #plt.plot(t, lowpassfilter, label='Low-Pass')
    return signal_window

def surface_waveletplot(data,time):
    scales = np.arange(1,100)
    waveletname = 'morl' 
    cmap = plt.cm.jet
    dt = time[1] - time[0]
    [coefficients, frequencies] = pywt.cwt(data, scales, waveletname, dt)
    power = (abs(coefficients)) ** 2
    period = 1. / frequencies
    lenthA= len(frequencies)
    frequencies= frequencies[frequencies < 100000]
    #frequencies= frequencies[frequencies > 10000]
    
    lenthB = len(frequencies)
    trimlenth = lenthA - lenthB
    power=np.delete(power, np.s_[0:trimlenth],axis=0)
    #freq=pywt.scale2frequency(waveletname, scales, precision=8)
    timeplot, frequencies = np.meshgrid(time,frequencies)   
    return frequencies,power,timeplot


def ThreeDwaveletplot(Material,sample_rate,rawspace,classspace,time,total_path):
    
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
        
        frequencies,power,timeplot = surface_waveletplot(data,time)
        X=timeplot
        Y=frequencies
        Z=power
        
        fig = plt.figure(figsize=(14,10), dpi=200)
        ax = fig.add_subplot(projection='3d')
        
        #plot_trisurf,plot_surface
        surf = ax.plot_surface(X,Y, Z, cmap=cm.coolwarm,
                                linewidth=0,rstride=1, cstride=1, antialiased=True,vmin=np.min(Z),vmax=np.max(Z)) #rstride=25, cstride=1,
        vmax=np.max(Z)
        ax.set_xlim(0, np.max(X))
        ax.set_ylim(0, 100000)
        ax.set_zlim(0, vmax)
        ax.ticklabel_format(axis='x', style='sci',scilimits=(0,0))
        ax.ticklabel_format(axis='y', style='sci',scilimits=(0,0))
        ax.ticklabel_format(axis='z', style='sci',scilimits=(0,0))
        plottitle=str(Material)+'_'+str(category)
        plt.title(plottitle, loc='center')
        
        
        ax.zaxis.set_rotate_label(False) 
        plt.xlabel('Time', labelpad=5)
        plt.ylabel('Frequency', labelpad=5)
        ax.set_zlabel('Power', labelpad=5,rotation=90)
        ax.grid(False)
        ax.set_facecolor('white') 
        
        ax.w_xaxis.pane.fill = False
        ax.w_yaxis.pane.fill = False
        ax.w_zaxis.pane.fill = False
        
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        
        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        
        ax.view_init(azim = 10,elev = 30)
        #cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        fig.colorbar(surf, shrink=0.4, aspect=10,ax=ax)
        fig.patch.set_visible(True)
        #plt.tight_layout(pad=4, w_pad=10, h_pad=1.0)
        plt.locator_params(nbins=4)
        graph_1= str(Material)+'_'+str(category)+'.png'
        plt.savefig(os.path.join(total_path, graph_1), bbox_inches='tight',dpi=800)
        plt.show()
        plt.clf()