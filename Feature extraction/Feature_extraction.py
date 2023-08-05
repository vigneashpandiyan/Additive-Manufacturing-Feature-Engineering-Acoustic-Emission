# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 17:38:41 2023

@author: srpv
"""
import numpy as np
import scipy.signal as signal
from scipy.stats import kurtosis, skew
from scipy.signal import welch, periodogram
from numpy.fft import fftshift, fft
from scipy.signal import find_peaks
import statistics 
from scipy import stats
from collections import Counter
from scipy.stats import entropy
from scipy.signal import hilbert, chirp
from scipy.stats import entropy
import pywt
from Utils import *

#%%
def feature_extraction(val,window,class_label,sample_rate,band_size,peaks_to_count):
    
    #data_new=data_new.transpose()
    
    Feature_vectors=[]
    signal_chop=np.split(val, window)   
    for i in range(window):
        
        
        signal_window=signal_chop[i]
        signal_window=filter(signal_window,sample_rate)
        
        #minimum
        Feature1 = signal_window.min()
        #maximum
        Feature2 = signal_window.max()
        #difference
        Feature3 = Feature2+Feature1
        #difference
        Feature4 = Feature2+abs(Feature1)
        #RMS
        Feature5 = np.sqrt(np.mean(signal_window**2))
        #print(Feature5)
        #STD
        Feature6 = statistics.stdev(signal_window)
        #Variance
        Feature7 = statistics.variance(signal_window)
        #Skewness
        Feature8 = skew(signal_window)
        #Kurtosis
        Feature9 = kurtosis(signal_window)
        #Mean
        Feature10 = statistics.mean(signal_window)
        #Harmonic Mean
        Feature11 = statistics.harmonic_mean(abs(signal_window))
        #Median
        Feature12 = statistics.median(signal_window)
        #Median_1
        Feature13 = Feature12-Feature11
        #Zerocrossing
        Feature14 = Zerocross(signal_window)
        #Mean Absolute Deviation
        Feature15 = stats.median_abs_deviation(signal_window)
        #Absolute Mean
        Feature16 = statistics.mean(abs(signal_window))
        #Absolute RMS
        Feature17 = np.sqrt(np.mean(abs(signal_window)**2))
        #Absolute Max
        Feature18 = max(abs(signal_window))
        #Absolute Min
        Feature19 = min(abs(signal_window))
        #Absolute Mean -  Mean      
        Feature20 = ((abs(signal_window)).mean())-(signal_window.mean())
        #difference+Median
        Feature21 = Feature3+Feature12
        #Crest factor - peak/ rms
        Feature22 = Feature2/Feature5
        #Auto correlation 4 peaks
        Feature23= get_autocorr_values(signal_window)
        
        
        #Frequency Domain Features
        
        win = 4 * sample_rate
        freqs, psd = periodogram(signal_window, sample_rate,window='hamming')
        band_max_size = 120000
        band = get_band(band_size,band_max_size)
        
        #PSD power in the signal periodgram
        Feature27 = sum(psd)
        
        #PSD absolute and relative power in each band 10 Features
        Feature28,Feature33 =spectrumpower(psd,band,freqs)
        Feature28 = np.asarray(Feature28)
        Feature33 = np.asarray(Feature33)
        
        
        win = 0.0001 * sample_rate
        freqs, psd = signal.welch(signal_window, sample_rate, nperseg=win)
        
        #PSD power in the signal Welch
        Feature38 = sum(psd)
        
        # Spectral peaks
        Feature39 =spectrumpeaks(psd)
        Feature39 = np.asarray(Feature39)
        
        #MeanFrequency
        Feature40 = meanfrequency(signal_window, sample_rate)
        
    
        #Wavelet Domain features
        
        enLev1,maxDecLev1,wpt1 = waveletenergy (signal_window)
        wav_energy = np.ravel(enLev1)
        
        #DWT statistical features
        w = 'db4'
        mode = pywt.Modes.smooth
        Wavelet_vectors = wavelet_features(signal_window, w, mode, maxDecLev1)
        wav_features=np.ravel(Wavelet_vectors)
        
        #CWT statistical features
        band = get_band(11,band_max_size)
        wav_power,w1,w2=CWTwaveletplot(signal_window,band,sample_rate)
        
        
        Feature= [Feature1,Feature2,Feature3,Feature4,Feature5,
               Feature6,Feature7,Feature8,Feature9,Feature10,Feature11,Feature12,Feature13,Feature14,Feature15,
               Feature16,Feature17,Feature18,Feature19,Feature20,Feature21,Feature22,Feature27,Feature38,Feature40]
            
        Feature_1=np.concatenate((Feature,Feature23,Feature28,Feature33,Feature39,wav_features,wav_energy,wav_power,w1,w2))
        
        
        label= [addlabels(class_label)]
        
        # Create the size of numpy array, by checking the size of "Feature_1" and creating "Feature_vectors" with the required shape on first run
        if  i ==0:
        #     print("--reached")
            size_of_Feature_vectors = int(len(Feature_1))
            size_of_class_vectors = int(len(label))
            size_of_dataset = int(len(signal_window))
            
            Feature_vectors=np.empty((0, size_of_Feature_vectors))
            classes=np.empty((0, size_of_class_vectors))
            rawdataset=np.empty((0, size_of_dataset))
               
       
        #print(label) 
        classes = np.append(classes,[label],axis=0)
        Feature_vectors = np.append(Feature_vectors,[Feature_1], axis=0)
        rawdataset = np.append(rawdataset,[signal_window],axis=0)
        
        
        print("Feature_vectors.shape",Feature_vectors.shape)
        
    return Feature_vectors,classes,rawdataset