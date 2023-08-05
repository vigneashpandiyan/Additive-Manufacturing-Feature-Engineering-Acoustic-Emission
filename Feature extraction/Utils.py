# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 17:18:58 2023

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

#%%
def filter(signal_window,sample_rate):
    lowpass = 100000  # Cut-off frequency of the filter
    lowpass_freq = lowpass / (sample_rate / 2) # Normalize the frequency
    b, a = signal.butter(5, lowpass_freq, 'low')
    lowpassfilter = signal.filtfilt(b, a, signal_window)
    #plt.plot(t, lowpassfilter, label='Low-Pass')
    return lowpassfilter

def addlabels(class_label):
    
    return class_label


#%%

def Zerocross(a):
    zero_crossings = np.where(np.diff(np.signbit(a)))[0]
    cross = zero_crossings.size
    #print (cross)
    return cross

#%%

def meanfrequency(y,fs):    
    
    spec = np.abs(np.fft.rfft(y))
    freq = np.fft.rfftfreq(len(y), d=1/fs)    
    amp = spec / spec.sum()
    mean = (freq * amp).sum()
    return mean 

#%%

def get_band(band_size,band_max_size):
    band_window = 0
    band = []
    for y in range(band_size):
        band.append(band_window)
        band_window += band_max_size / band_size
    return band

def spectrumpower(psd, band,freqs):
    length = len(band)
    Feature_deltapower=[]
    Feature_relativepower=[]
    for i in range(5):
        if i <= (len(band)): 
            ii=i
            low = band[ii]
            ii=i+1;
            high = band[ii]
            #freq_res = freqs[high] - freqs[low] 
            idx_delta = np.logical_and(freqs >= low, freqs <= high)
            total_power = sum(psd)
            delta_power = sum(psd[idx_delta])
            delta_rel_power = delta_power / total_power
            Feature_deltapower.append(delta_power)
            Feature_relativepower.append(delta_rel_power)
                        
    return Feature_deltapower,Feature_relativepower


def spectrumpeaks(psd):
    import scipy.signal 
    indexes,value = scipy.signal.find_peaks(psd, height=0, distance=5)
    
    a=value['peak_heights']
    sorted_list = sorted(a, reverse=True)
    b = sorted_list[0:7]
    b_size = int(len(b))

    if b_size < 7:
        # calc missing values
        b_missing = 7 - b_size
        for x in range(b_missing):
            b.append(0)
#    b = [0,0,0]     
    return b

#%%


def autopeaks(psd):
    import scipy.signal 
    indexes,value = scipy.signal.find_peaks(psd, height=0, distance=None)
    a=value['peak_heights']
    sorted_list = sorted(a, reverse=True)
    b = sorted_list[0:4]
    b_size = int(len(b))

    if b_size < 4:
        # replace missing values with zeros
        b_missing = 4 - b_size
        for x in range(b_missing):
            b.append(0)     
    return b

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[len(result)//2:]

def get_autocorr_values(y_values):
    autocorr_values = autocorr(y_values)
    peaks=autopeaks (autocorr_values)
    peaks=np.asarray(peaks)
    return  peaks

#%%

def wavelet_features(a, w, mode, num_steps):
    ca = []
    cd = []
    for i in range(num_steps + 1):
        (a, d) = pywt.dwt(a, w, mode)
        ca.append(a)
        cd.append(d)
   
    length = len(ca)
    Wavelet_vectors=np.empty((0, 22))
    for i in range(len(ca)):
        signal=ca[i]
        W_1= max(signal)
        W_2= min(signal)
        W_3= np.std(signal)
        W_4= np.mean(signal)
        W_5= np.mean(abs(signal))
        W_6= statistics.harmonic_mean(abs(signal))
        W_7 = statistics.median(signal)
        W_8= skew(signal)
        W_9= kurtosis(signal)
        W_10 = statistics.variance(signal)
        W_11= np.sqrt(np.mean(np.square(signal)))
        
        signal1=cd[i]
        
        W_12= max(signal1)
        W_13= min(signal1)
        W_14= np.std(signal1)
        W_15= np.mean(signal1)
        W_16= np.mean(abs(signal1))
        W_17= statistics.harmonic_mean(abs(signal1))
        W_18 = statistics.median(signal1)
        W_19= skew(signal1)
        W_20= kurtosis(signal1)
        W_21= statistics.variance(signal1)
        W_22= np.sqrt(np.mean(np.square(signal1)))
        
        Wavelets= [W_1,W_2,W_3,W_4,W_5,W_6,W_7,W_8,W_9,W_10,W_11,W_12,W_13,W_14,W_15,W_16,W_17,W_18,W_19,W_20,W_21,W_22]
        
        Wavelet_vectors = np.append(Wavelet_vectors,[Wavelets], axis=0)
        
    return Wavelet_vectors

#%%
def waveletenergy (ampl1):
    WAVELET='db4'
    MODE='symmetric'
    #signal 1
    wp = pywt.WaveletPacket(data=ampl1, wavelet=WAVELET, mode=MODE)
    maxDecLev1=wp.maxlevel
    
    #print("Signal length: ",np.shape(ampl1),"Max dec.lev for sign. 1:",maxDecLev1)
    wpt=[]
    nodes=[]
    nodes.append("a")
    nodes.append("d")
    sumEnerg=0
    countDecLev=1
    
    #this are the rel.energies for each level
    RelEnrgLev1=[]
    
    while(countDecLev<=maxDecLev1):
        
        #print("Dec.lev:",countDecLev," ,nodes: ",nodes)
        tmpNodesLev=[]
        
        for n in range(0,len(nodes)):
            a=wp[nodes[n]].data#approximations  and details      
        
            sumCoef=0
            for n in range(0,len(a)):   
                curValue=abs(a[n])*abs(a[n])
                sumCoef=sumCoef+curValue           
                sumEnerg=sumEnerg+sumCoef
                        
            wpt.append(sumCoef)
            tmpNodesLev.append(sumCoef) #this we leave for further
        
        RelEnrgLev1.append(tmpNodesLev)
                
        #remake the nodes lists    
        tmpnodes=[]
        for b in range(0,len(nodes)):
            curNodeApproximations=nodes[b]+"a"
            curNodeDetails=nodes[b]+"d"
            tmpnodes.append(curNodeApproximations)
            tmpnodes.append(curNodeDetails)
        nodes=[]
        nodes=tmpnodes
        tmpnodes=[]
        #increment the decomposition level    
        countDecLev=countDecLev+1               
    
    #this is relative enerrgy bands
    for i in range(0,len(wpt)):
            wpt[i]=wpt[i]/sumEnerg
    wpt1=np.asarray(wpt)
    
    #print("Size of WPT array for sign.1: ",np.shape(wpt1))   
    #print("Total energy sign.1: ",sumEnerg," ,max energ.: ",np.max(wpt1)," ,min energ.: ",np.min(wpt1))   
    
    #energies for each leveö
    enLev1=[]
    for k1 in range(0,len(RelEnrgLev1)):
        sumedLevEn=0
        for k2 in range(0,len(RelEnrgLev1[k1])):
            sumedLevEn=sumedLevEn+RelEnrgLev1[k1][k2]
        enLev1.append(sumedLevEn)

    return enLev1,maxDecLev1,wpt1

#%%
def waveletpower(psd, band,freqs):
    length = len(band)
    waveletpower_deltapower=[]
    waveletpower_relativepower=[]
    for i in range(10):
        
        if i <= (len(band)): 
            ii=i
            low = band[ii]
            ii=i+1;
            high = band[ii]
            #freq_res = freqs[high] - freqs[low] 
            idx_delta = np.logical_and(freqs >= low, freqs <= high)
            total_power = sum(psd)
            delta_power = sum(psd[idx_delta])
            
            delta_rel_power = delta_power / total_power
            waveletpower_deltapower.append(delta_power)
            waveletpower_relativepower.append(delta_rel_power)
                        
    return waveletpower_deltapower,waveletpower_relativepower

#%%

def CWTwaveletplot(rawspace,band,sample_rate):
    
    windowsize=len(rawspace)
    t0=0
    dt=1/sample_rate
    time = np.arange(0, windowsize) * dt + t0
    
    
    scales = np.arange(1, 50)
    waveletname = 'morl'
    
    data_new=rawspace
    data_new= filter(data_new,sample_rate)
    dt = time[1] - time[0]
    [coefficients, frequencies] = pywt.cwt(data_new, scales, waveletname, dt)
    wavenergy = (abs(coefficients)) ** 2
    wavefrequencyenergy=wavenergy.sum(axis=1)
    w1,w2=waveletpower(wavefrequencyenergy, band,frequencies)
    w1=np.ravel(w1)
    w2=np.ravel(w2)
    
    wavenergy=np.ravel(wavenergy)
    
    Feature1 = wavenergy.min()
    Feature2 = wavenergy.max()
    Feature3 = Feature2-Feature1
    Feature4 = np.sqrt(np.mean(wavenergy**2))
    Feature5 = wavenergy.mean()
    Feature6 = wavenergy.std()
    Feature7 = statistics.variance(wavenergy)
    Feature8 = statistics.harmonic_mean(abs(wavenergy))
    Feature9 = statistics.median(wavenergy)
    
    
    power= [Feature1,Feature2,Feature3,Feature4,Feature5,Feature6,Feature7,Feature8,Feature9]
    power = np.ravel(power)
    return power,w1,w2
