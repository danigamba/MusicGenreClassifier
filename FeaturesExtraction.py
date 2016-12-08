#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 20:45:34 2016

@author: Daniele Gamba
"""
import numpy as np
from scipy.io import wavfile
from scipy.signal import welch

def extractFeatures(filename, nFeatures):
    rate, data = wavfile.read(filename)
    f, Pxx = welch(data, fs=rate, nperseg=1024)
    if nFeatures == 127:
        return Pxx[[i for i in range(0, len(f)) if f[i] <= 5000]] #whole spectrum from 0 to 5kHz
    
    x = Pxx[[i for i in np.arange(0, 20*nFeatures, 20)]]    #first n point 20Hz harmonics - heuristic
    return x / np.max(x)    #normalization