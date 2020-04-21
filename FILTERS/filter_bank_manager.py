# -*- coding: utf-8 -*-
"""
@author: %(Mikel Val Calvo)s
@email: %(mikel1982mail@gmail.com)
@institution: %(Dpto. de Inteligencia Artificial, Universidad Nacional de Educación a Distancia (UNED))
"""


from scipy.signal import butter, iirnotch, filtfilt
import numpy as np
import pywt

class filter_bank_class(): 
    def __init__(self, constants):
        self.constants = constants
        
    def update_filters(self):
        self.b0, self.a0 = self.notch_filter()
        self.b, self.a = self.butter_bandpass()

    def pre_process(self, sample):
        sample = np.array(sample)
        [fil,col] = sample.shape	
        sample_processed = np.zeros([fil,col])
        for i in range(fil):
            data = sample[i,:] 
            data = data - np.mean(data) 	
            if self.constants.LOWCUT != None and self.constants.HIGHCUT != None: # 
                data = self.butter_bandpass_filter(data)
            data = data*1000000+(i+1)*100
            sample_processed[i,:] = data
  
        return sample_processed
        
    def notch_filter(self): # f0 50Hz, 60 Hz
        Q = 30.0  # Quality factor
        # Design notch filter
        b0, a0 = iirnotch(self.constants.NOTCH , Q, self.constants.SAMPLE_RATE)
        return b0,a0
    
    def butter_bandpass(self):
        nyq = 0.5 * self.constants.SAMPLE_RATE
        low = self.constants.LOWCUT / nyq
        high = self.constants.HIGHCUT / nyq
        b, a = butter(self.constants.ORDER , [low, high], btype='band')
        return b, a
    
    def butter_bandpass_filter(self, data):
        noth_data = filtfilt(self.b0, self.a0, data)
        band_passed_data = filtfilt(self.b, self.a, noth_data)
        return band_passed_data

    def butter_bandpass_specific_filter(self, data, lowcut, highcut, Fs, order):
        # -- notch filter --
        noth_data = filtfilt(self.b0, self.a0, data)
        # -- butterworth filter --
        nyq = 0.5 * Fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order , [low, high], btype='band')
        band_passed_data = filtfilt(b, a, noth_data)
        return band_passed_data
    
    def filter_bank(self, signal, Fs, filter_ranges, order=5):
        filterbank = []
        for [lowcut,highcut] in filter_ranges:
            y = self.butter_bandpass_specific_filter(signal, lowcut, highcut, Fs, order)
            filterbank.append(y)
        return np.asarray(filterbank)
    
    def eawica_wavelet_band_pass(self, signal, wavelet):
    
        import copy
        levels = 8
        coeffs = pywt.wavedec(signal, wavelet, mode='symmetric', level=levels, axis=-1)
        gamma_coeffs = copy.copy(coeffs)
        beta_coeffs = copy.copy(coeffs)
        alpha_coeffs = copy.copy(coeffs)
        theta_coeffs = copy.copy(coeffs)
        delta_coeffs = copy.copy(coeffs)
        
        # gamma
        for i in range(levels+1):
            if i != 7 and i != 8:
                gamma_coeffs[i] = np.zeros(gamma_coeffs[i].shape)
        # beta
        for i in range(levels+1):
            if i != 6:
                beta_coeffs[i] = np.zeros(beta_coeffs[i].shape)
        # alpha
        for i in range(levels+1):
            if i != 5:
                alpha_coeffs[i] = np.zeros(alpha_coeffs[i].shape)
        # theta
        for i in range(levels+1):
            if i != 4:
                theta_coeffs[i] = np.zeros(theta_coeffs[i].shape)
        # delta
        for i in range(levels+1):
            if i != 1 and i != 2 and i!= 3: 
                delta_coeffs[i] = np.zeros(delta_coeffs[i].shape)
                
        # -- reconstruction --
        gamma = pywt.waverec(gamma_coeffs, wavelet)
        beta = pywt.waverec(beta_coeffs, wavelet)
        alpha = pywt.waverec(alpha_coeffs, wavelet)
        theta = pywt.waverec(theta_coeffs, wavelet)
        delta = pywt.waverec(delta_coeffs, wavelet)
    
        return [gamma, beta, alpha, theta, delta]

    def wavelet_filter_aicaw(self, data, wavelet):
    
        import copy
        levels = 8
        coeffs = pywt.wavedec(data, wavelet, mode='symmetric', level=levels, axis=-1)
        gamma_coeffs = copy.copy(coeffs)
        beta_coeffs = copy.copy(coeffs)
        alpha_coeffs = copy.copy(coeffs)
        theta_coeffs = copy.copy(coeffs)
        delta_coeffs = copy.copy(coeffs)
        
        # gamma
        for i in range(levels+1):
            if i != 6:
                beta_coeffs[i] = np.zeros(beta_coeffs[i].shape)      
        # beta
        for i in range(levels+1):
            if i != 5:
                beta_coeffs[i] = np.zeros(beta_coeffs[i].shape)
        # alpha
        for i in range(levels+1):
            if i != 4:
                alpha_coeffs[i] = np.zeros(alpha_coeffs[i].shape)
        # theta
        for i in range(levels+1):
            if i != 3:
                theta_coeffs[i] = np.zeros(theta_coeffs[i].shape)
        # theta
        for i in range(levels+1):
            if i != 1 and i != 2:
                delta_coeffs[i] = np.zeros(delta_coeffs[i].shape)
           
        return [gamma_coeffs, beta_coeffs, alpha_coeffs, theta_coeffs, delta_coeffs]
        
    


