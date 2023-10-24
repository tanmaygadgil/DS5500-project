import numpy as np
import scipy.stats as stats
from scipy.fft import fft, fftfreq

def mean(array:np.ndarray):
    return np.mean(array, axis = 0)

def std(array:np.ndarray):
    return np.std(array, axis = 0)

def min_feature(array:np.ndarray):
    return np.min(array, axis = 0)

def max_feature(array:np.ndarray):
    return np.max(array, axis = 0)

def rms(array:np.ndarray):
    return np.sqrt(np.mean(array**2, axis = 0))

def power(array:np.ndarray):
    return np.mean(array**2, axis = 0)

def mav(array:np.ndarray):
    return  np.mean(np.abs(array), axis = 0)

def peak(array:np.ndarray):
    return  np.max(np.abs(array), axis = 0)

def form_factor(array:np.ndarray):
    return np.sqrt(np.mean(array**2, axis = 0)) / np.mean(array, axis = 0)
    
def pulse_indicator(array:np.ndarray): 
    return np.max(np.abs(array), axis = 0)/np.mean(array, axis = 0)

def mean_f(array:np.ndarray):
    freq_data = fft(array)
    S_f = np.abs(freq_data**2) / len(freq_data)
    return np.max(S_f, axis = 0)

def std_f(array:np.ndarray):
    freq_data = fft(array)
    S_f = np.abs(freq_data**2)/len(freq_data)
    return np.std(S_f, axis = 0)

def max_f(array:np.ndarray):
    freq_data = fft(array)
    S_f = np.abs(freq_data**2)/len(freq_data)
    return np.max(S_f, axis = 0)

def min_f(array:np.ndarray):
    freq_data = fft(array)
    S_f = np.abs(freq_data**2)/len(freq_data)
    return np.min(S_f, axis = 0)