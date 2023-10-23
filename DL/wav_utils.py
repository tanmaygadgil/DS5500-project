import numpy as np
import torch
import re
import os
import wfdb

U_INDICES = [16, 23, 24, 31]
NO_U_INDICES = [0,1,2,3,4,5,6,7,8,9,
                10,11,12,13,14,15,17,
                18,19,20,21,22,25,26,
                27,28,29,30]

def extract_info_from_name(filename):
    parsed = {}
    basename = os.path.basename(filename).split(".")[0]
    pattern = r"session(\d+)_participant(\d+)_gesture(\d+)_trial(\d+)"
    match = re.match(pattern, basename)
    parsed['session'] = match.group(1) 
    parsed['participant'] = match.group(2)
    parsed['gesture'] = match.group(3) 
    parsed['trial'] = match.group(4)
    parsed['filename'] = filename
    return parsed

def extract_basename(filename):
    """
    Function to extract just the unique headers of the dat and hea files (removing the file spec) 
    """
    return os.path.splitext(filename)[0]


def extract_unique_values_from_folder(folder:str):
    """
    Function to extract just the unique headers of the dat and hea files (removing the file name) 
    """
    unique = set()
    for f in os.listdir(folder):
        unique.add(extract_basename(os.path.join(folder, f)))
        
    return list(unique)

def get_label(file):
    info = extract_info_from_name(file)
    
    return int(info['gesture'])

def load_file(file:str):
    """Load in a wav file

    Args:
        file (str): This is a filename where the .dat / .hea 
                    has been stripped off

    Returns:
        _type_: Nd.Array
    """
    
    wave_file = wfdb.rdrecord(file)
    wave_data = wave_file.p_signal
    #filter out the U signal
    wave_data = wave_data[:, NO_U_INDICES]
    
    return wave_data
    
    