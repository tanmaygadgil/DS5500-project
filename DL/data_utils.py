import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
import numpy as np
from feature_ops import *
from wav_utils import (extract_basename,
                       extract_info_from_name,
                       extract_unique_values_from_folder,
                       get_label, load_file)

U_INDICES = [16, 23, 24, 31]
NO_U_INDICES = [0,1,2,3,4,5,6,7,8,9,
                10,11,12,13,14,15,17,
                18,19,20,21,22,25,26,
                27,28,29,30]

    
class Ops(object):
    """Class to coordinate multiple operations
    """
    def __init__(self, op_list) -> None:
        self.op_list = op_list
        
    def transform(self, x):
        """Executes a set of transform in order

        Args:
            x (_type_): np array

        Returns:
            _type_: _description_
        """
        features = np.array([])
        for op in self.op_list:
            res = op(x)
            
            features = np.hstack([features, res])
            
        return features

class FileDataset(Dataset):
    """Dataset which contains a list of .hea and .dat files 
    
    Here the assumption is made that each folder just has .dat, .hea files
    """
    def __init__(self, folder_list, transforms:Ops) -> None:
        super().__init__()
        self.folder_list = folder_list
        self.file_list = []
        self.labels = []
        for folder in folder_list:
            unique_sample_names = extract_unique_values_from_folder(folder)
            self.file_list += unique_sample_names
            
        for file in self.file_list:
            self.labels.append(get_label(file))
            
        self.transforms = transforms
        
        
        
    def __getitem__(self, index):
        
        wave_data = load_file(self.file_list[index])
        features = self.transforms.transform(wave_data)
        return torch.tensor(features, dtype=torch.float32), self.labels[index]
    
    def __len__(self):
        return len(self.file_list)
    
    
class FileDataLoader(DataLoader):
    
    def __init__(self, dataset, batch_size=1, 
                 shuffle=False, sampler=None, 
                 batch_sampler=None, num_workers=0):
        super().__init__(dataset, batch_size, 
                         shuffle, sampler,
                         batch_sampler, num_workers)
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.num_workers = num_workers
        
    def __iter__(self):
        return super().__iter__()
    
    def __len__(self):
        return self.dataset.__len__()
    
