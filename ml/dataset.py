import torch
import numpy as np
from torch.utils.data import Dataset

class PulseDataset(Dataset):
    def __init__(self, data_path):
        # Initialize the dataset
        self.data = []  # Placeholder for your data
        # Load or preprocess your data here
        
    def __len__(self):
        
        return len(self.data)
    
    def __getitem__(self, path):
        with open(base + path, 'rb') as f:
            signal = np.load(f) 
        with open(base_gt + path, 'rb') as f:
            ground_truth = np.load(f)  
        
        
        return signal, ground_truth
