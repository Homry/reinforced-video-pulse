import pandas
import numpy as np
from torch.utils.data import Dataset


class PulseDataset(Dataset):
    def __init__(self):
        self.data: np.array = pandas.read_csv('../regress_dataset.csv').to_numpy()  # Placeholder for your data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int):
        with open(self.data[idx][0], 'rb') as f:
            signal = np.load(f)
        ground_truth = float(self.data[idx][1])
        return signal, ground_truth


class ClassifireDataset(Dataset):
    def __init__(self):
        # Initialize the dataset
        self.data: np.array = pandas.read_csv('../classifire_dataset.csv').to_numpy()  # Placeholder for your data
        # Load or preprocess your data here

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        with open(self.data[idx][0], 'rb') as f:
            signal = np.load(f)
        ground_truth = self.data[idx][1]
        return signal, np.array(list(map(lambda x: int(x), ground_truth[1:-1].split(', '))))


class SelectionDataset(Dataset):
    def __init__(self):
        # Initialize the dataset
        self.data: np.array = pandas.read_csv('./selection_dataset.csv').to_numpy()  # Placeholder for your data
        # Load or preprocess your data here

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        with open(self.data[idx][0], 'rb') as f:
            signal = np.load(f)
        return signal, int(self.data[idx][1])
