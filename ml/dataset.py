import pandas
import numpy as np
from torch.utils.data import Dataset
from scipy.interpolate import interp1d


def interpolate_data(data, new_length=1300):
    old_indices = np.linspace(0, len(data)-1, num=len(data))
    new_indices = np.linspace(0, len(data)-1, num=new_length)
    interpolator = interp1d(old_indices, data)
    new_data = interpolator(new_indices)
    fft = np.fft.fft(new_data).flatten()
    res = np.concatenate((new_data, fft))
    return res

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


class PeaksDataset(Dataset):
    def __init__(self):
        self.data = pandas.read_csv('../peaks_dataset.csv').to_numpy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        with open(self.data[idx][0], 'rb') as f:
            signal = np.load(f)
        signal = signal[int(self.data[idx][1])]
        with open(self.data[idx][2], 'rb') as f:
            ground_truth = np.load(f)
        return interpolate_data(signal, 1300), ground_truth