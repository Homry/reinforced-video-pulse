import numpy as np
import pandas
from torch.utils.data import Dataset
from scipy.interpolate import interp1d



def load(name):
    with open(name, 'rb') as f:
        return np.load(f)

def interpolate_data(data, new_length=1300):
    old_indices = np.linspace(0, len(data)-1, num=len(data))
    new_indices = np.linspace(0, len(data)-1, num=new_length)
    interpolator = interp1d(old_indices, data)
    new_data = interpolator(new_indices)
    fft = np.fft.fft(new_data).flatten()
    res = np.concatenate((new_data, fft))
    return res

def get_ground_truth(path):
    with open(path, 'rb') as f:
        ground_truth = np.load(f)
        # ground_truth = np.where(ground_truth == 1, 0, 1)
    return ground_truth


class PeaksDataset(Dataset):
    def __init__(self):
        self.data = pandas.read_csv('../../peaks_dataset.csv').to_numpy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        with open(self.data[idx][0], 'rb') as f:
            signal = np.load(f)
        signal = signal[int(self.data[idx][1])]
        ground_truth = get_ground_truth(self.data[idx][2])
        return interpolate_data(signal, 1300), ground_truth


class BoostDatasetWindow(Dataset):
    def __init__(self):
        self.data = pandas.read_csv('../../peaks_dataset_new.csv').to_numpy()
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = load(self.data[idx][0])
        return data, int(self.data[idx][1])
