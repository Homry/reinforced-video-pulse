from __future__ import annotations

from typing import Type

import numpy as np
import torch
import random

import tqdm
from scipy.signal import find_peaks, peak_prominences
import pandas
from ml.classifire import Classifire, SelectModel
from ml.regress_model import Regress



def polynomial_smoothing(data, window_size=40, polynomial_order=4):
    if len(data.shape) == 1:
        return np.convolve(data, np.ones(window_size) / window_size, mode='same')
    smoothed_data = []
    for signal_ in data:
        # smoothed_data.append(savgol_filter(signal_, window_size, polynomial_order))
        smoothed_data.append(np.convolve(signal_, np.ones(window_size) / window_size, mode='same'))
    return np.array(smoothed_data)


def correlation_after_pca(vector, coef):

    rate = 250
    res = []
    for i in range(len(vector)):
        wave = vector[i]
        x = np.arange(0, len(wave) / rate, 1 / rate)

        # wave = polynomial_smoothing(np.array(wave))
        peaks, _ = find_peaks(wave)
        wave = np.array(wave)

        prominences = peak_prominences(wave, peaks)[0]

        peaks = peaks[prominences > coef * np.std(prominences)]
        res.append(len(peaks))
    return np.array(res)


def load_model(model_class: Type[SelectModel] | Type[Regress], model_file):
    model = model_class()
    model.load_state_dict(torch.load(model_file))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model


def load_file(file: str):
    with open(file, 'rb') as f:
        return torch.Tensor(np.array([np.load(f)])).to('cuda').float()


def get_gt_peaks(file):
    with open(file, 'rb') as f:
        wave = np.load(f)
        peaks, _ = find_peaks(wave)
        prominences = peak_prominences(wave, peaks)[0]
        peaks = peaks[prominences > 3 * np.std(prominences)]
        return len(peaks)


def choose_true_signal(model: SelectModel, signals: torch.Tensor):

    return torch.argmax(model(signals), dim=1).item()


def get_std_coef(model: Regress, signals: torch.Tensor):
    return model(signals).item()

if __name__ == "__main__":
    dataset_path = '../dataset_new.csv'
    dataset = pandas.read_csv(dataset_path).to_numpy()
    classifire = load_model(SelectModel, "./selection.pth")
    regress = load_model(Regress, "regress.pth")
    res = []
    new_dataset = []
    for i in tqdm.tqdm(range(len(dataset))):
        signals = load_file(dataset[i][0])
        # index = choose_true_signal(classifire, signals)
        std_coef = get_std_coef(regress, signals)
        gt_peaks = get_gt_peaks(dataset[i][1])
        predict_peaks = correlation_after_pca(polynomial_smoothing(signals[0].cpu().detach().numpy()), std_coef)
        diff = np.abs(predict_peaks - gt_peaks)
        res.append(min(diff))
        new_dataset.append([dataset[i][0], min(diff)])
    df = pandas.DataFrame(new_dataset)
    df.to_csv('./new_sample.csv', index=False)

    print(np.mean(res))



