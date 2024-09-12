import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks, peak_prominences

base = './dataset_npy/gt'
new = './dataset_npy/transform'
os.mkdir('./dataset_npy/transform')
for file in os.listdir(base):
    os.mkdir(f'{new}/{file}')
    for name in os.listdir(f'{base}/{file}'):
        with open(f'{base}/{file}/{name}', 'rb') as f:
            wave = np.load(f)
        peaks, _ = find_peaks(wave)
        prominences = peak_prominences(wave, peaks)[0]
        test = peaks[prominences > 2.5 * np.std(prominences)]
        data = np.array([1 if i in test else 0 for i in range(len(wave))])
        with open(f'{new}/{file}/{name}', 'wb') as f:
            np.save(f, data)