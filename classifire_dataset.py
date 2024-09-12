import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import peak_prominences, find_peaks
from tqdm import tqdm


def predict_pipeline(file):
    with open(file, 'rb') as f:
        data = np.load(f)
        return correlation_after_pca(data)


def polynomial_smoothing(data, window_size=40, polynomial_order=4):
    if len(data.shape) == 1:
        return np.convolve(data, np.ones(window_size) / window_size, mode='same')
    smoothed_data = []
    for signal_ in data:
        # smoothed_data.append(savgol_filter(signal_, window_size, polynomial_order))
        smoothed_data.append(np.convolve(signal_, np.ones(window_size) / window_size, mode='same'))
    return np.array(smoothed_data)

def correlation_after_pca(vector):
    rate = 250
    res = []
    for i in range(len(vector)):
        wave = vector[i]
        x = np.arange(0, len(wave) / rate, 1 / rate)

        wave = polynomial_smoothing(np.array(wave))
        peaks, _ = find_peaks(wave)
        wave = np.array(wave)

        prominences = peak_prominences(wave, peaks)[0]

        peaks = peaks[prominences > 0.35 * np.std(prominences)]
        res.append(len(peaks))
    return res


def get_gt_peaks(file):
    with open(file, 'rb') as f:
        wave = np.load(f)
        peaks, _ = find_peaks(wave)
        prominences = peak_prominences(wave, peaks)[0]
        peaks = peaks[prominences > 3 * np.std(prominences)]
        return len(peaks)


dataset = pd.read_csv('./dataset.csv').to_numpy()
data = []
test = []

for i in tqdm(range(len(dataset))):
    pred_peaks = np.array(predict_pipeline(dataset[i][0]))
    gt_peaks = get_gt_peaks(dataset[i][1])
    diff = np.abs(pred_peaks-gt_peaks)
    test.append(min(diff))
    res = [0 for i in range(5)]
    res[np.argmin(diff)] = 1
    data.append([dataset[i][0], res])
print(np.mean(test))
df = pd.DataFrame(data)
df.to_csv('./classifire_dataset.csv', index=False)



