import numpy as np
import pandas as pd
from scipy.signal import peak_prominences, find_peaks
from tqdm import tqdm


def predict_pipeline(file):
    with open(file, 'rb') as f:
        return np.load(f)


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
    coef = 0.1
    best_coef = None
    min_val = None
    gt_peaks = get_gt_peaks(dataset[i][1])
    wave = polynomial_smoothing(predict_pipeline(dataset[i][0]))
    while coef <= 2:
        pred_peaks = correlation_after_pca(wave, coef)
        diff = np.abs(pred_peaks - gt_peaks)
        if min_val is None or min(diff) < min_val:
            min_val = min(diff)
            best_coef = coef
        coef += 0.1
    data.append([dataset[i][0], best_coef])
    test.append(best_coef)
print(np.mean(test))
# print(np.mean(test))
df = pd.DataFrame(data)
df.to_csv('./regress_dataset.csv', index=False)
