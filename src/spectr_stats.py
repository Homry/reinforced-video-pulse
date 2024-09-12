import numpy as np
from scipy.integrate import simpson
from sklearn.preprocessing import normalize
from scipy.signal import find_peaks, peak_prominences
from scipy.fftpack import rfft, rfftfreq
import matplotlib.pyplot as plt


def integrate(x, y):
    return simpson(y, x)


def create_spectr(signal, rate):
    N = len(signal)
    yf = rfft(signal)
    xf = rfftfreq(N, 1 / rate)
    return xf, np.abs(yf)


def polynomial_smoothing(data, window_size=40, polynomial_order=4):
    if len(data.shape) == 1:
        return np.convolve(data, np.ones(window_size)/window_size, mode='same')
    smoothed_data = []
    for signal_ in data:
        # smoothed_data.append(savgol_filter(signal_, window_size, polynomial_order))
        smoothed_data.append(np.convolve(signal_, np.ones(window_size)/window_size, mode='same'))
    return smoothed_data


def find(wave):
    wave = polynomial_smoothing(np.array(wave))
    peaks, _ = find_peaks(wave)
    wave = np.array(wave)

    prominences = peak_prominences(wave, peaks)[0]

    peaks = peaks[prominences > 0.35 * np.std(prominences)]
    print('peaks', len(peaks))

def pipeline(data, rate):
    square = [integrate(*create_spectr(i, rate)) for i in data]
    print(square)

    for i, signal in enumerate(data):
        coef = 15
        find(signal)
        x, y = create_spectr(signal, rate)
        max_int = integrate(x, y)
        peaks, _ = find_peaks(y)
        tmp = [y[k] for k in peaks]
        std = np.std(tmp)
        # print(f'tmp std {std}')
        peaks = np.array([k for k in peaks if y[k] > coef*std])
        y = y[peaks[0]:peaks[-1]+1]
        x = x[peaks[0]:peaks[-1]+1]
        wind_int = integrate(x, y)
        print(f'integ {wind_int}')
        print(f' отношение = {wind_int/max_int}')
