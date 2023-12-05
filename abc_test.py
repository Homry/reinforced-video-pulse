
from scipy.signal import find_peaks, savgol_filter, peak_prominences


def polynomial_smoothing(data, window_size=40, polynomial_order=4):

    return np.convolve(data, np.ones(window_size) / window_size, mode='same')

import pandas
data = pandas.read_csv('./csv/pulsar_data_1662804846.0179787.csv')

dataset = []
for i in data.values:
    tmp = list(map(lambda x: int(x), i[1][1:-1].split(', ')))
    for i in tmp:
        dataset.append(i)

from matplotlib import pyplot as plt
import numpy as np

time_range = [75]
for i in time_range:
    wave = dataset[i*60*130:(i*60+10)*130]
    x = np.arange(0, len(wave) / 130, 1 / 130)

    peaks, _ = find_peaks(wave)
    prominences  =  peak_prominences(wave, peaks)[0]

    test = peaks[prominences > 2*np.std(prominences)]

    wave = np.array(wave)
    plt.plot(x[test], [0 for i in test], "o", color='orange', label='peaks')
    heart_beat = len(test)
    print(f'heart_beat gt = {heart_beat}')



    # import numpy as np
    # import matplotlib.pyplot as plt
    # file_name = str(i)
    #
    #
    #
    #
    # with open(f'./npy_data/2022-09-10 09-22-51/{file_name}/gt.npy', 'wb') as f:
    #     np.save(f, x[test])

