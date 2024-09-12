
from scipy.signal import find_peaks, savgol_filter, peak_prominences


def polynomial_smoothing(data, window_size=40, polynomial_order=4):

    return np.convolve(data, np.ones(window_size) / window_size, mode='same')

import pandas
data = pandas.read_csv('./csv/pulsar_data_1662790981.4648633.csv')

dataset = []
for i in data.values:
    tmp = list(map(lambda x: int(x), i[1][1:-1].split(', ')))
    for i in tmp:
        dataset.append(i)
import numpy as np

# with open('./dataset_npy/gt/2022-09-10 09-22-51/2022-09-10 09-22-51_5-15.npy', 'rb') as f:
#     dataset = np.load(f)

from matplotlib import pyplot as plt

time_range = [75]
for i in time_range:
    wave = dataset
    x = np.arange(0, len(wave) / 130, 1 / 130)
    wave = np.array(wave)
    peaks, _ = find_peaks(wave)
    prominences  =  peak_prominences(wave, peaks)[0]
    std = np.std(wave[peaks])
    test = peaks[prominences > 3*np.std(prominences)]


    plt.plot(x, wave)
    # plt.plot(x[test], wave[test], "o", color='orange')
    plt.hlines(y=std, color='red', linestyle='-', xmin=0, xmax=len(wave) / 130)
    print(len(wave))
    plt.show()

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

