import numpy as np
import pandas
from matplotlib import pyplot as plt
from scipy import signal
from sklearn.decomposition import PCA
import scipy.interpolate as interp

import pandas as pd


class Signal(list):
    def append(self, __object):
        if len(self) == 0:
            self.init_vector(__object)
        else:
            for i, j in enumerate(__object):
                self[i].append(j[1])

    def init_vector(self, __object):
        for i in __object:
            super().append([i[1]])


class PCAList(list):
    def append(self, __object):
        if len(self) == 0:
            for i in __object:
                super().append(i)
        else:
            for i, j in enumerate(__object):
                self[i] = self[i] + j

        diff = []
        for i in range(len(self[0])):
            diff.append(self[3][i] - self[4][i])

        if len(self) == 5:
            super().append(diff)
        else:
            self[5] = diff

        sum_ = []
        for i in range(len(self[0])):
            sum_.append(self[3][i] + self[4][i])

        if len(self) == 6:
            super().append(sum_)
        else:
            self[6] = sum_

    def get(self, time_range=1, freq=250):
        res = []
        len_ = len(self[0])
        for i in self:
            res.append(i[len_ - time_range*freq:len_])
        return np.array(res)


class PeaksList(list):
    def append(self, __object):
        if len(self) == 0:
            for i in __object:
                super().append(i)
        else:
            for i, j in enumerate(__object):
                self[i][0] = self[i][0] + j[0]
                self[i][1] = self[i][1] + j[1]
                self[i][2] = j[2]


class SignalProcessing:
    def __init__(self, freq):

        self.signal = Signal()
        self.pca_transform = PCA(n_components=5)
        self._old_freq = freq
        self._new_freq = 250
        self.vector = []
        self.plot_data = PCAList()
        self.index = 0
        self.peaks_list = PeaksList()
        self.fig, self.axes = plt.subplots(1, 8, figsize=(30, 5))
        plt.ion()
        self.time_range = 2
        self.last_update = 0
        self.std_coef = 1.5

        data = pandas.read_csv('../csv/pulsar_data_1662790981.4648633.csv')
        self.dataset = []
        for i in data.values:
            tmp = list(map(lambda x: int(x), i[1][1:-1].split(', ')))
            for i in tmp:
                self.dataset.append(i)

        self.gt_peaks = []

    def append(self, data):
        self.signal.append(data)
        if len(self.signal[0]) % self._old_freq == 0:
            self.index += 1
        if len(self.signal[0]) % self._old_freq == 0:
            self.interpolate_signal()
            self.butter_filter()
            pca = self.pca(self.vector)
            self.plot_data.append(pca.tolist())
            if self.index % self.time_range == 0:
                self.compute_peaks(self.plot_data.get(time_range=self.time_range))
            self.vector.clear()
            self.signal.clear()
            self.update_plots()
            plt.pause(0.1)
            return True
        return False

    def interpolate_signal(self):
        for i in self.signal:
            x = np.arange(0, len(i) / self._old_freq, 1 / self._old_freq)
            if len(x) > len(i):
                x = np.delete(x, -1)
            # x_int = np.linspace(np.min(x), np.max(x), int(len(x) * 8.333))
            x_int = np.linspace(np.min(x), np.max(x), self._new_freq)
            cs = interp.CubicSpline(x, i)
            self.vector.append(cs(x_int))

    def butter_filter(self):
        passband_freq = [0.75, 5]
        order = 5
        nyquist_freq = 0.5 * self._new_freq
        normalized_passband = [freq / nyquist_freq for freq in passband_freq]
        b, a = signal.butter(order, normalized_passband, btype='band', analog=False, output='ba')

        vector = []
        for i in self.vector:
            vector.append(signal.lfilter(b, a, i))
        self.vector = vector

    def compute_peaks(self, data):
        peaks_list = []
        self.last_update += self.time_range
        begin  = self.index - self.time_range if self.index - self.time_range > 0 else 0
        x = np.linspace(begin, self.index, self._new_freq*self.time_range)
        for i, vector in enumerate(data):
            peaks, _ = signal.find_peaks(vector)
            prominences = signal.peak_prominences(vector, peaks)[0]
            peaks = peaks[prominences > self.std_coef*np.std(prominences)]
            peaks_list.append([x[peaks].tolist(), vector[peaks].tolist(), len(peaks)])
        self.peaks_list.append(peaks_list)


    def pca(self, data):
        XPCA_reduced = self.pca_transform.fit_transform(np.transpose(data))
        return XPCA_reduced.transpose()

    def update_plots(self):
        x = np.linspace(0, self.index, self._new_freq*self.index)

        for i, data in enumerate(self.plot_data):
            # if i == 6:
            #     self.axes[0].clear()
            #     self.axes[0].plot(x, data, label='data')
            #     if len(self.peaks_list) != 0:
            #         peaks, sig, count = self.peaks_list[i]
            #         self.axes[0].plot(peaks, sig, "x", color='orange', label='peaks')
            #         self.axes[0].set_title(' '.join([f'{i}, pred window{ count*12 }/{int(len(peaks)*60/self.last_update)}']))

            self.axes[i].clear()
            self.axes[i].plot(x, data, label='data')
            if len(self.peaks_list) != 0:
                peaks, sig, count = self.peaks_list[i]
                self.axes[i].plot(peaks, sig, "x", color='orange', label='peaks')
                self.axes[i].set_title(' '.join([f'{i}, pred window{ count*12 }/{int(len(peaks)*60/self.last_update)}']))
                # self.axes[i].text(0, 0, f'{i},{ count*12 }')
                # print(f'{i}, count ={ count*12 }')
            # self.axes[i].legend()

            # plt.plot(signal)
            # plt.plot(peaks, signal[peaks], "x")
            # plt.show()
        #print((int(len(self.peaks_list[5][0])*60/self.last_update)+(int(len(self.peaks_list[6][0])*60/self.last_update)))//2)
        x = np.linspace(0, self.index, 130 * self.index)
        y = self.dataset[0:self.index*130]
        peaks, _ = signal.find_peaks(y)
        prominences = signal.peak_prominences(y, peaks)[0]
        peaks = peaks[prominences > 2 * np.std(prominences)]
        self.axes[7].clear()
        self.axes[7].plot(x, y)
        self.axes[7].set_title(' '.join([f'{int(len(peaks) * 60 / self.index)}']))
