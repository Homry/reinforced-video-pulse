import numpy as np
import scipy.interpolate as interp
from scipy import signal
import statistics
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from scipy.signal import find_peaks, savgol_filter
from scipy.fftpack import fft, ifft, fftfreq, fftshift


class TimeSeries:
    def __init__(self, debug=False):
        self.__vector = None
        self.__old_freq = 30
        self.__new_freq = 250
        self.pca_transform = PCA(n_components=4)
        self.debug = debug

    def init_vector(self, vector):
        self.__vector = [[i[1]] for i in vector]

    def add_in_vector(self, vector, status=None):
        [self.__vector[i].append(vector[i][1]) for i, j in enumerate(vector)]

    def __str__(self):
        return f'{self.__vector}, {len(self.__vector[0])}'

    def interpolate_signal(self):
        vector = []
        for i in self.__vector:
            x = np.arange(0, len(i) / self.__old_freq, 1 / self.__old_freq)
            if len(x) > len(i):
                x = np.delete(x, -1)
            x_int = np.linspace(np.min(x), np.max(x), int(len(x) * 8.333))
            cs = interp.CubicSpline(x, i)
            vector.append(cs(x_int))
        self.__vector = vector
        # for i, j in enumerate(self.__vector):
        #     x = np.arange(0, len(j) / self.__new_freq, 1 / self.__new_freq)
        #     plt.plot(x, j)
        #     plt.show()

    def filter_by_len(self):
        len_vector = [len(i) for i in self.__vector]
        uniq_len = list(set(len_vector))
        count_of_len = {i: [] for i in uniq_len}
        list(map(lambda x: count_of_len[x].append(1), len_vector))
        count_of_len = {i: len(j) for i, j in count_of_len.items()}

        max_key, max_item = list(count_of_len.items())[0]
        for key, item in count_of_len.items():
            if item > max_item:
                max_key, max_item = key, item

        self.__vector = [i for i in self.__vector if len(i) == max_key]


    def distance_filter(self):
        vector = []

        for i in self.__vector:
            distance = [abs(i[j] - i[j + 1]) for j in range(len(i) - 1)]
            mode = statistics.median(distance)
            status = [True if j <= mode else False for j in distance]
            if False not in status:
                vector.append(i)
        self.__vector = vector

    def butter_filter(self):
        passband_freq = [0.75, 5]
        order = 5
        nyquist_freq = 0.5 * self.__new_freq
        normalized_passband = [freq / nyquist_freq for freq in passband_freq]
        b, a = signal.butter(order, normalized_passband, btype='band', analog=False, output='ba')

        vector = []
        for i in self.__vector:
            vector.append(signal.lfilter(b, a, i))
        self.__vector = vector

    def slice_vector(self, window):
        window_begin = 0
        window_end = self.__new_freq*window
        window_offset = (self.__new_freq // 2)*window
        end_of_data = len(self.__vector[0])
        result = []
        while window_end <= end_of_data:
            data = []
            for signal_ in self.__vector:
                data.append(signal_[window_begin:window_end])
            window_begin += window_offset
            window_end += window_offset
            result.append(data)
        else:
            data = []
            for signal_ in self.__vector:
                data.append(signal_[window_begin:end_of_data])
            result.append(data)
        return result

    def pca(self, data):
        XPCA_reduced = self.pca_transform.fit_transform(np.transpose(data))
        return XPCA_reduced.transpose()

    @staticmethod
    def polynomial_smoothing(data, window_size=40, polynomial_order=4):
        smoothed_data = []
        for signal_ in data:
            # smoothed_data.append(savgol_filter(signal_, window_size, polynomial_order))
            smoothed_data.append(np.convolve(signal_, np.ones(window_size)/window_size, mode='same'))
        return smoothed_data

    def find_signals_peaks(self, vector, debug_vector=None):
        all_beats = []
        freq_ = []
        per = []
        for i, signal_ in enumerate(vector):
            x = np.arange(0, len(signal_) / self.__new_freq, 1 / self.__new_freq)
            peaks, _ = find_peaks(signal_)
            f = plt.figure(i)
            heart_beat = len(peaks) / (len(signal_) / self.__new_freq) * 60
            all_beats.append(heart_beat)
            print("beat", heart_beat)

            plt.plot(x, debug_vector[i], color='red', label='original')
            plt.plot(x[peaks], signal_[peaks], "o", color='orange', label='peaks')
            plt.plot(x, signal_, label='smooth')
            plt.legend()

            N = len(signal_)
            # sample spacing
            T = 1.0 / self.__new_freq

            # Get fft
            spectrum = np.abs(fft(signal_))
            spectrum *= spectrum
            xf = fftfreq(N, T)

            # Get maximum ffts index from second half
            maxInd = np.argmax(spectrum[:int(len(spectrum)/2)+1])
            # maxInd = np.argmax(spectrum)
            maxFreqPow = spectrum[maxInd]
            maxFreq = np.abs(xf[maxInd])

            total_power = np.sum(spectrum)
            # Get max frequencies power percentage in total power
            percentage = maxFreqPow / total_power
            freq_.append(maxFreq)
            per.append(percentage)
            print(f' maxFreq = {maxFreq}, percentage = {percentage}, bpm = {60*maxFreq}')

            # print("beat", heart_beat)
        # print(f'average = {sum(all_beats) / len(all_beats)}')
        idx = np.argmax(per)
        print(f'bpm = {60*freq_[idx]}')
        plt.show()


