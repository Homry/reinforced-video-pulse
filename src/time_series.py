import numpy as np
import scipy.interpolate as interp
from scipy import signal
import statistics
from src import pipeline
from itertools import zip_longest
from scipy.integrate import simpson, trapz
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from scipy.signal import find_peaks, savgol_filter, peak_prominences
from scipy.fftpack import rfft, ifft, rfftfreq, fftshift
import os
number = 0

class TimeSeries:
    def __init__(self, debug=False):
        self.__vector = None
        self.__old_freq = 30
        self.__new_freq = 250
        self.pca_transform = PCA(n_components=5)
        self.debug = debug
        self.used_signal = []

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
        no_else = False
        while window_end <= end_of_data:
            data = []
            if window_end == end_of_data:
                no_else = True
            for signal_ in self.__vector:
                data.append(signal_[window_begin:window_end])
            window_begin += window_offset
            window_end += window_offset
            result.append(data)
        else:
            data = []
            if window_begin < end_of_data and not no_else:
                for signal_ in self.__vector:
                    data.append(signal_[window_begin:end_of_data])
                result.append(data)
        return result

    def pca(self, data):
        XPCA_reduced = self.pca_transform.fit_transform(np.transpose(data))
        return XPCA_reduced.transpose()

    @staticmethod
    def polynomial_smoothing(data, window_size=40, polynomial_order=4):
        if len(data.shape) == 1:
            return np.convolve(data, np.ones(window_size)/window_size, mode='same')
        smoothed_data = []
        for signal_ in data:
            # smoothed_data.append(savgol_filter(signal_, window_size, polynomial_order))
            smoothed_data.append(np.convolve(signal_, np.ones(window_size)/window_size, mode='same'))
        return smoothed_data

    def find_signals_peaks(self, vector, debug_vector=None, ):
        global number
        all_beats = []
        freq_ = []
        per = []
        max_per = -np.inf
        max_per_num = 0
        for i, signal_ in enumerate(vector):
            x = np.arange(0, len(signal_) / self.__new_freq, 1 / self.__new_freq)
            peaks, _ = find_peaks(signal_)



            f = plt.figure(f'{number}_{i}')
            std = np.std(signal_)
            plt.axhline(y=3*std, color='black', linestyle='-', linewidth=2, label='+2std')
            plt.axhline(y=-3 * std, color='black', linestyle='-', linewidth=2, label='-2std')
            heart_beat = len(peaks) / (len(signal_) / self.__new_freq) * 60
            all_beats.append(heart_beat)

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

            max_per_num = None
            if max(signal_) < 3 * std and abs(min(signal_)) < 3 * std:
                if percentage > max_per:
                    max_per = percentage
                    max_per_num = i
            freq_.append(maxFreq)
            per.append(percentage)
            # print(f' maxFreq = {maxFreq}, percentage = {percentage}, bpm = {60*maxFreq}')

            # print("beat", heart_beat)
        # print(f'average = {sum(all_beats) / len(all_beats)}')
        self.used_signal.append(vector[max_per_num] if max_per_num is not None else vector[3])
        idx = np.argmax(per)
        # print(f'bpm = {60*freq_[idx]}')
        number += 1

    def find_most_periodic_signal(self, signals):
        max_periodicity = -1  # Начальное значение максимальной периодичности
        most_periodic_signal = None  # Начальное значение наиболее периодичного сигнала

        for signal in signals:
            spectrum = np.fft.fft(signal)
            amplitudes = np.abs(spectrum)
            max_power_frequency = np.argmax(amplitudes)
            if max_power_frequency * 2 < len(amplitudes):
                first_harmonic_frequency = 2 * max_power_frequency
            else:
                first_harmonic_frequency = max_power_frequency
            total_power = np.sum(amplitudes)
            power_explained = amplitudes[max_power_frequency] + amplitudes[first_harmonic_frequency]
            periodicity = (power_explained / total_power) * 100
            if periodicity > max_periodicity:
                max_periodicity = periodicity
                most_periodic_signal = signal

        return most_periodic_signal

    def plot(self):
        plt.show()

    def create_pulse_wave(self, vector, window, dir, file_name):
        window_begin = 0
        window_end = 250 * window

        wave = list(self.used_signal[0][0:window_end // 2])
        for i in range(len(self.used_signal) - 1):
            begin = self.used_signal[i][window_end // 2:window_end]
            end = self.used_signal[i][0:window_end // 2]
            correlation_coefficient, _ = pearsonr(begin, end)
            alpha = abs(correlation_coefficient)
            for a, b in zip_longest(begin, end):
                if b is not None:
                    wave.append(alpha * b + (1 - alpha) * a)
                else:
                    wave.append(a)

        if len(self.used_signal[-1]) > window_end // 2:
            for i in self.used_signal[-1][window_end // 2:window_end]:
                wave.append(i)
        x = np.arange(0, len(wave) / self.__new_freq, 1 / self.__new_freq)
        wave = self.polynomial_smoothing(np.array(wave))
        f = plt.figure(f'res')
        plt.plot(x, wave)

        # N = len(wave)
        # # sample spacing
        # T = 1.0 / self.__new_freq
        #
        # # Get fft
        # spectrum = np.abs(fft(wave))
        # spectrum *= spectrum
        # xf = fftfreq(N, T)
        #
        # # Get maximum ffts index from second half
        # maxInd = np.argmax(spectrum[:int(len(spectrum) / 2) + 1])
        # # maxInd = np.argmax(spectrum)
        # maxFreqPow = spectrum[maxInd]
        # maxFreq = np.abs(xf[maxInd])
        #
        # total_power = np.sum(spectrum)
        # # Get max frequencies power percentage in total power
        # percentage = maxFreqPow / total_power
        peaks, _ = find_peaks(wave)
        wave = np.array(wave)

        prominences = peak_prominences(wave, peaks)[0]
        contour_heights = wave[peaks] - prominences
        plt.vlines(x=x[peaks], ymin=contour_heights, ymax=wave[peaks], color='red')
        peaks = peaks[prominences > 0.35*np.std(prominences)]
        prominences = peak_prominences(wave, peaks)[0]
        contour_heights = wave[peaks] - prominences

        plt.plot(x[peaks], wave[peaks], "o", color='orange', label='peaks')
        if not os.path.exists(dir):
            os.mkdir(dir)
        with open(f'{dir}/{file_name}.npy', 'wb') as f:
            np.save(f, x[peaks])

        plt.vlines(x=x[peaks], ymin=contour_heights, ymax=wave[peaks], color='orange')
        heart_beat = len(peaks) / (len(wave) / self.__new_freq) * 60
        # print(f'bpm_wave = {60*maxFreq}')
        # print(f'test = {250*maxFreq}')


    def correlation_after_pca(self, vector, dir, file_name):
        rate = 250
        pipeline(vector, self.__new_freq)
        for i in range(len(vector)):
            wave = vector[i]
            x = np.arange(0, len(wave) / self.__new_freq, 1 / self.__new_freq)


            N = rate*10-1
            yf = rfft(wave)
            xf = rfftfreq(N, 1/rate)

            wave = self.polynomial_smoothing(np.array(wave))
            peaks, _ = find_peaks(wave)
            wave = np.array(wave)

            prominences = peak_prominences(wave, peaks)[0]
            contour_heights = wave[peaks] - prominences

            peaks = peaks[prominences > 0.35 * np.std(prominences)]
            # print('peaks', len(peaks))




