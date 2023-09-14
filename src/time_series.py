import numpy as np
import scipy.interpolate as interp
from scipy import signal
import statistics
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from scipy.signal import find_peaks
from scipy.fftpack import fft, ifft, fftfreq, fftshift


class TimeSeries:
    def __init__(self):
        self.__vector = None
        self.__old_freq = 30
        self.__new_freq = 250
        self.pca_transform = PCA(n_components=5)

    def init_vector(self, vector):
        self.__vector = [[i[1]] for i in vector]

    def add_in_vector(self, vector, status):
        # [self.__vector[i].append(vector[i][0][1]) for i, j in enumerate(status) if j == 1]
        [self.__vector[i].append(vector[i][1]) for i, j in enumerate(status)]

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
        # for i, j in enumerate(self.__vector):
        #     x = np.arange(0, len(j) / self.__new_freq, 1 / self.__new_freq)

    def pca(self):



        # vector = np.array(self.__vector.copy())
        #
        # shape = vector.shape
        # FF = 1/shape[1]*vector@vector.T
        #
        # covmat = np.cov(vector)
        # _, vecs = np.linalg.eig(covmat)
        # self.my_pca1 = np.array([np.dot(vecs[i], vector) for i in range(5)])
        #
        # _, vecs = np.linalg.eig(FF)
        # # Xnew = np.dot(v, vector)
        # print(vecs.shape)
        # v = -vecs[:, 1]
        # print(v.shape)
        # self.my_pca = np.array([np.dot(vecs[i], vector) for i in range(5)])
        # print(f'my {self.my_pca.shape}')
        XPCAreduced = self.pca_transform.fit_transform(np.transpose(self.__vector))
        self.__vector = XPCAreduced.transpose()
       


        # vector = []
        #
        # for i in self.__vector:
        #     x = np.arange(0, len(i) / self.__new_freq, 1 / self.__new_freq)
        #     # print(f'len - x = {len(x)}')
        #     # print(f'len - y = {len(i)}')
        #     X = np.vstack((x, i))
        #     XPCAreduced = self.pca_transform.fit_transform(np.transpose(X))
        #     points_after_pca = [j[0] for j in XPCAreduced]
        #     # print(len(points_after_pca))
        #     # plt.plot(x, points_after_pca)
        #     # plt.show()

    def find_signals_peaks(self):
        all_beats = []
        freq_ = []
        per = []
        for i, signal in enumerate(self.__vector):
            x = np.arange(0, len(signal) / self.__new_freq, 1 / self.__new_freq)
            peaks, _ = find_peaks(signal)
            f = plt.figure(i)
            # plt.plot(x, signal, label=f'{i}', color='red')
            plt.plot(x,signal)
            # plt.plot(x[peaks], signal[peaks], "o", color='red')
            # plt.plot(x[peaks], signal[peaks], "o", color='red')

            heart_beat = len(peaks) / (len(signal) / self.__new_freq) * 60
            all_beats.append(heart_beat)

            N = len(signal)
            # sample spacing
            T = 1.0 / 250

            # Get fft
            spectrum = np.abs(fft(signal))
            spectrum *= spectrum
            xf = fftfreq(N, T)

            # Get maximum ffts index from second half
            # maxInd = np.argmax(spectrum[:int(len(spectrum)/2)+1])
            maxInd = np.argmax(spectrum)
            maxFreqPow = spectrum[maxInd]
            maxFreq = np.abs(xf[maxInd])

            total_power = np.sum(spectrum)
            # Get max frequencies power percentage in total power
            percentage = maxFreqPow / total_power
            freq_.append(maxFreq)
            per.append(percentage)
            print(f' maxFreq = {maxFreq}, percentage = {percentage}, bpm = {60/maxFreq}')

            print("beat", heart_beat)
        print(f'average = {sum(all_beats) / len(all_beats)}')
        idx = np.argmax(per)
        print(f'bpm = {60/freq_[idx]}')
        plt.show()

        # for i, signal in enumerate(self.my_pca):
        #     x = np.arange(0, len(signal) / self.__new_freq, 1 / self.__new_freq)
        #     peaks, _ = find_peaks(signal)
        #     f = plt.figure(i)
        #     plt.plot(x, signal, label=f'{i}')
        #
        #     plt.plot(x[peaks], signal[peaks], "o", color='red')
        #
        #
        #     heart_beat = len(peaks) / (len(signal) / self.__new_freq) * 60
        #     all_beats.append(heart_beat)
        #     print("beat", heart_beat)
        # print(f'average = {sum(all_beats) / len(all_beats)}')
        # plt.show()


