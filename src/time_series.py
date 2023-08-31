import numpy as np
import scipy.interpolate as interp
from scipy import signal
from collections import Counter
from matplotlib import pyplot as plt

class TimeSeries:
    def __init__(self):
        self.__vector = None
        self.__old_freq = 30
        self.__new_freq = 250

    def init_vector(self, vector):
        self.__vector = [[i[1]] for i in vector]

    def add_in_vector(self, vector):
        [self.__vector[i].append(vector[i][1]) for i in range(len(vector))]

    def __str__(self):
        return f'{self.__vector}, {len(self.__vector[0])}'

    def interpolate_signal(self):


        x = np.arange(0, len(self.__vector[0])/self.__old_freq, 1/self.__old_freq)
        x_int = np.linspace(np.min(x), np.max(x), int(len(x)*8.333))

        vector = []
        for i in self.__vector:
            cs = interp.CubicSpline(x, i)
            vector.append(cs(x_int))
            print(f'before = {len(i)} after =  {len(vector[-1])}')
        self.__vector = vector
        # for i, j in enumerate(self.__vector):
        #     x = np.arange(0, len(j) / self.__new_freq, 1 / self.__new_freq)
        #     plt.plot(x, j)
        #     plt.show()

    def calculate_mode(self, data):

        mode = Counter(data).most_common(1)[0][0]
        return mode

    def distance_filter(self):
        vector = []
        for i in self.__vector:
            distance = [abs(i[j]-i[j+1]) for j in range(len(i)-1)]
            mode = self.calculate_mode(distance)
            status = [True if j <= mode else False for j in distance]
            if False not in status:
                vector.append(i)
        self.__vector = vector
        for i, j in enumerate(self.__vector):
            x = np.arange(0, len(j) / self.__new_freq, 1 / self.__new_freq)
            plt.plot(x, j)
            plt.show()


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
        for i, j in enumerate(self.__vector):
            x = np.arange(0, len(j) / self.__new_freq, 1 / self.__new_freq)
            plt.plot(x, j)
            plt.show()



