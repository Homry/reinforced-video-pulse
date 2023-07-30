import numpy as np
import scipy.interpolate as interp
from collections import Counter

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
        old_time_step = 1.0 / self.__old_freq
        new_time_step = 1.0 / self.__new_freq

        old_time_points = np.arange(0, len(self.__vector[0])) * old_time_step
        new_time_points = np.arange(0, len(self.__vector[0]) - 1, old_time_step / new_time_step)

        vector = []
        for i in self.__vector:
            cs = interp.CubicSpline(old_time_points, i)
            vector.append(cs(new_time_points))
        self.__vector = vector

    def calculate_mode(self, data):

        mode = Counter(data).most_common(1)[0][0]
        return mode

    def distance_filter(self):
        vector = []
        for i in self.__vector:
            distance = [abs(i[j]-i[j+1]) for j in range(len(i)-1)]
            mode = self.calculate_mode(distance)
            print(mode)
            status = [True if j <= mode else False for j in distance]
            if False not in status:
                vector.append(i)
        self.__vector = vector



