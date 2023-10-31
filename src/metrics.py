from math import sqrt
from numpy import std, var, mean


def mse(predict, ground_truth):
    if len(predict) != len(ground_truth):
        raise ValueError('Shape not the same')
    return sum([(ground_truth[i]-predict[i])**2 for i in range(len(predict))])/len(predict)


def rmse(predict, ground_truth):
    return sqrt(mse(predict, ground_truth))


def distance(data):
    return [abs(data[i]-data[i+1]) for i in range(len(data)-1)]


def distance_std(data):
    return std(distance(data))


def distance_var(data):
    return var(distance(data))


def distance_mean(data):
    return mean(distance(data))
