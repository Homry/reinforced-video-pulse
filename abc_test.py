'''import random
from itertools import zip_longest
array = [random.randint(0, 10000) for i in range(31*250)]

window = 10

window_begin = 0
window_end = 250 * window
window_offset = (250 // 2) * window
end_of_data = len(array)
result = []
print(len(array))
no_else = False
while window_end <= end_of_data:
    if window_end == end_of_data:
        no_else = True
    result.append(array[window_begin:window_end])
    window_begin += window_offset
    window_end += window_offset
    print(f'win_beg {window_begin}, win end = {window_end}')
else:
    if window_begin < end_of_data and not no_else:
        result.append(array[window_begin:end_of_data])


window = 10

window_begin = 0
window_end = 250 * window

wave = result[0][0:window_end//2]
for i in range(len(result)-1):
    begin = result[i][window_end//2:window_end]
    end = result[i][0:window_end//2]
    for a, b in zip_longest(begin, end):
        if b is not None:
            wave.append((a+b)/2)
        else:
            wave.append(a)

if len(result[-1]) > window_end//2:
    for i in result[-1][window_end//2:window_end]:
        wave.append(i)
print(f'wave {len(wave)}')'''
from scipy.signal import find_peaks, savgol_filter

'''import numpy as np
from scipy.stats import pearsonr
from matplotlib import pyplot as plt


def combine_functions_with_correlation(func1, func2, data):
    result1 = func1(np.linspace(0, 2 * np.pi, 100))
    result2 = func2(np.linspace(0, 3 * np.pi, 100))
    correlation_coefficient, _ = pearsonr(result2, result1)
    print(correlation_coefficient)

    # Используем абсолютное значение коэффициента корреляции в качестве веса
    alpha = abs(correlation_coefficient)

    # Комбинируем результаты с весом alpha и (1 - alpha)
    combined_result = alpha * result1 + (1 - alpha) * result2
    return combined_result


# Пример использования:
def function1(x):
    return np.sin(x)


def function2(x):
    return np.cos(x)


result = combine_functions_with_correlation(function1, function1, np.linspace(0, 2 * np.pi, 100))
plt.plot(function1(np.linspace(0, 2 * np.pi, 100)), label='sin', color='red')
plt.plot(function2(np.linspace(0, 3 * np.pi, 100)), label='data', color='blue')
plt.plot(result, label='res', color='green')
plt.legend()
plt.show()'''

import pandas
data = pandas.read_csv('./pulsar_data_1663827841.1197596.csv')
print(data.values[0][1])
print(type(data.values[0][1]))
wave = []
for i in data.values:
    tmp = list(map(lambda x: int(x), i[1][1:-1].split(', ')))
    for i in tmp:
        wave.append(i)

from matplotlib import pyplot as plt
import numpy as np
from scipy.fftpack import fft, ifft, fftfreq, fftshift
wave = wave[25*60*130:(25*60+30)*130]
x = np.arange(0, len(wave) / 130, 1 / 130)
plt.plot(x, wave)
peaks, _ = find_peaks(wave)
heart_beat = len(peaks) / (len(wave) / 130) * 60
wave = np.array(wave)
# plt.plot(x[peaks], wave[peaks], "o", color='orange', label='peaks')

N = len(wave)
# sample spacing
T = 1.0 /230

# Get fft
spectrum = np.abs(fft(wave))
spectrum *= spectrum
xf = fftfreq(N, T)

# Get maximum ffts index from second half
maxInd = np.argmax(spectrum[:int(len(spectrum) / 2) + 1])
# maxInd = np.argmax(spectrum)
maxFreqPow = spectrum[maxInd]
maxFreq = np.abs(xf[maxInd])

total_power = np.sum(spectrum)
# Get max frequencies power percentage in total power
percentage = maxFreqPow / total_power
print(f'bpm_wave = {60 * maxFreq}')

print(heart_beat)
print(f'test = {130*maxFreq}')
plt.show()