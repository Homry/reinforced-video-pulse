import matplotlib.pyplot as plt
import numpy as np

with open('./dataset_npy/value/2022-09-13 09-21-29/2022-09-13 09-21-29_10015-10025.npy', 'rb') as f:
    data = np.load(f)

for j, i in enumerate(data):
    f = plt.figure(j)
    plt.plot(i)
plt.show()