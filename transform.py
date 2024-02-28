import os.path

import pandas as pd
import numpy as np

df = pd.read_csv('./classifire_dataset.csv')
base_line = '/home/sergey/reinforced-video-pulse/dataset_npy/classifire'
data = df.to_numpy()
res = []
for line in data:
    split = line[0].split('/')
    dir_name = split[-2]
    file_name = split[-1].split('.')[0]
    array = [int(i) for i in line[1][1:-1].split(', ')]

    print(array, dir_name, file_name)
    with open(line[0], 'rb') as f:
        waves = np.load(f)
    for i in range(len(array)):
        wave = waves[i]
        markup = [1, 0] if array[i] == 1 else [0, 1]
        print(wave.shape, markup)
        if not os.path.exists(f'{base_line}/{dir_name}'):
            os.mkdir(f'{base_line}/{dir_name}')
        with open(f'{base_line}/{dir_name}/{file_name}_{i}.npy', 'wb') as f:
            np.save(f, wave)
        res.append([f'{base_line}/{dir_name}/{file_name}_{i}.npy', markup])

df = pd.DataFrame(res)
df.to_csv('./new_classifire.csv', index=False)

