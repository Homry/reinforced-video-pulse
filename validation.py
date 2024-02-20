import numpy as np
import src.metrics as metrics
import os

import pandas as pd

base_path = './npy_data'


name_ = []
mean_ = []
var_ = []
std_ = []
mse_ = []
rmse_ = []

df = pd.DataFrame()
for video_file_name in os.listdir(base_path):
    for time_range in os.listdir(f'{base_path}/{video_file_name}'):
        files = os.listdir(f'{base_path}/{video_file_name}/{time_range}')
        print(f'{base_path}/{video_file_name}/{time_range}')



        with open(f'{base_path}/{video_file_name}/{time_range}/{files[0]}', 'rb') as f:
            gt = np.load(f)

        with open(f'{base_path}/{video_file_name}/{time_range}/{files[2]}', 'rb') as f:
            media = np.load(f)

        with open(f'{base_path}/{video_file_name}/{time_range}/{files[1]}', 'rb') as f:
            lk = np.load(f)

        name_.append(f'{base_path}/{video_file_name}/{time_range}/{files[0]}')
        name_.append(f'{base_path}/{video_file_name}/{time_range}/{files[1]}')
        name_.append(f'{base_path}/{video_file_name}/{time_range}/{files[2]}')


        print('ground truth')
        mean = metrics.mean(gt)
        var = metrics.var(gt)
        std = metrics.std(gt)
        mean_.append(mean)
        var_.append(var)
        std_.append(std)

        print(
            f'mean = {int(mean)},{str(str(mean).split(".")[1])}, var = {int(var)},{str(str(var).split(".")[1])}, std = {int(std)},{str(str(std).split(".")[1])}')
        print()


        print('lk')
        mean = metrics.mean(lk)
        var = metrics.var(lk)
        std = metrics.std(lk)
        mean_.append(mean)
        var_.append(var)
        std_.append(std)
        print(f'mean = {int(mean)},{str(str(mean).split(".")[1])}, var = {int(var)},{str(str(var).split(".")[1])}, std = {int(std)},{str(str(std).split(".")[1])}')
        print()
        print('media')
        mean = metrics.mean(media)
        var = metrics.var(media)
        std = metrics.std(media)
        mean_.append(mean)
        var_.append(var)
        std_.append(std)
        print(
            f'mean = {int(mean)},{str(str(mean).split(".")[1])}, var = {int(var)},{str(str(var).split(".")[1])}, std = {int(std)},{str(str(std).split(".")[1])}')
        print()


        mse_.append(None)
        rmse_.append(None)


        print('hear_beat')
        print(f'gt = {len(gt)}, media = {len(media)}, lk = {len(lk)}')





        min_len = min(len(lk), len(gt))
        validation_orig = lk[0:min_len]
        validation_gt = gt[0:min_len]

        print()
        print('metrics lk')
        mse = metrics.mse(validation_orig, validation_gt)
        rmse = metrics.rmse(validation_orig, validation_gt)
        mse_.append(mse)
        rmse_.append(rmse)
        print(f'mse = {int(mse)},{str(str(mse).split(".")[1])}, rmse = {int(rmse)},{str(str(rmse).split(".")[1])}')

        min_len = min(len(media), len(gt))
        validation_orig = media[0:min_len]
        validation_gt = gt[0:min_len]
        print()
        print('metrics media')
        mse = metrics.mse(validation_orig, validation_gt)
        rmse = metrics.rmse(validation_orig, validation_gt)
        mse_.append(mse)
        rmse_.append(rmse)
        print(f'mse = {int(mse)},{str(str(mse).split(".")[1])}, rmse = {int(rmse)},{str(str(rmse).split(".")[1])}')


df['name'] = name_
df['mean'] = mean_
df['var'] = var_
df['std'] = std_
df['mse'] = mse_
df['rmse'] = rmse_
df.to_csv('data.csv')
print(df)