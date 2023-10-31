import numpy as np
import src.metrics as metrics

with open('./npy_data/gt.npy', 'rb') as f:
    gt = np.load(f)

with open('./npy_data/orig_lk.npy', 'rb') as f:
    orig = np.load(f)

print('orig')
print(f'mean = {metrics.mean(orig)}, var = {metrics.var(orig)}, std = {metrics.std(orig)}')
print()
print('ground truth')
print(f'mean = {metrics.mean(gt)}, var = {metrics.var(gt)}, std = {metrics.std(gt)}')
print()
print('hear_beat')
print(f'gt = {len(gt)}, orig = {len(orig)}')

min_len = min(len(orig), len(gt))
validation_orig = orig[0:min_len]
validation_gt = gt[0:min_len]

print()
print('metrics')
print(f'mse = {metrics.mse(validation_orig, validation_gt)}, rmse = {metrics.rmse(validation_orig, validation_gt)}')

