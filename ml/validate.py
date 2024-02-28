from matplotlib import pyplot as plt
from model import ResNet, BasicBlock
from autoencode import Autoencoder
import numpy as np
from torch import Tensor
import torch


def mse(predict, ground_truth):
    if len(predict) != len(ground_truth):
        raise ValueError('Shape not the same')
    return sum([(ground_truth[i]-predict[i])**2 for i in range(len(predict))])/len(predict)


# model = ResNet(BasicBlock, [2,2,2,2])
model = Autoencoder()
model.to('cuda')

model.load_state_dict(torch.load('model_auto_test.pth'))

with open('../dataset_npy/value/2022-09-10 09-22-51/2022-09-10 09-22-51_1040-1050.npy', 'rb') as f:
    val = np.load(f)

with open('../dataset_npy/gt/2022-09-10 09-22-51/2022-09-10 09-22-51_1040-1050.npy', 'rb') as f:
    gt = np.load(f)

with open('../dataset_npy/transform/2022-09-10 09-22-51/2022-09-10 09-22-51_1040-1050.npy', 'rb') as f:
    transform = np.load(f)


val = Tensor([val]).to('cuda')
model.eval()
res = model(val)
print(res)
res = res.to('cpu').detach().numpy()
print(res)
f = plt.figure(f'res')

plt.plot(res[0])

f = plt.figure(f'gt')

plt.plot(gt)

plt.figure('transform')
transform = transform*100
plt.plot(transform)

print(res.shape)
print(transform.shape)

print(mse(res[0], transform))

plt.show()
