from catboost import CatBoostClassifier, Pool, sum_models
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pandas
from scipy.interpolate import interp1d
import torch
from tqdm import tqdm
import sklearn
from sklearn.metrics import recall_score, f1_score, precision_score


def load(name):
    with open(name, 'rb') as f:
        return np.load(f)


def interpolate_data(data, begin_index=0, end_index=0, new_length=1300):
    old_indices = np.linspace(0, len(data) - 1, num=len(data))
    new_indices = np.linspace(0, len(data) - 1, num=new_length)
    interpolator = interp1d(old_indices, data)
    new_data = interpolator(new_indices)
    fft = np.fft.fft(new_data).flatten()
    return new_data[begin_index:end_index], fft[begin_index:end_index].real


class BoostDatasetWindow(Dataset):
    def __init__(self):
        self.data = pandas.read_csv('../../peaks_dataset_new.csv').to_numpy()
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = load(self.data[idx][0])
        return data, int(self.data[idx][1])


class BoostDatasetFull(Dataset):
    def __init__(self):
        self.data = pandas.read_csv('../../peaks_dataset_full.csv').to_numpy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = load(self.data[idx][0])
        Y = load(self.data[idx][1])
        X = []

        for i in data:
            fft = np.fft.fft(i).flatten().real
            X.append(np.concatenate((i, fft)))

        return np.array(X), Y.astype(int)


def validate():
    for x, y in val_loader:
        x = x.numpy().reshape(x.shape[0], -1)
        y = y.numpy()
        return x, y


if __name__ == "__main__":
    dataset = BoostDatasetWindow()
    train_size = int(0.9 * len(dataset))
    val_size = int(0.01 * len(dataset))
    print(train_size)
    print(val_size)
    test_size = len(dataset) - train_size - val_size

    train_data, val_data, test_data = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=200000, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=128, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=512, shuffle=False)

    epoch = 5
    models = []
    params = {
        'task_type': 'CPU',
        'iterations': 1000,
        'learning_rate': 0.1,
        'depth': 10,
        'eval_metric': 'F1',

    }
    model = CatBoostClassifier(**params)

    i = 0
    for x, y in tqdm(train_loader):
        x = x.numpy().reshape(x.shape[0], -1)
        y = y.numpy()
        # x_val, y_val = validate()
        # val = Pool(data=x_val, label=y_val)
        train_dataset = Pool(data=x, label=y)
        if i == 0:
            # model.fit(train_dataset, verbose=1, eval_set=val)
            model.fit(train_dataset, verbose=1)
        else:
            # model.fit(train_dataset, verbose=1, eval_set=val, init_model='model.cbm')
            model.fit(train_dataset, verbose=1, init_model='model.cbm')
        i += 1
        model.save_model('model.cbm')
        break

    f1, rec, pres = [], [], []

    for x, y in val_loader:
        x = x.numpy().reshape(x.shape[0], -1)
        y = y.numpy()
        y_pred = model.predict(Pool(data=x))
        metric = model.eval_metrics(Pool(x, y), ['F1'])
        print(np.mean(metric['F1']))
        rec.append(recall_score(y_pred, y))
        pres.append(precision_score(y_pred, y))
        f1.append(f1_score(y_pred, y))

    print(f'f1 = {np.mean(f1)}, precision = {np.mean(pres)}, recall = {np.mean(rec)}')









