import numpy as np
import pandas as pd


from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, f1_score, precision_score, recall_score

import torch
from torch import nn
from torch.nn import functional as F





from sklearn.base import BaseEstimator

out_dict = dict()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

from ml.dataset.dataset import BoostDatasetWindow


from tqdm import tqdm
from ml.peaks.model import PeaksModel

dataset = BoostDatasetWindow()
train_size = int(0.7 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_data, val_data, test_data = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

model = PeaksModel()
model = model.to(device)

loss_function = nn.CrossEntropyLoss ()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_loss_history = []
train_f1_history = []
train_rec_history = []
train_pres_history = []
val_loss_history = []
val_f1_history = []
val_rec_history = []
val_pres_history = []

local_train_loss_history = []
local_train_f1_history = []
local_train_rec_history = []
local_train_pres_history = []
eval_every = 50
i = 0
for x, y in train_loader:
    x = x.to(device).float()
    y = y.to(device).long()
    optimizer.zero_grad()
    pred = model(x)
    loss = loss_function(pred, y)
    loss.backward()
    optimizer.step()
    local_train_loss_history.append(loss.item())
    local_train_f1_history.append(f1_score(y.cpu().numpy(), pred.argmax(dim=1).cpu().numpy()))
    local_train_rec_history.append(recall_score(y.cpu().numpy(), pred.argmax(dim=1).cpu().numpy()))
    local_train_pres_history.append(precision_score(y.cpu().numpy(), pred.argmax(dim=1).cpu().numpy()))

    if i % eval_every == 0:
        train_loss_history.append(np.mean(local_train_loss_history))
        train_f1_history.append(np.mean(local_train_f1_history))
        train_rec_history.append(np.mean(local_train_rec_history))
        train_pres_history.append(np.mean(local_train_pres_history))

        local_train_loss_history = []
        local_train_f1_history = []
        local_train_rec_history = []
        local_train_pres_history = []

        with torch.no_grad():
            model.eval()
            local_val_loss_history = []
            local_val_f1_history = []
            local_val_rec_history = []
            local_val_pres_history = []
            k = 0
            for x, y in val_loader:
                x = x.to(device).float()
                y = y.to(device).long()
                pred = model(x)
                loss = loss_function(pred, y)
                local_val_loss_history.append(loss.item())
                local_val_f1_history.append(f1_score(y.cpu().numpy(), pred.argmax(dim=1).cpu().numpy()))
                local_val_rec_history.append(recall_score(y.cpu().numpy(), pred.argmax(dim=1).cpu().numpy()))
                local_val_pres_history.append(precision_score(y.cpu().numpy(), pred.argmax(dim=1).cpu().numpy()))
                k += 1
                if k == 10:
                    break

            val_loss_history.append(np.mean(local_val_loss_history))
            val_f1_history.append(np.mean(local_val_f1_history))
            val_rec_history.append(np.mean(local_val_rec_history))
            val_pres_history.append(np.mean(local_val_pres_history))
            # display.clear_output(wait=True)
            # plot_train_process(train_loss_history, val_loss_history, train_f1_history, val_f1_history,
            #                    train_rec_history, val_rec_history, train_pres_history, val_pres_history)
    i += 1