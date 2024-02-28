import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader

from classifire import Classifire
from ml.dataset import ClassifireDataset


print('here')

model = Classifire()
model.eval()

# Load the trained weights
model.load_state_dict(torch.load('./classifire.pth'))



model.to('cuda')
dataset = ClassifireDataset()  # Replace YourDataset with your own dataset class
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
pres = []
rec = []
f1 = []
print('here')
for inputs, labels in train_loader:
    inputs = inputs.to('cuda')
    labels = labels.to('cuda')
    inputs = inputs.float()  # Convert inputs to float type
    outputs = model(inputs)
    labels = labels.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()
    pred = [[0 for i in range(5)] for j in range(64)]
    for i, val in enumerate(np.argmax(outputs, axis=1)):
        pred[i][val] = 1
    try:
        pres.append(precision_score(labels, pred, average=None).mean())
        rec.append(recall_score(labels, pred, average=None).mean())
        f1.append(f1_score(labels, pred, average=None).mean())
    except:
        pass


print("pres", np.array(pres).mean())
print("rec", np.array(rec).mean())
print("f1", np.array(f1).mean())
