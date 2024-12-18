from torch import nn
import torch.nn.functional as F


class PeaksModel(nn.Module):

    def __init__(self, num_channels=5, num_features=198):
        super(PeaksModel, self).__init__()
        self.conv1 = nn.Conv1d(num_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(64 * num_features, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        print(x.shape)
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        print(x.shape)
        x = x.view(x.size(0), -1)
        print(x.shape)
        x = self.sigmoid(self.fc(x))
        return x

