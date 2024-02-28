import numpy as np
import torch
import torch.nn as nn


class Regress(nn.Module):
    def __init__(self):
        super(Regress, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(5, 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Conv1d(4, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(3),
            nn.ReLU(),
            nn.Conv1d(3, 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(2),
            nn.ReLU(),
            nn.Conv1d(2, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(1),
            nn.ReLU(),

            # nn.Linear(250, 100)
        )

        self.linear = nn.Sequential(
            nn.Linear(2499, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
        )

    def forward(self, x):
        return self.linear(self.conv(x).view(-1, 2499))


if __name__ == "__main__":
    model = Regress().double()
    data = torch.tensor([np.zeros([5, 2499], dtype=np.double), np.zeros([5, 2499], dtype=np.double)])
    x = model(data)
    print(x.shape)
    print(x)