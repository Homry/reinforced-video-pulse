import numpy as np
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(5 * 2499, 10000),
            nn.ReLU(),
            nn.Linear(10000, 5000),
            nn.ReLU(),
            nn.Linear(5000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, 100)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(100, 250),
            nn.ReLU(),
            nn.Linear(250, 500),
            nn.ReLU(),
            nn.Linear(500, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1300)
        )
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.view(x.size(0), 1300)

# Create an instance of the Autoencoder model



class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        self.encoder_conv = nn.Sequential(
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

        self.encoder_linear = nn.Sequential(
            nn.Linear(2499, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, 100),
        )


        self.decoder = nn.Sequential(
            nn.Linear(100, 250),
            nn.ReLU(),
            nn.Linear(250, 500),
            nn.ReLU(),
            nn.Linear(500, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1300)
        )

    def forward(self, x):

        encoded = self.encoder_linear(self.encoder_conv(x).view(-1, 2499))
        decoded = self.decoder(encoded)
        return decoded.view(x.size(0), 1300)



if __name__ == "__main__":
    model = ConvAutoencoder().double()
    data = torch.tensor([np.zeros([5, 2499], dtype=np.double)])
    x = model(data)
    print(x.shape)