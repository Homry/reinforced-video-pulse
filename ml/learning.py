import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from model import ResNet
from dataset import PulseDataset

import matplotlib.pyplot as plt

class ModelTrainer:
    def __init__(self, model_file, num_epochs=10):
        self.model_file = model_file
        self.loss_history = []
        self.num_epochs = num_epochs

    def train_model(self):
        # Load the data
        train_dataset = PulseDataset()  # Replace YourDataset with your own dataset class
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        # Create the model
        model = ResNet()

        # Define the loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            self.loss_history.append(epoch_loss)

        # Save the trained model
        torch.save(model.state_dict(), self.model_file)

    def load_model(self):
        # Create an instance of the model
        model = ResNet()

        # Load the trained weights
        model.load_state_dict(torch.load(self.model_file))

        return model

    def plot_loss(self):
        plt.plot(self.loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig('loss_graph.png')
        plt.show()

# Usage example
trainer = ModelTrainer('model.pth')
trainer.train_model()
trainer.plot_loss()
