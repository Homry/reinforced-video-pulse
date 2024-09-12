import numpy as np
import torch
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, LinearLR
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from model import ResNet, BasicBlock
from dataset import PulseDataset
from regress_model import Regress

import matplotlib.pyplot as plt

class ModelTrainer:
    def __init__(self, model_file, num_epochs=10):
        print(torch.cuda.is_available())
        self.model_file = model_file
        self.loss_history = []
        self.val_loss_history = []
        self.num_epochs = num_epochs
        self.model = Regress()
        # self.model.apply(self.weights_init)


    def train_model(self):
        # Load the data
        dataset = PulseDataset()  # Replace YourDataset with your own dataset class
        # Create the data loaders
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size

        # Разделите датасет на тренировочный и тестовый
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        # Определите размеры тренировочного и валидационного датасетов
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size

        # Разделите тренировочный датасет на тренировочный и валидационный
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

        # Создайте загрузчики данных (DataLoaders)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

        # Define the loss function and optimizer
        criterion = nn.MSELoss()
        # optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        # scheduler = ExponentialLR(optimizer, gamma=0.1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        # Train the model
        for epoch in range(self.num_epochs):
            epoch_loss = []
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.reshape([labels.shape[0], 1])
                labels = labels.to(device)
                optimizer.zero_grad()
                # Forward pass
                inputs = inputs.float()  # Convert inputs to float type

                labels = labels.float()  # Convert outputs to float type
                outputs = self.model(inputs)
                # print(outputs)
                loss = criterion(outputs, labels)
                # print(loss)
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
            #
            print(f' {epoch = }, {np.array(epoch_loss).mean() = }, lr = {optimizer.param_groups[0]["lr"]}')

            self.loss_history.append(np.array(epoch_loss).mean())
            val = self.validate_model(model=self.model, criterion=criterion, val_loader=val_loader, device=device)
            print(f'Validation Loss: {np.array(val).mean()}')
            self.val_loss_history.append(np.array(val).mean())
            # Validation
            # val_loss = self.validate_model(self.model, criterion, val_loader, device)
            # print(f'Validation Loss: {np.array(val_loss).mean()}')
            # self.val_loss_history.append(np.array(val_loss).mean())
        # Save the trained model
        torch.save(self.model.state_dict(), self.model_file)
        val = self.validate_model(model=self.model, criterion=criterion, val_loader=test_loader, device=device)
        print(f'test Loss: {np.array(val).mean()}')


    def validate_model(self, model, criterion, val_loader, device):
        model.eval()
        val_loss = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.reshape([labels.shape[0], 1])
                labels = labels.to(device)

                # Forward pass
                inputs = inputs.float()  # Convert inputs to float type
                labels = labels.float()  # Convert outputs to float type
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss.append(loss.item())

        model.train()  # Set the model back to training mode

        return val_loss

    def save(self):
        torch.save(self.model.state_dict(), self.model_file)

    def plot_loss(self):
        plt.plot(self.loss_history, label='Training Loss')
        # plt.plot(self.val_loss_history, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig('loss_graph_test.png')
        plt.show()

# Usage example
trainer = ModelTrainer('regress.pth', num_epochs=5)
try:
    trainer.train_model()
except KeyboardInterrupt:
    trainer.save()

trainer.plot_loss()
