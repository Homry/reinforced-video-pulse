import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from dataset import ClassifireDataset, SelectionDataset


class Classifire(nn.Module):
    def __init__(self):
        super(Classifire, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(2499, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1500),
            nn.ReLU(),
            nn.Linear(1500, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, 1)
        )

    def forward(self, x):
        x = self.linear(x)
        return torch.sigmoid(x)




class ModelTrainer:
    def __init__(self, model_file, num_epochs=10):
        print(torch.cuda.is_available())
        self.model_file = model_file
        self.loss_history = []
        self.val_loss_history = []
        self.num_epochs = num_epochs
        self.model = Classifire()
        # self.model.apply(self.weights_init)


    def train_model(self):
        # Load the data
        dataset = ClassifireDataset()  # Replace YourDataset with your own dataset class



        # Create the data loaders
        train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

        # Create the model
        # model =


        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # Train the model
        for epoch in range(self.num_epochs):
            epoch_loss = []
            running_loss = 0.0
            i = 0
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                # Forward pass
                inputs = inputs.float()  # Convert inputs to float type

                labels = labels.float()  # Convert outputs to float type
                outputs = self.model(inputs)
                # print(outputs)
                # print(f'{labels = }, {outputs = }')
                loss = criterion(outputs, labels)
                # print(loss)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                epoch_loss.append(loss.item())
                if i % 2658 == 2657:  # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2658:.3f}')
                    running_loss = 0.0
                i += 1
            #
            print(f' {epoch = }, {np.array(epoch_loss).mean() = }, lr = {optimizer.param_groups[0]["lr"]}')

            self.loss_history.append(np.array(epoch_loss).mean())

            # Validation
            # val_loss = self.validate_model(self.model, criterion, val_loader, device)
            # print(f'Validation Loss: {np.array(val_loss).mean()}')
            # self.val_loss_history.append(np.array(val_loss).mean())
        # Save the trained model
        torch.save(self.model.state_dict(), self.model_file)

    def save(self):
        torch.save(self.model.state_dict(), self.model_file)

    def validate_model(self, model, criterion, val_loader, device):
        model.eval()  # Set the model to evaluation mode
        val_loss = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)

                labels = labels.to(device)

                # Forward pass
                inputs = inputs.float()  # Convert inputs to float type
                labels = labels.float()  # Convert outputs to float type
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss.append(loss.item())

        model.train()  # Set the model back to training mode

        return val_loss

    def load_model(self):
        # Create an instance of the model
        model = Classifire()

        # Load the trained weights
        model.load_state_dict(torch.load(self.model_file))

        # Send the model to CUDA if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        return model

    def plot_loss(self):
        plt.plot(self.loss_history, label='Training Loss')
        # plt.plot(self.val_loss_history, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig('classifire.png')
        plt.show()


class SelectModel(nn.Module):
    def __init__(self):
        super(SelectModel, self).__init__()
        self.conv1 = nn.Conv1d(5, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64*2499, 500)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(500, 5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)


class Teacher:
    def __init__(self, model, num_epochs=10):
        self.model = model()
        self.num_epochs = num_epochs
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)


    def train(self):
        self.model.train()
        epoch_loss = []
        dataset = SelectionDataset()

        # Calculate the sizes of each part
        train_size = int(0.7 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        # Split the dataset
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
        for epoch in range(self.num_epochs):
            for inputs, labels in self.train_loader:
                inputs = inputs.float().to(self.device)
                labels = labels.long().to(self.device)
                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                loss.backward()

                self.optimizer.step()
                epoch_loss.append(loss.item())

            print(f'Epoch: {epoch+1}, Loss: {np.mean(epoch_loss)}')
            self.validate_model('val')
        self.validate_model('test')
        self.save_model('selection.pth')

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def validate_model(self, mode='val'):
        self.model.eval()  # Set the model to evaluation mode
        val_loss = []
        all_outputs = []
        all_labels = []
        if mode == 'val':
            loader = self.val_loader
        else:
            loader = self.test_loader
        with torch.no_grad():
            for inputs, labels in  loader:
                inputs = inputs.float().to(self.device)
                labels = labels.long().to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                val_loss.append(loss.item())

                # Convert outputs and labels to numpy arrays and add them to the lists

                all_outputs.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        f1 = f1_score(all_labels, all_outputs, average='weighted')
        precision = precision_score(all_labels, all_outputs, average='weighted')
        recall = recall_score(all_labels, all_outputs, average='weighted')

        print(f'Validation Loss: {np.mean(val_loss)}, F1 Score: {f1}, Precision: {precision}, Recall: {recall}')

        self.model.train()


if __name__ == "__main__":
    trainer = Teacher(SelectModel, num_epochs=30)
    try:
        trainer.train()
    except KeyboardInterrupt:
        trainer.validate_model('test')
        trainer.save_model('selection.pth')
