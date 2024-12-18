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
from ml.dataset import ClassifireDataset, SelectionDataset, PeaksDataset




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


class PeaksModel(nn.Module):
    def __init__(self):
        super(PeaksModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2600, 2000),
            nn.ReLU(),
            nn.Linear(2000, 3000),
            nn.ReLU(),
            nn.Linear(3000, 1300),
            nn.ReLU(),
            nn.Linear(1300, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1300*2),

        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), 1300, 2)
        return self.sig(x)


class Teacher:
    def __init__(self, model, num_epochs=10, loss_fn=nn.NLLLoss):
        self.model = model()
        self.num_epochs = num_epochs
        self.criterion = loss_fn() #nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)


    def train(self, dataset_model):
        self.model.train()
        epoch_loss = []
        dataset = dataset_model()

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
                labels = labels.long().to(self.device)  # Ensure labels are of type long
                labels = F.one_hot(labels, num_classes=2)  # One-hot encode the labels
                labels = labels.float()

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
        peaks_count = []
        if mode == 'val':
            loader = self.val_loader
        else:
            loader = self.test_loader
        with torch.no_grad():
            for inputs, labels in  loader:
                inputs = inputs.float().to(self.device)
                labels = labels.long().to(self.device)

                labels = F.one_hot(labels, num_classes=2)  # One-hot encode the labels
                labels = labels.float()
                outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)

                val_loss.append(loss.item())
                peaks_labels = []
                peaks_output = []
                # Convert outputs and labels to numpy arrays and add them to the lists
                outputs = np.argmax(outputs.cpu().numpy(), axis=-1)
                all_outputs.extend(outputs)
                labels = np.argmax(labels.cpu().numpy(), axis=-1)
                all_labels.extend(labels)
                peaks_labels = np.count_nonzero(labels, axis=-1)
                peaks_output = np.count_nonzero(outputs, axis=-1)
                peaks_count.extend(np.abs(peaks_output-peaks_labels))


        # Calculate metrics

        f1 = f1_score(all_labels, all_outputs, average='weighted', zero_division=1)
        precision = precision_score(all_labels, all_outputs, average='weighted', zero_division=1)
        recall = recall_score(all_labels, all_outputs, average='weighted',zero_division=1)

        print(f'Validation Loss: {np.mean(val_loss)}, F1 Score: {f1}, Precision: {precision}, Recall: {recall}, peaks = {np.mean(peaks_count)}')

        self.model.train()


if __name__ == "__main__":
    trainer = Teacher(PeaksModel, num_epochs=30, loss_fn=nn.CrossEntropyLoss) #nn.BCEWithLogitsLoss)
    try:
        trainer.train(PeaksDataset)
    except KeyboardInterrupt:
        trainer.validate_model('test')
        trainer.save_model('peaks.pth')
