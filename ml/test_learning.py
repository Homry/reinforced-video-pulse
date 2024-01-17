import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import your dataset and model classes
from your_dataset import YourDataset
from your_model import ResNet

def test_model(model, test_dataset):
    # Set the model to evaluation mode
    model.eval()

    # Define the loss function
    criterion = nn.MSELoss()

    # Create a data loader for the test dataset
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Initialize variables for metrics
    total_loss = 0.0
    total_samples = 0
    total_mse = 0.0
    total_rmse = 0.0
    total_mape = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Update metrics
            total_loss += loss.item()
            total_samples += inputs.size(0)
            total_mse += loss.item()
            total_rmse += torch.sqrt(loss).item()
            total_mape += torch.mean(torch.abs(outputs - labels) / labels).item()

    # Calculate average metrics
    avg_loss = total_loss / total_samples
    avg_mse = total_mse / total_samples
    avg_rmse = total_rmse / total_samples
    avg_mape = total_mape / total_samples

    # Print the metrics
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average RMSE: {avg_rmse:.4f}")
    print(f"Average MAPE: {avg_mape:.4f}")

# Create an instance of the model trainer
model_trainer = ModelTrainer(model_file="model.pth")

# Load the test dataset
test_dataset = YourDataset()  # Replace YourDataset with your own test dataset class

# Train the model
model_trainer.train_model()

# Load the trained model
model = model_trainer.load_model()

# Test the model
test_model(model, test_dataset)