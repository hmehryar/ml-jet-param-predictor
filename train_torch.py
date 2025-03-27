import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from loader_torch import JetDataset  # Import the dataset from your loader script

# -------------------------------
# CNN Architecture Definition
# -------------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 32 * 32, 512)  # Adjust input size based on your final output of conv layers
        self.fc2 = nn.Linear(512, 3)  # energy_loss_output (3 classes)
        self.fc3 = nn.Linear(512, 3)  # alpha_output (3 classes)
        self.fc4 = nn.Linear(512, 4)  # q0_output (4 classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(torch.relu(self.conv2(x)), 2)
        x = torch.max_pool2d(torch.relu(self.conv3(x)), 2)
        
        # Flatten the output for fully connected layer
        x = x.view(x.size(0), -1)
        
        # Fully connected layers for each output
        energy_loss = self.fc2(torch.relu(self.fc1(x)))  # Energy loss classification
        alpha = self.fc3(torch.relu(self.fc1(x)))  # alpha_s classification
        q0 = self.fc4(torch.relu(self.fc1(x)))  # Q0 classification
        
        return energy_loss, alpha, q0

# -------------------------------
# Training & Validation Loops
# -------------------------------
def train_model(train_loader, model, criterion, optimizer, device):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels['energy_loss_output'] = labels['energy_loss_output'].to(device)
        labels['alpha_output'] = labels['alpha_output'].to(device)
        labels['q0_output'] = labels['q0_output'].to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        energy_loss, alpha, q0 = model(inputs)

        # Compute the loss for each output
        loss_energy_loss = criterion[0](energy_loss, labels['energy_loss_output'])
        loss_alpha = criterion[1](alpha, labels['alpha_output'])
        loss_q0 = criterion[2](q0, labels['q0_output'])

        # Total loss
        total_loss = loss_energy_loss + loss_alpha + loss_q0

        # Backward pass
        total_loss.backward()

        # Optimize
        optimizer.step()

        # Update running loss
        running_loss += total_loss.item()

        # Calculate accuracy for energy loss (example, similar for others)
        _, predicted = torch.max(energy_loss.data, 1)
        total += labels['energy_loss_output'].size(0)
        correct += (predicted == labels['energy_loss_output']).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    return epoch_loss, epoch_accuracy

def validate_model(val_loader, model, criterion, device):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # No gradient computation during validation
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels['energy_loss_output'] = labels['energy_loss_output'].to(device)
            labels['alpha_output'] = labels['alpha_output'].to(device)
            labels['q0_output'] = labels['q0_output'].to(device)

            # Forward pass
            energy_loss, alpha, q0 = model(inputs)

            # Compute the loss for each output
            loss_energy_loss = criterion[0](energy_loss, labels['energy_loss_output'])
            loss_alpha = criterion[1](alpha, labels['alpha_output'])
            loss_q0 = criterion[2](q0, labels['q0_output'])

            # Total loss
            total_loss = loss_energy_loss + loss_alpha + loss_q0

            # Update running loss
            running_loss += total_loss.item()

            # Calculate accuracy for energy loss (example, similar for others)
            _, predicted = torch.max(energy_loss.data, 1)
            total += labels['energy_loss_output'].size(0)
            correct += (predicted == labels['energy_loss_output']).sum().item()

    epoch_loss = running_loss / len(val_loader)
    epoch_accuracy = 100 * correct / total
    return epoch_loss, epoch_accuracy

# -------------------------------
# Main Script
# -------------------------------
def main():
    # Hyperparameters and configurations
    batch_size = 512
    num_epochs = 10
    learning_rate = 1e-4
    global_max = 121.79151153564453  # from your script
    root_dir = 'your_dataset_root_dir_here'
    
    # Prepare DataLoader
    train_list = 'your_train_file_list.csv'
    val_list = 'your_val_file_list.csv'
    
    # Apply transformations (if needed)
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = JetDataset(train_list, global_max, transform=transform)
    val_dataset = JetDataset(val_list, global_max, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize the model, loss function, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    
    criterion = [
        nn.CrossEntropyLoss(),  # For energy_loss classification
        nn.CrossEntropyLoss(),  # For alpha_s classification
        nn.CrossEntropyLoss()   # For q0 classification
    ]
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training and validation loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train the model
        train_loss, train_acc = train_model(train_loader, model, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        
        # Validate the model
        val_loss, val_acc = validate_model(val_loader, model, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")
        
if __name__ == "__main__":
    main()
