import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from data.loader_torch import JetDataset  # Import the dataset from your loader script
from data.loader_torch import load_split_from_csv, JetDataset


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
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # Adjust input size based on your final output of conv layers
        self.fc2 = nn.Linear(512, 2)  # energy_loss_output (2 classes)
        self.fc3 = nn.Linear(512, 3)  # alpha_output (3 classes)
        self.fc4 = nn.Linear(512, 4)  # q0_output (4 classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(torch.relu(self.conv2(x)), 2)
        x = torch.max_pool2d(torch.relu(self.conv3(x)), 2)
        
        # Print the size of the tensor after the convolution and pooling layers
        print(f"Shape before flattening: {x.shape}")
        
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

        # Check types of model outputs and labels
        print(f"energy_loss type: {energy_loss.dtype}, alpha type: {alpha.dtype}, q0 type: {q0.dtype}")
        print(f"labels['energy_loss_output'] type: {labels['energy_loss_output'].dtype}")
        print(f"labels['alpha_output'] type: {labels['alpha_output'].dtype}")
        print(f"labels['q0_output'] type: {labels['q0_output'].dtype}")
        
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
# Main Training Function
# -------------------------------
def train(root_dir, output_dir, model_tag, global_max=121.79151153564453, batch_size=512, epochs=10,  learning_rate=1e-4, resume=None):
    # Prepare DataLoader
    train_file = os.path.join(root_dir, 'train_files.csv')
    val_file = os.path.join(root_dir, 'val_files.csv')
    print(f"Loading datasets from \n {train_file} \n and {val_file}")
    # Apply transformations (if needed)
    transform = transforms.Compose([transforms.ToTensor()])
    # train_file = os.path.join(root_dir, "train_files.csv")
    
    train_list = load_split_from_csv(train_file,root_dir)
    val_list = load_split_from_csv(val_file,root_dir)
    # shown the length of the datasets
    print(f"Length of training dataset: {len(train_list)}")
    print(f"Length of validation dataset: {len(val_list)}")

    # train_dataset = JetDataset(train_list, global_max=global_max, transform=transform)
    # val_dataset = JetDataset(val_list, global_max=global_max, transform=transform)
    train_dataset = JetDataset(train_list, global_max=global_max)
    val_dataset = JetDataset(val_list, global_max=global_max)

    # #Show the length of the datasets
    print(f"Length of training dataset: {len(train_dataset)}")
    print(f"Length of validation dataset: {len(val_dataset)}")
    
    img, labels = val_dataset[0]
    print(f"Format of validation dataset: {img.shape}, {labels['energy_loss_output'].shape}, {labels['alpha_output'].shape}, {labels['q0_output'].shape}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    #show the length of the dataloaders
    print(f"Length of training dataloader: {len(train_loader)}")
    print(f"Length of validation dataloader: {len(val_loader)}")

    # Initialize the model, loss function, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # model = SimpleCNN().to(device)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    criterion = [
        nn.CrossEntropyLoss(),  # For energy_loss classification
        nn.CrossEntropyLoss(),  # For alpha_s classification
        nn.CrossEntropyLoss()   # For q0 classification
    ]

    from models.model_torch import create_model
    # ✅ Choose backbone: 'efficientnet', 'convnext', 'swin', or 'mamba'
    backbone = 'efficientnet'
    # backbone = 'convnext'
    # backbone = 'swin'
    # backbone = 'mamba'

    # ✅ Create model and optimizer
    model, optimizer = create_model(backbone=backbone, input_shape=(1, 32, 32), learning_rate=learning_rate)
    model = model.to(device)
    print(f"Model architecture: {model}")
    print(f"Optimizer: {optimizer}")

    # # Resume from checkpoint if specified
    if resume:
        print(f"Resuming training from checkpoint: {resume}")
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint['best_val_acc']
        print(f"Resumed training from epoch {start_epoch}. Best validation accuracy: {best_val_acc}%")
    else:
        start_epoch = 0
        best_val_acc = 0

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    # Training and validation loop
    for epoch in range(start_epoch, epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Train the model
        train_loss, train_acc = train_model(train_loader, model, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        
    #     # Validate the model
    #     val_loss, val_acc = validate_model(val_loader, model, criterion, device)
    #     print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")

    #     # Save the model if it's the best validation accuracy so far
    #     if val_acc > best_val_acc:
    #         best_val_acc = val_acc
    #         torch.save({
    #             'epoch': epoch + 1,
    #             'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'best_val_acc': best_val_acc
    #         }, os.path.join(output_dir, f'{model_tag}_best_model.pth'))
    #         print(f"Best model saved with validation accuracy: {best_val_acc}%")

    #     # Save the model after every epoch
    #     torch.save({
    #         'epoch': epoch + 1,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'best_val_acc': best_val_acc
    #     }, os.path.join(output_dir, f'{model_tag}_epoch_{epoch+1}.pth'))

# -------------------------------
# Command-line interface
# -------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Training pipeline for ML-JET multi-task classifier")
    parser.add_argument('--root_dir', type=str, required=True, help='Path to dataset root (containing splits)')
    parser.add_argument('--global_max', type=float, required=True, help='Global max for normalization')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--output_dir', type=str, default='checkpoints/', help='Directory to save models and logs')
    parser.add_argument('--model_tag', type=str, required=True, help='Unique model tag (e.g., EfficientNet, Transformer)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    return parser.parse_args()

# -------------------------------
# Main Script
# -------------------------------
def main():
    global_max = 121.79151153564453  # from your script
    args = parse_args()
    train(
        root_dir=args.root_dir,
        global_max=args.global_max,
        batch_size=args.batch_size,
        epochs=args.epochs,
        output_dir=args.output_dir,
        model_tag=args.model_tag,
        learning_rate=args.learning_rate,
        resume=args.resume
    )
    


        
if __name__ == "__main__":
    main()
