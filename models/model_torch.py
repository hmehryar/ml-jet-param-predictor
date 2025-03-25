# model_torch.py

import torch
import torch.nn as nn
import timm
from transformers import SwinForImageClassification
from mamba_ssm import Mamba  # Assuming `mamba-ssm` is installed for Mamba model
import torch.nn.functional as F
import argparse


class MultiHeadClassifier(nn.Module):
    def __init__(self, backbone='efficientnet', input_shape=(1, 32, 32)):
        super(MultiHeadClassifier, self).__init__()

        self.backbone_name = backbone.lower()

        # ----------------------
        # Shared Backbone (Flexible)
        # ----------------------
        if self.backbone_name == 'efficientnet':
            # EfficientNet: Modify input layer to accept 1 channel instead of 3
            self.backbone = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
            self.backbone.conv_stem = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        elif self.backbone_name == 'convnext':
            # ConvNeXt: Modify input layer to accept 1 channel instead of 3
            self.backbone = timm.create_model('convnext_base', pretrained=False, num_classes=0)
            self.backbone.stem[0] = nn.Conv2d(1, 96, kernel_size=(4, 4), stride=(4, 4), padding=(1, 1), bias=False)  # Adjust the first conv layer
        elif self.backbone_name == 'swin':
            self.backbone = SwinForImageClassification.from_pretrained("microsoft/swin-base-patch4-window7-224")
            self.backbone = self.backbone.backbone  # Extract only the backbone part of SwinV2
        elif self.backbone_name == 'mamba':
            self.backbone = Mamba()  # Assuming we have Mamba integrated for state-space models
        else:
            raise ValueError(f"Unsupported backbone model: {self.backbone_name}")

        # ----------------------
        # Output Heads
        # ----------------------
        self.energy_loss_head = nn.Linear(self.backbone.num_features, 1)  # Sigmoid for binary output
        self.alpha_head = nn.Linear(self.backbone.num_features, 3)  # Softmax for 3-class output
        self.q0_head = nn.Linear(self.backbone.num_features, 4)  # Softmax for 4-class output

    def forward(self, x):
        # Backbone forward pass
        x = self.backbone(x)

        # Multi-head outputs
        energy_loss_output = torch.sigmoid(self.energy_loss_head(x))  # Sigmoid activation for binary classification
        alpha_output = F.softmax(self.alpha_head(x), dim=1)  # Softmax for αₛ classification
        q0_output = F.softmax(self.q0_head(x), dim=1)  # Softmax for Q₀ classification

        return {
            'energy_loss_output': energy_loss_output,
            'alpha_output': alpha_output,
            'q0_output': q0_output
        }

# ---------------------------
# Model Creation Helper
# ---------------------------
def create_model(backbone='efficientnet', input_shape=(1, 32, 32), learning_rate=0.001):
    """
    Helper function to create and compile a MultiHeadClassifier.
    """
    model = MultiHeadClassifier(backbone=backbone, input_shape=input_shape)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return model, optimizer

# Example input to test the model
# python models/model_torch.py --backbone efficientnet
# python models/model_torch.py --backbone convnext
# python models/model_torch.py --backbone swin
# python models/model_torch.py --backbone mamba

# ---------------------------
# Main function to test the model
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Test different backbone models for ML-JET multi-head classifier.")
    parser.add_argument('--backbone', type=str, default='efficientnet',
                        help='Backbone model to test (options: efficientnet, convnext, swin, mamba)')
    parser.add_argument('--input_shape', type=int, nargs=3, default=[1, 32, 32],
                        help='Input shape as three integers (default: 1 32 32)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    args = parser.parse_args()

    print(f"\n[INFO] Testing backbone: {args.backbone}")
    print(f"[INFO] Input shape: {args.input_shape}")
    print(f"[INFO] Learning rate: {args.learning_rate}")

    # Create the model with specified parameters
    model, optimizer = create_model(backbone=args.backbone,
                                     input_shape=tuple(args.input_shape),
                                     learning_rate=args.learning_rate)
    
    model.eval()  # Set model to evaluation mode

    # Example random input for testing
    dummy_input = torch.randn(1, *args.input_shape)  # Example batch with one 32x32 event image
    outputs = model(dummy_input)

    print(f"\n[INFO] Model summary for {args.backbone}:")
    for key, value in outputs.items():
        print(f"{key}: {value.shape}")

if __name__ == "__main__":
    main()
