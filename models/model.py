# model_torch.py

import torch
import torch.nn as nn
import timm
from mamba_ssm import Mamba
from timm.models.swin_transformer import SwinTransformer
from timm.models.vision_transformer import VisionTransformer


import argparse


# from transformers import ConvNextModel
# class ConvNextBackboneWrapper(nn.Module):
#     def __init__(self, hf_model):
#         super().__init__()
#         self.model = hf_model

#     def forward(self, x):
#         out = self.model(pixel_values=x)
#         return out.last_hidden_state.flatten(1)  # Shape: [B, C]

# def load_hf_convnext_backbone(hf_repo="ahsanjavid/convnext-tiny-finetuned-cifar10"):
#     model = ConvNextModel.from_pretrained(hf_repo)
#     # return ConvNextBackboneWrapper(model)
#     return model


# class ConvNextBackboneWrapper(nn.Module):
#     def __init__(self, hf_model):
#         super().__init__()
#         self.model = hf_model

#     def forward(self, x):
#         # Hugging Face ConvNeXt expects input as `pixel_values=...`
#         out = self.model(pixel_values=x)
#         # Extract last hidden state [B, C, 1, 1] → flatten to [B, C]
#         last_hidden = out.last_hidden_state
#         return last_hidden.flatten(1)


# from transformers import ConvNextForImageClassification
# def load_hf_convnext_backbone(hf_repo="ahsanjavid/convnext-tiny-finetuned-cifar10", to_gray=True):
#     model = ConvNextForImageClassification.from_pretrained(hf_repo)
#     model.config.num_channels = 1

#     backbone = model.convnext  # Extract the backbone (without the classifier head)
#     # backbone=model

#     if to_gray:
#         # Convert first conv layer from (3,...) → (1,...)
#         old_conv = backbone.embeddings.patch_embeddings
#         new_conv = nn.Conv2d(
#             in_channels=1,
#             out_channels=old_conv.out_channels,
#             kernel_size=old_conv.kernel_size,
#             stride=old_conv.stride,
#             padding=old_conv.padding,
#             bias=old_conv.bias is not None,
#         )

#         # Average RGB weights to grayscale
#         with torch.no_grad():
#             new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
#             if old_conv.bias is not None:
#                 new_conv.bias[:] = old_conv.bias
#         backbone.embeddings.patch_embeddings = new_conv

#     # return backbone
#     return ConvNextBackboneWrapper(backbone)
# class ConvNeXtCIFARClassifier(nn.Module):
#     def __init__(self, input_shape=(1, 32, 32), pretrained=True):
#         super().__init__()
#         self.backbone_name = 'convnext_cifar'
        
#         # Load ConvNeXt-Tiny CIFAR10 pretrained
#         self.backbone = timm.create_model(
#             'convnext_tiny.fb_in22k_ft_in1k',  # Can be replaced with CIFAR if available
#             pretrained=pretrained,
#             in_chans=3,  # Keep 3 channels
#             num_classes=0
#         )
#         self.features = self.backbone
#         self.features.num_features = self.backbone.num_features

#         self.energy_loss_head = nn.Linear(self.features.num_features, 1)
#         self.alpha_head = nn.Linear(self.features.num_features, 3)
#         self.q0_head = nn.Linear(self.features.num_features, 4)

#     def forward(self, x):
#         # Assume x: (B, 1, 32, 32) — replicate to 3 channels
#         if x.shape[1] == 1:
#             x = x.repeat(1, 3, 1, 1)
#         feats = self.features(x)
#         return {
#             'energy_loss_output': self.energy_loss_head(feats),
#             'alpha_output': self.alpha_head(feats),
#             'q0_output': self.q0_head(feats),
#         }
class ConvNeXtClassifier(nn.Module):
    def __init__(self, input_shape=(1, 32, 32), pretrained=True, model_name='convnext_tiny'):
        super().__init__()
        self.backbone_name = 'convnext'

        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, in_chans=3, num_classes=0
        )
        self.features = self.backbone
        self.features.num_features = self.backbone.num_features

        self.energy_loss_head = nn.Linear(self.features.num_features, 1)
        self.alpha_head = nn.Linear(self.features.num_features, 3)
        self.q0_head = nn.Linear(self.features.num_features, 4)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        feats = self.features(x)
        return {
            'energy_loss_output': self.energy_loss_head(feats),
            'alpha_output': self.alpha_head(feats),
            'q0_output': self.q0_head(feats),
        }


# from mamba_ssm import Mamba  # Assuming `mamba-ssm` is installed for Mamba model
import torch.nn.functional as F

class MambaVisionMultiHead(nn.Module):
    def __init__(self, in_chans=1, img_size=32, embed_dim=128, mamba_layers=4, mamba_hidden=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=3, padding=1),
            nn.Flatten(2),
            nn.Linear(img_size*img_size, img_size),
        )
        self.norm= nn.LayerNorm(embed_dim)
        self.mamba = Mamba(d_model=embed_dim, d_state=mamba_hidden, d_conv=mamba_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head_energy = nn.Linear(embed_dim, 1)
        self.head_alpha  = nn.Linear(embed_dim, 3)
        self.head_q0     = nn.Linear(embed_dim, 4)

    def forward(self, x):
        # x: (B,1,32,32)
        z = self.proj(x)               # (B, embed_dim, 32)
        z = z.permute(2,0,1)           # (seq_len, B, embed_dim)
        out_seq = self.mamba(z)        # (seq_len, B, embed_dim)
        feat = out_seq[-1]             # (B, embed_dim)
        return {
            'energy_loss_output': self.head_energy(feat),
            'alpha_output':  self.head_alpha(feat),
            'q0_output':     self.head_q0(feat)
        }

class MultiHeadClassifier(nn.Module):
    def __init__(self, backbone='efficientnet', input_shape=(1, 32, 32), 
                 d_model=512, init_weights='imagenet'):
        super(MultiHeadClassifier, self).__init__()

        self.backbone_name = backbone.lower()

        # ----------------------
        # Shared Backbone (Flexible)
        # ----------------------

        if self.backbone_name == 'efficientnet':
            # EfficientNet: Modify input layer to accept 1 channel instead of 3
            self.backbone = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
            self.backbone.conv_stem = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.features = self.backbone
        elif self.backbone_name == 'convnext':
            print(f"[INFO] Using ConvNeXt backbone with input shape: {input_shape}")
            # ConvNeXt: Modify input layer to accept 1 channel instead of 3
            self.backbone = timm.create_model('convnext_small.in12k_ft_in1k', pretrained=False, in_chans=1, num_classes=0)
            # self.backbone.stem[0] = nn.Conv2d(1, 96, kernel_size=(4, 4), stride=(4, 4), padding=(1, 1), bias=False)  # Adjust the first conv layer
            # self.backbone.norm = nn.LayerNorm(96)  # Adjust normalization layer (optional, depending on the model)
            self.backbone.conv_stem = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            print(f"[INFO] ConvNeXt backbone initialized with input shape: {input_shape}")
            self.features = self.backbone
        elif self.backbone_name == 'swin':
            # 1) Build a Swin that takes 1×32×32 inputs directly:
            #    - patch_size divides 32, e.g. 4 → produces (32/4=8) patches per dim
            #    - window_size also divides 8, e.g. 4 → non‐overlapping windows
            self.backbone = SwinTransformer(
                img_size=32,
                patch_size=4,
                in_chans=1,
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=4,
                mlp_ratio=4.,
                qkv_bias=True,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.1,
                norm_layer=nn.LayerNorm,
                patch_norm=True,
                use_checkpoint=False,
                num_classes=0  # <— so it returns features, not 1000‐class logits
            )
            self.features=self.backbone
            # 2) The final feature dim is embed_dim * 2^(len(depths)-1)
            #    Since swin builds hierarchies, use its num_features attr:
            self.features.num_features = self.backbone.num_features
        elif self.backbone_name == 'vit':
            # ViT: Modify input layer to accept 1 channel instead of 3
            # self.backbone = timm.create_model('vit_base_patch16_224', pretrained=False, in_chans=1, num_classes=0)
            # self.backbone.patch_embed.proj = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16), bias=False)
            # self.features = self.backbone

            # 1) Build a ViT that takes 1×32×32 inputs directly:
            #    - img_size=32, patch_size=4 → (32/4=8)^2 = 64 patches
            #    - embed_dim small for speed, e.g. 192
            #    - depth=4 layers, num_heads=3 (must divide 192)
            self.backbone = VisionTransformer(
                img_size=32,
                patch_size=4,
                in_chans=1,
                embed_dim=192,
                depth=4,
                num_heads=3,
                mlp_ratio=4.0,
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.1,
                norm_layer=nn.LayerNorm,
                num_classes=0,        # returns features
            )

            self.features=self.backbone
            # 2) Feature dimension from ViT’s output:
            self.features.num_features = self.backbone.embed_dim  # 192
        elif self.backbone_name == 'mamba_vision':
              # Use the MambaVisionMultiHead as a self-contained backbone
            self.features = MambaVisionMultiHead(
                in_chans=input_shape[0],
                img_size=input_shape[1]
            )
        # elif self.backbone_name == 'convnext_cifar':
        #     self.backbone = load_hf_convnext_backbone()
        #     self.features = self.backbone
        #     # self.features.num_features = self.backbone.layernorm.normalized_shape[0]
        #     # Automatically detect feature dim
        #     with torch.no_grad():
        #         dummy = torch.randn(1, 1, 32, 32)
        #         self.features.num_features = self.features(dummy).shape[1]
        # elif self.backbone_name == 'convnext_cifar':
        #     self.backbone = load_hf_convnext_backbone()
        #     self.features = self.backbone

        #     # Automatically detect feature dimension
        #     with torch.no_grad():
        #         # dummy = torch.randn(1, 3, 32, 32)  # 3-channel dummy input
        #         # self.features.num_features = self.features(dummy).shape[1]
        #         self.features.num_features = self.backbone.layernorm.normalized_shape[0]
        else:
            raise ValueError(f"Unsupported backbone model: {self.backbone_name}")

        # ----------------------
        # Output Heads
        # ----------------------
        # Output Heads for other backbones
        if self.backbone_name != 'mamba_vision':
            # For EfficientNet/ConvNeXt/SwinV2, the feature dimension is typically `num_features` or `hidden_size`
            self.energy_loss_head = nn.Linear(self.features.num_features, 1)  # Sigmoid for binary output
            self.alpha_head = nn.Linear(self.features.num_features, 3)  # Softmax for 3-class output
            self.q0_head = nn.Linear(self.features.num_features, 4)  # Softmax for 4-class output


    def forward(self, x):
        if self.backbone_name == 'mamba_vision':
            return self.features(x)
        
        feats = self.features(x)
        # if self.backbone_name == 'convnext_cifar' and x.shape[1] == 1:
        #     x = x.repeat(1, 3, 1, 1)
        #     feats = self.features(pixel_values=x) 
        return {
            'energy_loss_output': self.energy_loss_head(feats),
            'alpha_output': self.alpha_head(feats),
            'q0_output': self.q0_head(feats)
        }
# Helper: Weight Initialization

def weights_init_normal(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
# ---------------------------
# Model Creation Helper
# ---------------------------
def create_model(backbone='efficientnet', input_shape=(1, 32, 32),
                learning_rate=0.001, d_model=512,
                init_weights='imagenet'):# 'imagenet', 'cifar', or 'gaussian'
    """
    Helper function to create and compile a MultiHeadClassifier.
    """
    # if backbone == 'convnext_cifar':
    #     model = ConvNeXtCIFARClassifier(input_shape=input_shape,pretrained=pretrained)
    if backbone.startswith('convnext'):
        suffix = backbone[len('convnext_'):]  # get everything after 'convnext_'

        if suffix == 'fb_in22k_ft_in1k':
            model_name = 'convnext_tiny.fb_in22k_ft_in1k'
            pretrained = True
        elif suffix == 'fb_in1k':
            model_name = 'convnext_tiny.fb_in1k'
            pretrained = True
        elif suffix == 'gaussian':
            model_name = 'convnext_tiny'
            pretrained = False
        elif suffix == '':
            model_name = 'convnext_tiny'
            pretrained = False
        else:
            raise ValueError(f"Unrecognized convnext variant: '{suffix}'")
        
        print(f"Using model: {model_name}, pretrained: {pretrained}")


        model = ConvNeXtClassifier(input_shape=input_shape, pretrained=pretrained, model_name=model_name)

        if suffix == 'gaussian':
            model.apply(weights_init_normal)
    else:
        model = MultiHeadClassifier(backbone=backbone, input_shape=input_shape,
                                d_model=d_model, init_weights=init_weights)

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
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension (for Mamba)')  # Add d_model argument

    args = parser.parse_args()

    print(f"\n[INFO] Testing backbone: {args.backbone}")
    print(f"[INFO] Input shape: {args.input_shape}")
    print(f"[INFO] Learning rate: {args.learning_rate}")
    print(f"[INFO] d_model: {args.d_model}")

    # Create the model with specified parameters
    model, _ = create_model(backbone=args.backbone,
                                     input_shape=tuple(args.input_shape),
                                     learning_rate=args.learning_rate,
                                    d_model=args.d_model)
    # Display the model summary
    print("\n[INFO] Model Summary:")
    print(model)
    

    # # Example random input for testing
    x_gray = torch.randn(16, 1, 32, 32)
    # 1-channel grayscale batch
    print(f"\n[INFO] Model output for input shape {x_gray.shape}:")
    out = model(x_gray)

    print(out)  # Should be a dict with logits

    
    # for key, value in outputs.items():
    #     print(f"{key}: {value.shape}")

if __name__ == "__main__":
    main()
