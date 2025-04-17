
# models/model_factory.py
import torch
import torch.nn as nn
from timm import create_model as timm_create_model


def create_model(backbone, input_shape, learning_rate):
    if backbone == "efficientnet":
        model = timm_create_model("efficientnet_b0", pretrained=False, in_chans=input_shape[0], num_classes=0)
    elif backbone == "convnext":
        model = timm_create_model("convnext_tiny", pretrained=False, in_chans=input_shape[0], num_classes=0)
    elif backbone == "swin":
        model = timm_create_model("swin_tiny_patch4_window7_224", pretrained=False, in_chans=input_shape[0], num_classes=0)
    elif backbone == "mamba":
        from mamba_ssm.models.mixer_seq_simple import Mamba
        model = Mamba(d_model=192, n_layer=6, vocab_size=None)  # placeholder config
    elif backbone == "vision_mamba":
        from mamba_ssm.models.vit_mamba import VisionMamba
        model = VisionMamba(img_size=32, patch_size=4, in_chans=input_shape[0])
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    # === Attach classification heads ===
    class MultiHeadClassifier(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            embed_dim = base.num_features if hasattr(base, 'num_features') else 192  # fallback for Mamba

            self.energy_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(embed_dim, 1), nn.Sigmoid()
            )
            self.alpha_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(embed_dim, 3)
            )
            self.q0_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(embed_dim, 4)
            )

        def forward(self, x):
            feats = self.base(x)
            return {
                "energy_loss_output": self.energy_head(feats),
                "alpha_output": self.alpha_head(feats),
                "q0_output": self.q0_head(feats)
            }

    model = MultiHeadClassifier(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return model, optimizer
