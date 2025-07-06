import torch.nn as nn
import timm
# from mamba_ssm import Mamba  # Assuming `mamba-ssm` is installed for Mamba model
import torch.nn.functional as F

class MambaClassifier(nn.Module):
    def __init__(self, input_shape=(1, 32, 32), pretrained=True, model_name='mambaout_base_plus_rw'):
        super().__init__()
        self.backbone_name = 'mamba'

        # Load ViT backbone from timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=3,  # Use 3 channels, replicate grayscale input if needed
            num_classes=0
        )
        self.features = self.backbone
        # self.features.num_features = self.backbone.num_features  # Get embedding dim
        self.features.num_features = 3072  # Assuming Mamba's output feature size is 3072, adjust as needed
        # print(f"[DEBUG] Mamba features num_features: {self.features.num_features}")
        # Multi-task heads
        self.energy_loss_head = nn.Linear(self.features.num_features, 1)
        self.alpha_head = nn.Linear(self.features.num_features, 3)
        self.q0_head = nn.Linear(self.features.num_features, 4)

    def forward(self, x):
        # If input is grayscale (1 channel), replicate to 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        # Resize: (B, 3, 32, 32) → (B, 3, 224, 224)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # feats = self.features(x)  # Shape: (B, C)
        feats = self.features(x)  # likely (B, L, C)
        if feats.ndim == 3:
            feats = feats.mean(dim=1)  # average pool over sequence length
        # print("[DEBUG] Mamba features shape:", feats.shape)

        return {
            'energy_loss_output': self.energy_loss_head(feats),
            'alpha_output': self.alpha_head(feats),
            'q0_output': self.q0_head(feats),
        }
