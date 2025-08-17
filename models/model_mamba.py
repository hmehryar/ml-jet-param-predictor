import torch
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

class MambaVisionMultiHead(nn.Module):
    
    def __init__(self, in_chans=1, img_size=32, embed_dim=128, mamba_layers=4, mamba_hidden=256):
        from mamba_ssm import Mamba
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
    

from models.model import weights_init_normal
from models.model import MultiHeadClassifier
def create_model(backbone="vit_gaussian", 
                 input_shape=(1, 32, 32),
                 learning_rate=0.0001,
                 ):

    if backbone.startswith('mambaout'):
        suffix = backbone[len('mambaout_'):]

        if suffix == 'base_plus_rw.sw_e150_in12k_ft_in1k':
            model_name = 'mambaout_base_plus_rw.sw_e150_in12k_ft_in1k'
            pretrained = True
        elif suffix == 'base_plus_rw.sw_e150_in12k':
            model_name = 'mambaout_base_plus_rw.sw_e150_in12k'
            pretrained = True
        elif suffix == 'base_plus_rw_gaussian':
            model_name = 'mambaout_base_plus_rw'
            pretrained = False
        elif suffix == 'base_plus_rw':
            model_name = 'mambaout_base_plus_rw'
            pretrained = False
        else:
            raise ValueError(f"Unrecognized mamba variant: '{suffix}'")

        print(f"Using Mamba model: {model_name}, pretrained: {pretrained}")
        model = MambaClassifier(input_shape=input_shape, pretrained=pretrained, model_name=model_name)
        if suffix == 'gaussian':
            model.apply(weights_init_normal)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        return model, optimizer
    else:
        model = MultiHeadClassifier(backbone=backbone, input_shape=input_shape)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return model, optimizer
