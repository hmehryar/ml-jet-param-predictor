import torch
import torch.nn as nn
import timm


class MambaToViTClassifier(nn.Module):
    def __init__(self, mamba_model_name='mambaout_base_plus_rw', vit_model_name='vit_tiny_patch16_224', 
                 output_dims=(1, 3, 4)):
        super().__init__()

        # Add this line before loading self.mamba
        self.input_proj = nn.Conv2d(1, 3, kernel_size=1)  # grayscale → RGB

        # Load Mamba backbone (features only mode)
        self.mamba = timm.create_model(
            mamba_model_name, 
            pretrained=False,
            features_only=True)
        self.reshape_to_image = nn.Sequential(
            nn.Linear(768, 3 * 224 * 224),
            nn.Unflatten(1, (3, 224, 224))  # [B, 3, 16, 16]
        )
        self.vit = timm.create_model(vit_model_name, pretrained=False, in_chans=3, num_classes=0)

        vit_embed_dim = self.vit.num_features

        # Task-specific classification heads
        self.head_eloss = nn.Linear(vit_embed_dim, output_dims[0])  # Binary
        self.head_alpha = nn.Linear(vit_embed_dim, output_dims[1])  # 3-class
        self.head_q0    = nn.Linear(vit_embed_dim, output_dims[2])  # 4-class

    def forward(self, x):  # x shape: [B, 1, 32, 32]
        x = self.input_proj(x)                 # [B, 3, 32, 32]
        x = self.mamba(x)[-1]                  # [B, C, H, W]

        if x.ndim == 4:
            x = x.view(x.size(0), -1)  # flatten if [B, C, 1, 1]
        x = self.reshape_to_image(x)   # [B, 3, 16, 16]
        x = self.vit.forward_features(x)[:, 0]  # [B, D] ← CLS token
        return {
            "energy_loss_output": self.head_eloss(x),
            "alpha_output": self.head_alpha(x),
            "q0_output": self.head_q0(x)
        }
    

def create_model(backbone=['mambaout_base_plus_rw.sw_e150_in12k', 'vit_tiny_patch16_224'], 
                 input_shape=(1, 32, 32),
                 learning_rate=0.0001,
                 ):

    # ✅ Hybrid: Mamba ➝ ViT stack
    if isinstance(backbone, list) and len(backbone) == 2:
        mamba_name = backbone[0]
        vit_name = backbone[1]
        print(f"Using Hybrid Mamba ➝ ViT model: {mamba_name} → {vit_name}")
        model = MambaToViTClassifier(
            mamba_model_name=mamba_name,
            vit_model_name=vit_name,
            output_dims=(1, 3, 4)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        return model, optimizer
