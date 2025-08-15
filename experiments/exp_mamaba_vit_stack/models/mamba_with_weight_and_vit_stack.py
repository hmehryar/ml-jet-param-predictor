import torch

def load_mamba_backbone_weights(hybrid_model, checkpoint_path,device='cuda'):
    """
    Load pretrained Mamba weights into a hybrid Mamba+ViT model
    by stripping off classification heads.

    Args:
        hybrid_model (nn.Module): instance of MambaToViTClassifier
        checkpoint_path (str): path to trained MambaClassifier .pt file
    """
    # Load state_dict from original model
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Extract only keys for backbone from the saved model
    backbone_state_dict = {}
    for k, v in ckpt.items():
        if k.startswith("features."):
            # Strip 'features.' prefix to match timm internal keys
            backbone_state_dict[k[len("features."):]] = v

    missing, unexpected = hybrid_model.mamba.load_state_dict(backbone_state_dict, strict=False)
    print(f"[INFO] Loaded Mamba backbone. Missing keys: {len(missing)} | Unexpected: {len(unexpected)}")

    return hybrid_model

from hybrid_mamba_vit import MambaToViTClassifier
def create_model(backbone=['mambaout_base_plus_rw.sw_e150_in12k', 'vit_tiny_patch16_224'], 
                 input_shape=(1, 32, 32),
                 learning_rate=0.0001,
                 best_mamba_path=None):

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

        model = load_mamba_backbone_weights(model, best_mamba_path)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        return model, optimizer
    else:
        raise ValueError("Backbone must be a list with two elements: [mamba_model_name, vit_model_name]")

best_mamba_path= "/home/arsalan/wsu-grid/ml-jet-param-predictor/training_output/"\
"mambaout_base_plus_rw_g500_bs16_ep50_lr1e-04_ds7200000_g500_sched_ReduceLROnPlateau/last_model.pth"
model, optimizer = create_model(
    backbone=['mambaout_base_plus_rw.sw_e150_in12k', 'vit_tiny_patch16_224'],
    input_shape=(1, 32, 32),
    learning_rate=0.0001,
    best_mamba_path=best_mamba_path
)
    
