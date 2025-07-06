import torch.nn as nn
import timm

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