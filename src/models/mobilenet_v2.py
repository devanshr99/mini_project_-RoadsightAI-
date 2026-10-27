import os
import sys
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# Fix Windows DLL search path for PyTorch C++ binaries
if sys.platform == "win32":
    torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
    if os.path.exists(torch_lib):
        try:
            os.add_dll_directory(torch_lib)
        except Exception:
            pass


class TransferMobileNetV2(nn.Module):
    """
    MobileNetV2 Transfer Learning Architecture for Traffic Sign Recognition.
    Pre-trained on ImageNet, fine-tuned for 43 traffic sign classes.
    Supports backbone freezing and selective fine-tuning of upper inverted residual blocks.
    """
    def __init__(self, num_classes=43, pretrained=True, freeze_backbone=True):
        super(TransferMobileNetV2, self).__init__()
        
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v2(weights=weights)
        
        # Replace classification head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
        
        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        """Freezes feature extractor layers except classifier head."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

    def unfreeze_top_blocks(self, n_blocks=4):
        """Unfreezes top n inverted residual blocks in features for fine-tuning."""
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True
        # Unfreeze feature blocks from back
        for i in range(len(self.backbone.features) - n_blocks, len(self.backbone.features)):
            for param in self.backbone.features[i].parameters():
                param.requires_grad = True

    def unfreeze_all(self):
        """Unfreezes all parameters across the entire network."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)

    def get_last_conv_layer(self):
        """Returns the final convolutional layer of MobileNetV2 features for Grad-CAM."""
        return self.backbone.features[-1]


def build_mobilenet_v2(num_classes=43, pretrained=True, freeze_backbone=True):
    return TransferMobileNetV2(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)
