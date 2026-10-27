import os
import sys
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# Windows DLL search path fix for PyTorch C++ binaries
if sys.platform == "win32":
    torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
    if os.path.exists(torch_lib):
        try:
            os.add_dll_directory(torch_lib)
        except Exception:
            pass


class TransferResNet18(nn.Module):
    """
    ResNet-18 Transfer Learning Architecture for Traffic Sign Recognition.
    Pre-trained on ImageNet, modified with a 43-class classification head.
    Supports backbone freezing and selective fine-tuning of upper layers.
    """
    def __init__(self, num_classes=43, pretrained=True, freeze_backbone=True):
        super(TransferResNet18, self).__init__()
        
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        
        # Replace fully-connected classification head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
        
        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        """Freezes all layers in the backbone except the custom FC classification head."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def unfreeze_layer4(self):
        """Unfreezes layer4 and FC head for selective fine-tuning."""
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def unfreeze_all(self):
        """Unfreezes all parameters across the entire network."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)

    def get_last_conv_layer(self):
        """Returns final conv layer of layer4 for Grad-CAM explainability hook."""
        return self.backbone.layer4[-1].conv2


def build_resnet18(num_classes=43, pretrained=True, freeze_backbone=True):
    return TransferResNet18(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)
