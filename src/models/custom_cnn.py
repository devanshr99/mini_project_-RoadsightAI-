import torch
import torch.nn as nn
import torch.nn.functional as F


class SignNetLite(nn.Module):
    """
    Custom 4-layer Convolutional Neural Network baseline for Road Sign Classification.
    
    Designed specifically for mini-project presentation & viva defense:
    - 4 Convolutional blocks (Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d)
    - Fully connected classifier with Dropout regularization
    - Explicit parameter count and compact size for edge/embedded deployment.
    """
    def __init__(self, num_classes=43, input_channels=3):
        super(SignNetLite, self).__init__()
        
        # Block 1: Input (3, 64, 64) -> Output (32, 32, 32)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Block 2: Input (32, 32, 32) -> Output (64, 16, 16)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Block 3: Input (64, 16, 16) -> Output (128, 8, 8)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Block 4: Input (128, 8, 8) -> Output (128, 4, 4)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.4)
        
        # Dense classifier
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Feature extraction
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Classification head
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits

    def get_last_conv_layer(self):
        """Returns the final conv layer for Grad-CAM explainability hook."""
        return self.conv4


def build_custom_cnn(num_classes=43):
    return SignNetLite(num_classes=num_classes)
