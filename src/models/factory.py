from .custom_cnn import build_custom_cnn
from .mobilenet_v2 import build_mobilenet_v2
from .resnet18 import build_resnet18


ARCHITECTURE_CHECKPOINT_MAP = {
    "custom_cnn": "checkpoints/custom_cnn_best.pth",
    "signnet_lite": "checkpoints/signnet_lite_best.pth",
    "mobilenet_v2": "checkpoints/mobilenet_v2_best.pth",
    "resnet18": "checkpoints/resnet18_best.pth",
}

MODEL_NAME_ALIASES = {
    "custom_cnn": "custom_cnn",
    "custom cnn": "custom_cnn",
    "custom": "custom_cnn",
    "signnet_lite": "signnet_lite",
    "signnet-lite": "signnet_lite",
    "signnet": "signnet_lite",
    "signnetlite": "signnet_lite",
    "baseline": "custom_cnn",
    "mobilenet_v2": "mobilenet_v2",
    "mobilenet-v2": "mobilenet_v2",
    "mobilenetv2": "mobilenet_v2",
    "mobilenet": "mobilenet_v2",
    "transfer_net": "mobilenet_v2",
    "resnet18": "resnet18",
    "resnet-18": "resnet18",
    "resnet": "resnet18",
}


def resolve_model_key(model_name: str) -> str:
    """Normalizes any model name string to its canonical registry key."""
    cleaned = str(model_name).lower().strip().replace("-", "_").replace(" ", "_")
    if cleaned in MODEL_NAME_ALIASES:
        return MODEL_NAME_ALIASES[cleaned]
    if "custom" in cleaned:
        return "custom_cnn"
    if "signnet" in cleaned or "lite" in cleaned:
        return "signnet_lite"
    if "mobile" in cleaned:
        return "mobilenet_v2"
    if "resnet" in cleaned:
        return "resnet18"
    raise ValueError(
        f"Unknown model architecture: '{model_name}'. "
        f"Supported choices: {list(ARCHITECTURE_CHECKPOINT_MAP.keys())}"
    )


def get_checkpoint_for_model(model_name: str) -> str:
    """Returns the explicit checkpoint file path mapped to the requested architecture."""
    key = resolve_model_key(model_name)
    return ARCHITECTURE_CHECKPOINT_MAP[key]


def get_model(model_name="custom_cnn", num_classes=43, pretrained=False, freeze_backbone=True):
    """
    Factory function to instantiate the exact model architecture before loading weights.
    
    Supported model_name values:
    - 'custom_cnn' / 'Custom CNN'
    - 'signnet_lite' / 'SignNet-Lite'
    - 'mobilenet_v2' / 'MobileNetV2'
    - 'resnet18' / 'ResNet18'
    """
    key = resolve_model_key(model_name)
    if key in ["custom_cnn", "signnet_lite"]:
        return build_custom_cnn(num_classes=num_classes)
    elif key == "mobilenet_v2":
        return build_mobilenet_v2(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)
    elif key == "resnet18":
        return build_resnet18(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)
    else:
        raise ValueError(f"Unknown model architecture key: '{key}'")
