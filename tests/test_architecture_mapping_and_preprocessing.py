import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest
import torch
from PIL import Image
import numpy as np

from src.dataset import preprocess_image
from src.models.factory import get_model, get_checkpoint_for_model, resolve_model_key, ARCHITECTURE_CHECKPOINT_MAP
from src.inference import TrafficSignPredictor
from src.explain import GradCAM


def test_explicit_checkpoint_mapping():
    """Verifies each architecture explicitly resolves to its designated checkpoint."""
    assert get_checkpoint_for_model("Custom CNN") == "checkpoints/custom_cnn_best.pth"
    assert get_checkpoint_for_model("custom_cnn") == "checkpoints/custom_cnn_best.pth"
    
    assert get_checkpoint_for_model("SignNet-Lite") == "checkpoints/signnet_lite_best.pth"
    assert get_checkpoint_for_model("signnet_lite") == "checkpoints/signnet_lite_best.pth"
    
    assert get_checkpoint_for_model("MobileNetV2") == "checkpoints/mobilenet_v2_best.pth"
    assert get_checkpoint_for_model("mobilenet_v2") == "checkpoints/mobilenet_v2_best.pth"
    
    assert get_checkpoint_for_model("ResNet18") == "checkpoints/resnet18_best.pth"
    assert get_checkpoint_for_model("resnet18") == "checkpoints/resnet18_best.pth"


def test_preprocessing_arbitrary_dimensions_and_modes():
    """Verifies that arbitrary high-resolution, RGBA, or grayscale images are always preprocessed to [1, 3, 64, 64]."""
    # High resolution RGBA image (like user's videoframe_78842.png: 1280x720)
    high_res_rgba = Image.fromarray(np.random.randint(0, 255, (720, 1280, 4), dtype=np.uint8), mode="RGBA")
    tensor = preprocess_image(high_res_rgba)
    assert tensor.shape == (1, 3, 64, 64), f"Expected [1, 3, 64, 64], got {tensor.shape}"

    # Odd size Grayscale image
    odd_gray = Image.fromarray(np.random.randint(0, 255, (137, 219), dtype=np.uint8), mode="L")
    tensor_gray = preprocess_image(odd_gray)
    assert tensor_gray.shape == (1, 3, 64, 64), f"Expected [1, 3, 64, 64], got {tensor_gray.shape}"


def test_preprocessing_strict_shape_validation():
    """Verifies that invalid tensor dimensions passed to GradCAM generate_heatmap raise a ValueError."""
    model = get_model("custom_cnn", num_classes=43)
    target_layer = model.get_last_conv_layer()
    grad_cam = GradCAM(model, target_layer)

    # Invalid tensor: shape [1, 3, 128, 128]
    invalid_tensor = torch.randn(1, 3, 128, 128)
    with pytest.raises(ValueError, match=r"Grad-CAM input_tensor shape must be \[B, 3, 64, 64\]"):
        grad_cam.generate_heatmap(invalid_tensor)


def test_strict_state_dict_loading_mismatch_detection():
    """Verifies that strict=True catches architecture mismatches instead of hiding them."""
    # Attempting to load ResNet18 weights into Custom CNN architecture
    custom_model = get_model("custom_cnn", num_classes=43)
    resnet_checkpoint = torch.load("checkpoints/resnet18_best.pth", map_location="cpu")

    with pytest.raises(RuntimeError):
        custom_model.load_state_dict(resnet_checkpoint["model_state_dict"], strict=True)


def test_all_four_predictors_instantiate_and_predict():
    """Verifies all four models instantiate, load checkpoints with strict=True, and predict on a high-res image."""
    models_to_test = ["custom_cnn", "signnet_lite", "mobilenet_v2", "resnet18"]
    test_img = Image.fromarray(np.random.randint(0, 255, (720, 1280, 4), dtype=np.uint8), mode="RGBA")

    for m in models_to_test:
        predictor = TrafficSignPredictor(model_name=m, img_dim=64)
        assert predictor.checkpoint_path == ARCHITECTURE_CHECKPOINT_MAP[m]

        result = predictor.predict(test_img)
        assert "predicted_class_id" in result
        assert 0 <= result["predicted_class_id"] < 43
        assert "confidence" in result
        assert len(result["top3_predictions"]) == 3
        assert result["gradcam_overlay_b64"].startswith("data:image/png;base64,")
