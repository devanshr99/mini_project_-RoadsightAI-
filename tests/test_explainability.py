import pytest
import torch
from PIL import Image
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.custom_cnn import SignNetLite
from src.explain import GradCAM, overlay_heatmap


def test_gradcam_heatmap_generation():
    model = SignNetLite(num_classes=43)
    target_layer = model.get_last_conv_layer()
    grad_cam = GradCAM(model, target_layer)

    dummy_input = torch.randn(1, 3, 64, 64)
    heatmap, pred_class, probs = grad_cam.generate_heatmap(dummy_input)

    assert heatmap is not None
    assert isinstance(heatmap, np.ndarray)
    assert heatmap.ndim == 2, "Heatmap must be 2D array"
    assert 0 <= pred_class < 43


def test_overlay_heatmap():
    dummy_heatmap = np.random.rand(8, 8)
    dummy_pil = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))

    blended_img, heatmap_img = overlay_heatmap(dummy_heatmap, dummy_pil)
    assert blended_img.size == (64, 64)
    assert heatmap_img.size == (64, 64)
