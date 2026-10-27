import pytest
from PIL import Image
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.inference import TrafficSignPredictor


def test_predictor_pipeline():
    predictor = TrafficSignPredictor(model_name="custom_cnn", img_dim=64)
    dummy_img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))

    result = predictor.predict(dummy_img)

    assert "predicted_class_id" in result
    assert "predicted_class_name" in result
    assert "confidence" in result
    assert "top3_predictions" in result
    assert "gradcam_overlay_b64" in result

    assert len(result["top3_predictions"]) == 3
    assert result["gradcam_overlay_b64"].startswith("data:image/png;base64,")
