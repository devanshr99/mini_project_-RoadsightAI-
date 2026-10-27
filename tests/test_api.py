import pytest
import io
import json
from PIL import Image
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_api_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "healthy"
    assert "torch_version" in data
    assert "models" in data
    assert "custom_cnn" in data["models"]
    assert data["models"]["custom_cnn"]["available"] is True


def test_api_models(client):
    res = client.get("/api/models")
    assert res.status_code == 200
    data = res.get_json()
    assert "models" in data
    assert len(data["models"]) == 4
    model_ids = [m["id"] for m in data["models"]]
    assert "custom_cnn" in model_ids
    assert "mobilenet_v2" in model_ids
    assert "resnet18" in model_ids


def test_api_classes(client):
    res = client.get("/api/classes")
    assert res.status_code == 200
    data = res.get_json()
    assert "classes" in data
    assert data["total_classes"] == 43
    assert data["classes"]["14"] == "Stop"


def test_api_dataset_summary(client):
    res = client.get("/api/dataset/summary")
    assert res.status_code == 200
    data = res.get_json()
    assert data["num_classes"] == 43
    assert data["total_images"] > 0
    assert "splits" in data


def test_api_evaluation(client):
    res = client.get("/api/evaluation")
    assert res.status_code == 200
    data = res.get_json()
    assert "benchmark" in data
    assert len(data["benchmark"]) >= 4
    assert "plots" in data


def test_api_experiments(client):
    res = client.get("/api/experiments")
    assert res.status_code == 200
    data = res.get_json()
    assert "experiments" in data
    assert data["total_runs"] > 0


def test_api_predict_missing_image(client):
    res = client.post("/api/predict", data={})
    assert res.status_code == 400
    data = res.get_json()
    assert "error" in data


def test_api_predict_invalid_file_extension(client):
    data = {
        "image": (io.BytesIO(b"fake file"), "test.exe")
    }
    res = client.post("/api/predict", data=data, content_type="multipart/form-data")
    assert res.status_code == 400
    data = res.get_json()
    assert "Invalid file format" in data["error"]


def test_api_predict_success(client):
    # Create valid synthetic RGB traffic sign image
    img = Image.new("RGB", (64, 64), color=(200, 30, 30))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    data = {
        "image": (buf, "test_sign.png"),
        "model": "custom_cnn"
    }
    res = client.post("/api/predict", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    result = res.get_json()
    
    assert "predicted_class_id" in result
    assert "predicted_class_name" in result
    assert "confidence" in result
    assert "top3_predictions" in result
    assert len(result["top3_predictions"]) == 3
    assert "gradcam_overlay_b64" in result
    assert result["gradcam_overlay_b64"].startswith("data:image/png;base64,")
    assert "heatmap_b64" in result
    assert "inference_latency_ms" in result
    assert result["is_valid"] is True


def test_api_predict_json_base64(client):
    import base64
    img = Image.new("RGB", (64, 64), color=(30, 100, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    res = client.post(
        "/api/predict",
        data=json.dumps({"image_base64": b64, "model": "mobilenet_v2"}),
        content_type="application/json"
    )
    assert res.status_code == 200
    result = res.get_json()
    assert result["model_name"] == "mobilenet_v2"
    assert len(result["top3_predictions"]) == 3
