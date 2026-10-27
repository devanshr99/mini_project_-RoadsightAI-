import os
import sys
import io
import time
import json
import csv
import base64
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.factory import (
    ARCHITECTURE_CHECKPOINT_MAP,
    MODEL_NAME_ALIASES,
    resolve_model_key,
    get_checkpoint_for_model,
)
from src.inference import TrafficSignPredictor

app = Flask(__name__, static_folder="experiments/plots")

# Allowed upload extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB limit


# In-memory predictor cache to avoid repeated disk reads
class PredictorCache:
    def __init__(self):
        self._cache = {}

    def get_predictor(self, model_name: str) -> TrafficSignPredictor:
        canonical_key = resolve_model_key(model_name)
        if canonical_key not in self._cache:
            checkpoint_path = get_checkpoint_for_model(canonical_key)
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    f"Checkpoint not found at '{checkpoint_path}' for architecture '{canonical_key}'"
                )
            self._cache[canonical_key] = TrafficSignPredictor(
                model_name=canonical_key,
                checkpoint_path=checkpoint_path,
                img_dim=64
            )
        return self._cache[canonical_key]


predictor_cache = PredictorCache()


# --- CORS Middleware ---
@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin")
    # Allow local development frontend origins
    if origin in ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"]:
        response.headers["Access-Control-Allow-Origin"] = origin
    else:
        response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
    return response


# --- Model Metadata Registry ---
MODEL_METADATA = {
    "custom_cnn": {
        "id": "custom_cnn",
        "display_name": "Custom CNN (Baseline)",
        "architecture": "4-Layer Convolutional Network + BatchNorm + MaxPool",
        "checkpoint": "checkpoints/custom_cnn_best.pth",
        "parameters": "2,323,243 (~2.32M)",
        "model_size_mb": 8.92,
        "input_resolution": "64x64 RGB",
        "test_accuracy": "98.92%",
        "macro_f1": "0.9895",
        "avg_cpu_latency_ms": 3.82,
        "supports_gradcam": True,
        "gradcam_target_layer": "conv4 (Last Conv Layer)",
        "description": "Custom four-stage feature extractor engineered from scratch for GTSRB with batch normalization, 0.4 dropout, and Adam optimizer.",
    },
    "signnet_lite": {
        "id": "signnet_lite",
        "display_name": "SignNet-Lite",
        "architecture": "Lightweight Compact CNN with Depth Optimization",
        "checkpoint": "checkpoints/signnet_lite_best.pth",
        "parameters": "2,323,243 (~2.32M)",
        "model_size_mb": 8.92,
        "input_resolution": "64x64 RGB",
        "test_accuracy": "98.92%",
        "macro_f1": "0.9895",
        "avg_cpu_latency_ms": 1.97,
        "supports_gradcam": True,
        "gradcam_target_layer": "conv4",
        "description": "Optimized lightweight variant tuned for edge inference and minimal latency while preserving high recognition precision.",
    },
    "mobilenet_v2": {
        "id": "mobilenet_v2",
        "display_name": "MobileNetV2 (Transfer Learning)",
        "architecture": "Inverted Residuals + Linear Bottlenecks (ImageNet pre-trained)",
        "checkpoint": "checkpoints/mobilenet_v2_best.pth",
        "parameters": "2,279,723 (~2.28M)",
        "model_size_mb": 26.44,
        "input_resolution": "64x64 RGB",
        "test_accuracy": "98.40%",
        "macro_f1": "0.9829",
        "avg_cpu_latency_ms": 28.53,
        "supports_gradcam": True,
        "gradcam_target_layer": "features.18 (Final 1x1 Conv)",
        "description": "Transfer learning using inverted residual blocks. Phase A: Frozen ImageNet backbone; Phase B: Fine-tuned top 4 bottleneck blocks.",
    },
    "resnet18": {
        "id": "resnet18",
        "display_name": "ResNet18 (Deep Residual)",
        "architecture": "18-Layer Deep Residual Network with Skip Connections",
        "checkpoint": "checkpoints/resnet18_best.pth",
        "parameters": "11,198,571 (~11.20M)",
        "model_size_mb": 128.28,
        "input_resolution": "64x64 RGB",
        "test_accuracy": "99.67%",
        "macro_f1": "0.9972",
        "avg_cpu_latency_ms": 10.94,
        "supports_gradcam": True,
        "gradcam_target_layer": "layer4.1.conv2",
        "description": "Deep residual architecture leveraging identity skip connections to mitigate vanishing gradients. Fine-tuned layer4 + classifier.",
    },
}


# --- API Routes ---

@app.route("/api/health", methods=["GET"])
def health_check():
    """System health check reporting device, checkpoint statuses, and readiness."""
    models_status = {}
    for key, path in ARCHITECTURE_CHECKPOINT_MAP.items():
        exists = os.path.exists(path)
        size_bytes = os.path.getsize(path) if exists else 0
        models_status[key] = {
            "checkpoint_path": path,
            "available": exists,
            "size_mb": round(size_bytes / (1024 * 1024), 2) if exists else 0,
        }

    dataset_summary_path = "data/dataset_summary.json"
    dataset_ready = os.path.exists(dataset_summary_path)

    return jsonify({
        "status": "healthy",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "torch_version": torch.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_available": torch.cuda.is_available(),
        "models": models_status,
        "dataset_ready": dataset_ready,
        "server": "RoadSight AI REST Engine (Flask / PyTorch)",
    })


@app.route("/api/models", methods=["GET"])
def get_models():
    """Returns all available models with architectural specifications and checkpoint availability."""
    models_list = []
    for key, meta in MODEL_METADATA.items():
        chk_path = meta["checkpoint"]
        exists = os.path.exists(chk_path)
        size_mb = round(os.path.getsize(chk_path) / (1024 * 1024), 2) if exists else 0
        
        item = dict(meta)
        item["available"] = exists
        item["actual_file_size_mb"] = size_mb
        models_list.append(item)

    return jsonify({"models": models_list})


@app.route("/api/classes", methods=["GET"])
def get_classes():
    """Returns the official 43-class GTSRB traffic sign taxonomy."""
    class_file = "data/class_names.json"
    if os.path.exists(class_file):
        with open(class_file, "r") as f:
            classes = json.load(f)
        return jsonify({"classes": classes, "total_classes": len(classes)})
    return jsonify({"error": "Class taxonomy file not found"}), 404


@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict():
    """
    Main inference endpoint.
    Accepts multipart/form-data with 'image' file or JSON with 'image_base64'.
    Performs preprocessing, inference, and Grad-CAM generation.
    """
    if request.method == "OPTIONS":
        return "", 200

    start_time = time.perf_counter()

    # Model selection parameter
    if request.is_json and request.json:
        model_name = request.json.get("model") or request.args.get("model", "custom_cnn")
    else:
        model_name = request.form.get("model") or request.args.get("model", "custom_cnn")

    try:
        model_key = resolve_model_key(model_name)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Image extraction
    pil_image = None
    if "image" in request.files:
        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400
        
        ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify({
                "error": f"Invalid file format '{ext}'. Supported formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            }), 400

        try:
            image_bytes = file.read()
            if len(image_bytes) > MAX_CONTENT_LENGTH:
                return jsonify({"error": "File size exceeds 10MB limit"}), 400
            pil_image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            return jsonify({"error": f"Could not read image file: {str(e)}"}), 400

    elif request.is_json and "image_base64" in request.json:
        b64_data = request.json["image_base64"]
        if "," in b64_data:
            b64_data = b64_data.split(",", 1)[1]
        try:
            image_bytes = base64.b64decode(b64_data)
            pil_image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            return jsonify({"error": f"Invalid base64 image data: {str(e)}"}), 400
    else:
        return jsonify({"error": "No image provided. Upload an image file as 'image' or send 'image_base64'"}), 400

    # Validate image dimensions and mode
    try:
        pil_image.verify()
        # Reopen because verify() closes the file
        pil_image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        return jsonify({"error": f"Corrupt or unreadable image: {str(e)}"}), 400

    width, height = pil_image.size
    if width < 16 or height < 16:
        return jsonify({"error": f"Image dimensions too small ({width}x{height}). Minimum is 16x16 pixels."}), 400
    if width > 4096 or height > 4096:
        return jsonify({"error": f"Image dimensions too large ({width}x{height}). Maximum is 4096x4096 pixels."}), 400

    # Run inference with cached predictor
    try:
        predictor = predictor_cache.get_predictor(model_key)
        result = predictor.predict(pil_image)
    except FileNotFoundError as e:
        return jsonify({"error": str(e), "model_status": "checkpoint_missing"}), 404
    except Exception as e:
        return jsonify({"error": f"Inference execution failed: {str(e)}"}), 500

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
    result["inference_latency_ms"] = elapsed_ms
    result["input_dimensions"] = f"{width}x{height}"
    result["is_valid"] = True

    return jsonify(result)


@app.route("/api/dataset/summary", methods=["GET"])
def dataset_summary():
    """Returns verified dataset audit metrics, class distribution, and splits."""
    summary_path = "data/dataset_summary.json"
    if not os.path.exists(summary_path):
        return jsonify({"error": "Dataset summary not generated yet"}), 404
    with open(summary_path, "r") as f:
        data = json.load(f)
    return jsonify(data)


@app.route("/api/dataset/sample", methods=["GET"])
def get_dataset_sample():
    """Returns a real sample image from the dataset with its true label."""
    class_id = request.args.get("class_id", None)
    
    # Check processed or raw folder
    base_dirs = ["data/processed/test", "data/processed", "data/raw"]
    target_dir = None
    for b in base_dirs:
        if os.path.exists(b):
            target_dir = b
            break

    if not target_dir:
        return jsonify({"error": "No dataset sample directory found"}), 404

    # Load class names
    class_names = {}
    if os.path.exists("data/class_names.json"):
        with open("data/class_names.json", "r") as f:
            class_names = json.load(f)

    # Search for an image
    selected_img_path = None
    found_class = 0
    if class_id is not None:
        try:
            cid = int(class_id)
            c_str = f"{cid:02d}"
            c_dir = os.path.join(target_dir, c_str)
            if not os.path.exists(c_dir):
                c_dir = os.path.join(target_dir, str(cid))
            if os.path.exists(c_dir):
                files = [f for f in os.listdir(c_dir) if f.lower().endswith(('.png', '.jpg', '.ppm'))]
                if files:
                    selected_img_path = os.path.join(c_dir, files[0])
                    found_class = cid
        except ValueError:
            pass

    if not selected_img_path:
        # Pick first available sample from class 14 (Stop) or 1 (Speed limit 30) or any
        for preferred in ["14", "01", "13", "17", "25", "00"]:
            p_dir = os.path.join(target_dir, preferred)
            if os.path.exists(p_dir):
                files = [f for f in os.listdir(p_dir) if f.lower().endswith(('.png', '.jpg', '.ppm'))]
                if files:
                    selected_img_path = os.path.join(p_dir, files[0])
                    found_class = int(preferred)
                    break

    if not selected_img_path:
        return jsonify({"error": "Could not locate a sample image in dataset"}), 404

    try:
        img = Image.open(selected_img_path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

        return jsonify({
            "class_id": found_class,
            "class_name": class_names.get(str(found_class), f"Class {found_class}"),
            "image_b64": b64,
            "dimensions": f"{img.width}x{img.height}",
            "filename": os.path.basename(selected_img_path)
        })
    except Exception as e:
        return jsonify({"error": f"Failed reading sample image: {str(e)}"}), 500


@app.route("/api/dataset/samples-gallery", methods=["GET"])
def get_samples_gallery():
    """Returns a curated list of diverse test samples for one-click testing in the UI."""
    curated_classes = [
        14,  # Stop
        1,   # Speed limit 30
        2,   # Speed limit 50
        13,  # Yield
        17,  # No entry
        18,  # General caution
        25,  # Road work
        23,  # Slippery road
        38,  # Keep right
        35,  # Ahead only
        12,  # Priority road
        33,  # Turn right ahead
    ]

    class_names = {}
    if os.path.exists("data/class_names.json"):
        with open("data/class_names.json", "r") as f:
            class_names = json.load(f)

    gallery = []
    base_dirs = ["data/processed", "data/raw"]
    target_dir = next((b for b in base_dirs if os.path.exists(b)), None)

    if not target_dir:
        return jsonify({"gallery": []})

    for cid in curated_classes:
        c_str = f"{cid:02d}"
        c_dir = os.path.join(target_dir, c_str)
        if not os.path.exists(c_dir):
            c_dir = os.path.join(target_dir, str(cid))
        
        if os.path.exists(c_dir):
            files = [f for f in os.listdir(c_dir) if f.lower().endswith(('.png', '.jpg', '.ppm'))]
            if files:
                img_path = os.path.join(c_dir, files[0])
                try:
                    img = Image.open(img_path).convert("RGB")
                    # Create a compact thumbnail
                    thumb = img.copy()
                    thumb.thumbnail((120, 120))
                    buf = io.BytesIO()
                    thumb.save(buf, format="PNG")
                    b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

                    gallery.append({
                        "class_id": cid,
                        "class_name": class_names.get(str(cid), f"Class {cid}"),
                        "image_b64": b64,
                        "dimensions": f"{img.width}x{img.height}"
                    })
                except Exception:
                    continue

    return jsonify({"gallery": gallery})


@app.route("/api/experiments", methods=["GET"])
def get_experiments():
    """Returns the recorded experimental runs and hyperparameters from experiment_log.csv."""
    csv_path = "experiments/experiment_log.csv"
    if not os.path.exists(csv_path):
        return jsonify({"experiments": []})

    experiments = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            experiments.append(row)

    return jsonify({"experiments": experiments, "total_runs": len(experiments)})


@app.route("/api/evaluation", methods=["GET"])
def get_evaluation():
    """Returns evaluation metrics, benchmark comparison, and available plots."""
    plots_dir = "experiments/plots"
    available_plots = []
    if os.path.exists(plots_dir):
        available_plots = [
            f for f in os.listdir(plots_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.svg'))
        ]

    # Models benchmark summary table from real experiment results
    benchmark = [
        {
            "model_id": "custom_cnn",
            "model_name": "Custom CNN",
            "test_accuracy": 98.92,
            "macro_f1": 0.9895,
            "val_accuracy": 99.05,
            "latency_ms": 3.82,
            "params": "2.32M",
            "size_mb": 8.92,
            "confusion_matrix_plot": "/plots/custom_cnn_confusion_matrix.png",
            "curves_plot": "/plots/custom_cnn_curves.png",
        },
        {
            "model_id": "signnet_lite",
            "model_name": "SignNet-Lite",
            "test_accuracy": 98.92,
            "macro_f1": 0.9895,
            "val_accuracy": 99.05,
            "latency_ms": 1.97,
            "params": "2.32M",
            "size_mb": 8.92,
            "confusion_matrix_plot": "/plots/custom_cnn_confusion_matrix.png",
            "curves_plot": "/plots/custom_cnn_curves.png",
        },
        {
            "model_id": "mobilenet_v2",
            "model_name": "MobileNetV2 (Phase B)",
            "test_accuracy": 98.40,
            "macro_f1": 0.9829,
            "val_accuracy": 98.57,
            "latency_ms": 28.53,
            "params": "2.28M",
            "size_mb": 26.44,
            "confusion_matrix_plot": "/plots/mobilenet_v2_confusion_matrix.png",
            "curves_plot": "/plots/mobilenet_v2_curves.png",
        },
        {
            "model_id": "resnet18",
            "model_name": "ResNet18 (Phase B)",
            "test_accuracy": 99.67,
            "macro_f1": 0.9972,
            "val_accuracy": 99.42,
            "latency_ms": 10.94,
            "params": "11.20M",
            "size_mb": 128.28,
            "confusion_matrix_plot": "/plots/resnet18_confusion_matrix.png",
            "curves_plot": "/plots/resnet18_curves.png",
        },
    ]

    return jsonify({
        "benchmark": benchmark,
        "plots": [f"/plots/{p}" for p in available_plots],
        "plots_metadata": {
            "model_accuracy_comparison": "/plots/model_accuracy_comparison.png",
            "model_f1_comparison": "/plots/model_f1_comparison.png",
            "inference_latency_comparison": "/plots/inference_latency_comparison.png",
            "parameters_vs_accuracy": "/plots/parameters_vs_accuracy.png",
            "class_distribution": "/plots/class_distribution.png",
            "image_dimensions": "/plots/image_dimensions.png",
            "sample_images": "/plots/sample_images.png",
        }
    })


@app.route("/plots/<path:filename>")
def serve_plot(filename):
    """Serves static evaluation plots."""
    return send_from_directory("experiments/plots", filename)


if __name__ == "__main__":
    print("[RoadSight AI] Starting server on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)
