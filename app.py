"""
RoadSight AI — Flask Backend
Memory-safe prediction on Render CPU (512 MB free tier).

Design constraints:
  - Only ONE model is ever resident in RAM at a time (PredictorCache).
  - threading.Lock prevents concurrent requests from loading two models simultaneously.
  - Every request releases raw image bytes, PIL image, and tensors via try/finally.
  - /api/memory endpoint reports live RSS and peak RSS for diagnostics.
  - Flask MAX_CONTENT_LENGTH enforces 10 MB upload limit at the framework level.
"""

import os
import sys
import gc
import io
import time
import json
import csv
import base64
import threading
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.factory import (
    ARCHITECTURE_CHECKPOINT_MAP,
    resolve_model_key,
    get_checkpoint_for_model,
)
from src.inference import TrafficSignPredictor

# ---------------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder="experiments/plots")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1_048_576   # 10 MB — enforced by Flask

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
MAX_IMAGE_BYTES    = 10 * 1_048_576
MAX_IMAGE_DIM      = 4096
MIN_IMAGE_DIM      = 16

# ---------------------------------------------------------------------------
# Memory helper (optional — graceful fallback if psutil not installed)
# ---------------------------------------------------------------------------
try:
    import psutil as _psutil
    _PROC = _psutil.Process()
    def _rss_mb() -> float:
        return _PROC.memory_info().rss / 1_048_576
except ImportError:
    _psutil = None
    def _rss_mb() -> float:
        return -1.0


# ---------------------------------------------------------------------------
# Model cache — strict single-model-at-a-time policy with thread safety
# ---------------------------------------------------------------------------
class PredictorCache:
    """
    Maintains at most ONE TrafficSignPredictor in memory.

    Thread safety:
      - A threading.Lock() prevents two threads from loading models concurrently
        (gunicorn --threads 1 means only one request at a time, but the lock is
        here as a safety net in case the config changes).
      - Double-checked locking: if the requested model is already loaded, we
        return immediately without acquiring the lock.

    Eviction:
      - When a new model is requested, the existing predictor's .model attribute
        is explicitly deleted before gc.collect() so PyTorch can free the tensors.
    """

    def __init__(self):
        self._cache: dict[str, TrafficSignPredictor] = {}
        self._lock = threading.Lock()

    def get_predictor(self, model_name: str) -> TrafficSignPredictor:
        key = resolve_model_key(model_name)

        # Fast path — already loaded, no lock needed
        if key in self._cache:
            return self._cache[key]

        with self._lock:
            # Double-checked: another thread may have loaded it while we waited
            if key in self._cache:
                return self._cache[key]

            before = _rss_mb()
            print(f"[Cache] Loading '{key}'  |  RAM before: {before:.1f} MB", flush=True)

            # Evict the previous model first to free its tensors before loading the new one
            if self._cache:
                old_key, old_predictor = next(iter(self._cache.items()))
                try:
                    del old_predictor.model
                except Exception:
                    pass
                del old_predictor
                self._cache.clear()
                gc.collect()
                after_evict = _rss_mb()
                print(f"[Cache] Evicted '{old_key}'  |  RAM after eviction: {after_evict:.1f} MB", flush=True)

            ckpt_path = get_checkpoint_for_model(key)
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(
                    f"Checkpoint not found: '{ckpt_path}' for model '{key}'"
                )

            self._cache[key] = TrafficSignPredictor(
                model_name=key,
                checkpoint_path=ckpt_path,
                img_dim=64,
            )
            after_load = _rss_mb()
            print(f"[Cache] Loaded '{key}'   |  RAM after load: {after_load:.1f} MB  (+{after_load - before:.1f} MB)", flush=True)

        return self._cache[key]

    def current_model(self) -> str | None:
        return next(iter(self._cache), None)

    def clear(self):
        with self._lock:
            for p in self._cache.values():
                try:
                    del p.model
                except Exception:
                    pass
            self._cache.clear()
            gc.collect()


predictor_cache = PredictorCache()


# ---------------------------------------------------------------------------
# CORS middleware
# ---------------------------------------------------------------------------
ALLOWED_ORIGINS = {
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
}


@app.before_request
def handle_options_preflight():
    if request.method == "OPTIONS":
        resp = app.make_default_options_response()
        origin = request.headers.get("Origin", "*")
        resp.headers["Access-Control-Allow-Origin"] = origin if origin in ALLOWED_ORIGINS else "*"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
        resp.headers["Access-Control-Max-Age"] = "86400"
        return resp


@app.after_request
def add_cors_headers(resp):
    origin = request.headers.get("Origin", "*")
    resp.headers["Access-Control-Allow-Origin"] = origin if origin in ALLOWED_ORIGINS else "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
    return resp


# ---------------------------------------------------------------------------
# Model metadata registry (display only — not used for loading)
# ---------------------------------------------------------------------------
MODEL_METADATA = {
    "custom_cnn": {
        "id": "custom_cnn",
        "display_name": "Custom CNN (Baseline)",
        "architecture": "4-Layer Convolutional Network + BatchNorm + MaxPool",
        "checkpoint": "checkpoints/custom_cnn_best.pth",
        "parameters": "2,323,243 (~2.32M)",
        "model_size_mb": 2.98,
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
        "model_size_mb": 2.98,
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
        "model_size_mb": 8.93,
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
        "model_size_mb": 42.80,
        "input_resolution": "64x64 RGB",
        "test_accuracy": "99.67%",
        "macro_f1": "0.9972",
        "avg_cpu_latency_ms": 10.94,
        "supports_gradcam": True,
        "gradcam_target_layer": "layer4.1.conv2",
        "description": "Deep residual architecture leveraging identity skip connections to mitigate vanishing gradients. Fine-tuned layer4 + classifier.",
    },
}


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health_check():
    """Fast health check — no model loading."""
    models_status = {}
    for key, path in ARCHITECTURE_CHECKPOINT_MAP.items():
        exists = os.path.exists(path)
        size_bytes = os.path.getsize(path) if exists else 0
        models_status[key] = {
            "checkpoint_path": path,
            "available": exists,
            "size_mb": round(size_bytes / 1_048_576, 2) if exists else 0,
        }

    return jsonify({
        "status": "healthy",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "torch_version": torch.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_available": torch.cuda.is_available(),
        "models": models_status,
        "dataset_ready": os.path.exists("data/dataset_summary.json"),
        "server": "RoadSight AI REST Engine (Flask / PyTorch)",
        "ram_mb": round(_rss_mb(), 1),
    })


@app.route("/api/memory", methods=["GET"])
def memory_status():
    """
    Live RAM diagnostics endpoint.
    Reports current RSS, resident model, and system memory if psutil is available.
    """
    info: dict = {
        "current_model_loaded": predictor_cache.current_model(),
        "rss_mb": round(_rss_mb(), 1),
    }
    if _psutil is not None:
        vm = _psutil.virtual_memory()
        info["system_total_mb"]     = round(vm.total / 1_048_576, 1)
        info["system_available_mb"] = round(vm.available / 1_048_576, 1)
        info["system_used_pct"]     = vm.percent
    return jsonify(info)


@app.route("/api/models", methods=["GET"])
def get_models():
    """Returns all available models with architectural specs and checkpoint availability."""
    models_list = []
    for key, meta in MODEL_METADATA.items():
        chk_path = meta["checkpoint"]
        exists = os.path.exists(chk_path)
        size_mb = round(os.path.getsize(chk_path) / 1_048_576, 2) if exists else 0
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

    Memory lifecycle per request:
      1. Read image bytes → open PIL → delete bytes immediately.
      2. Validate dimensions.
      3. Get cached predictor (loads model if needed, evicts old one first).
      4. predictor.predict(pil_image) → result dict.
      5. finally: delete PIL image, gc.collect().
    """
    if request.method == "OPTIONS":
        return "", 200

    start_time = time.perf_counter()
    ram_before = _rss_mb()

    # --- Model selection ---
    if request.is_json and request.json:
        model_name = request.json.get("model") or request.args.get("model", "custom_cnn")
    else:
        model_name = request.form.get("model") or request.args.get("model", "custom_cnn")

    try:
        model_key = resolve_model_key(model_name)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # --- Image extraction ---
    image_bytes = None
    pil_image   = None
    width = height = 0
    result = None

    try:
        # -- Read bytes once --
        if "image" in request.files:
            file = request.files["image"]
            if not file.filename:
                return jsonify({"error": "No selected file"}), 400

            ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
            if ext not in ALLOWED_EXTENSIONS:
                return jsonify({
                    "error": f"Unsupported format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
                }), 400

            image_bytes = file.read()
            if len(image_bytes) > MAX_IMAGE_BYTES:
                return jsonify({"error": "File exceeds 10 MB limit"}), 413

        elif request.is_json and request.json and "image_base64" in request.json:
            b64 = request.json["image_base64"]
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            try:
                image_bytes = base64.b64decode(b64)
            except Exception as e:
                return jsonify({"error": f"Invalid base64: {e}"}), 400
            if len(image_bytes) > MAX_IMAGE_BYTES:
                return jsonify({"error": "Base64 image exceeds 10 MB"}), 413
        else:
            return jsonify({"error": "No image provided ('image' file or 'image_base64')"}), 400

        # -- Decode PIL (once) then immediately free raw bytes --
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
            pil_image.load()                # fully decode into memory
            pil_image = pil_image.convert("RGB")  # normalize mode
        except Exception as e:
            return jsonify({"error": f"Cannot read image: {e}"}), 400
        finally:
            del image_bytes          # release raw bytes as soon as PIL has decoded
            image_bytes = None

        # -- Validate dimensions --
        width, height = pil_image.size
        if width < MIN_IMAGE_DIM or height < MIN_IMAGE_DIM:
            return jsonify({"error": f"Image too small ({width}×{height}). Min {MIN_IMAGE_DIM}px."}), 400
        if width > MAX_IMAGE_DIM or height > MAX_IMAGE_DIM:
            return jsonify({"error": f"Image too large ({width}×{height}). Max {MAX_IMAGE_DIM}px."}), 400

        # -- Inference --
        try:
            predictor = predictor_cache.get_predictor(model_key)
            result = predictor.predict(pil_image)
        except FileNotFoundError as e:
            return jsonify({"error": str(e), "model_status": "checkpoint_missing"}), 404
        except MemoryError:
            gc.collect()
            return jsonify({"error": "Server out of memory — try again or use a lighter model (custom_cnn)"}), 503
        except Exception as e:
            return jsonify({"error": f"Inference failed: {e}"}), 500

    finally:
        # Always release PIL image even on exception
        if pil_image is not None:
            del pil_image
            pil_image = None
        if image_bytes is not None:
            del image_bytes
        gc.collect()

    if result is None:
        return jsonify({"error": "No result returned from inference"}), 500

    # --- Enrich result ---
    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
    ram_after  = _rss_mb()
    result["inference_latency_ms"] = elapsed_ms
    result["input_dimensions"]     = f"{width}x{height}"
    result["is_valid"]             = True
    result["ram_before_mb"]        = round(ram_before, 1)
    result["ram_after_mb"]         = round(ram_after, 1)

    confidence = result.get("confidence", 0)
    result["low_confidence"] = confidence < 70.0
    if result["low_confidence"]:
        result["confidence_note"] = (
            f"Confidence {confidence:.1f}% — sign may be ambiguous, partially visible, "
            f"or out-of-distribution. Try ResNet18 for better accuracy."
        )
    result["model_recommendation"] = "resnet18"

    print(
        f"[Predict] model={model_key}  conf={confidence:.1f}%  "
        f"latency={elapsed_ms:.0f}ms  RAM={ram_after:.1f}MB",
        flush=True
    )
    return jsonify(result)


# ---------------------------------------------------------------------------
# Dataset endpoints
# ---------------------------------------------------------------------------

@app.route("/api/dataset/summary", methods=["GET"])
def dataset_summary():
    path = "data/dataset_summary.json"
    if not os.path.exists(path):
        return jsonify({"error": "Dataset summary not generated yet"}), 404
    with open(path, "r") as f:
        data = json.load(f)
    return jsonify(data)


@app.route("/api/dataset/sample", methods=["GET"])
def get_dataset_sample():
    class_id = request.args.get("class_id", None)
    base_dirs = ["data/processed/test", "data/processed", "data/raw"]
    target_dir = next((b for b in base_dirs if os.path.exists(b)), None)
    if not target_dir:
        return jsonify({"error": "No dataset directory found"}), 404

    class_names = {}
    if os.path.exists("data/class_names.json"):
        with open("data/class_names.json", "r") as f:
            class_names = json.load(f)

    selected_path = None
    found_class = 0

    if class_id is not None:
        try:
            cid = int(class_id)
            for name in [f"{cid:02d}", str(cid)]:
                c_dir = os.path.join(target_dir, name)
                if os.path.exists(c_dir):
                    files = [f for f in os.listdir(c_dir) if f.lower().endswith((".png", ".jpg", ".ppm"))]
                    if files:
                        selected_path = os.path.join(c_dir, files[0])
                        found_class = cid
                        break
        except ValueError:
            pass

    if not selected_path:
        for preferred in ["14", "01", "13", "17", "25", "00"]:
            p_dir = os.path.join(target_dir, preferred)
            if os.path.exists(p_dir):
                files = [f for f in os.listdir(p_dir) if f.lower().endswith((".png", ".jpg", ".ppm"))]
                if files:
                    selected_path = os.path.join(p_dir, files[0])
                    found_class = int(preferred)
                    break

    if not selected_path:
        return jsonify({"error": "No sample image found"}), 404

    try:
        img = Image.open(selected_path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        buf.close()
        return jsonify({
            "class_id": found_class,
            "class_name": class_names.get(str(found_class), f"Class {found_class}"),
            "image_b64": b64,
            "dimensions": f"{img.width}x{img.height}",
            "filename": os.path.basename(selected_path),
        })
    except Exception as e:
        return jsonify({"error": f"Failed reading sample: {e}"}), 500


@app.route("/api/dataset/samples-gallery", methods=["GET"])
def get_samples_gallery():
    curated = [14, 1, 2, 13, 17, 18, 25, 23, 38, 35, 12, 33]
    class_names = {}
    if os.path.exists("data/class_names.json"):
        with open("data/class_names.json", "r") as f:
            class_names = json.load(f)

    gallery = []
    base_dirs = ["data/processed", "data/raw"]
    target_dir = next((b for b in base_dirs if os.path.exists(b)), None)
    if not target_dir:
        return jsonify({"gallery": []})

    for cid in curated:
        for name in [f"{cid:02d}", str(cid)]:
            c_dir = os.path.join(target_dir, name)
            if os.path.exists(c_dir):
                files = [f for f in os.listdir(c_dir) if f.lower().endswith((".png", ".jpg", ".ppm"))]
                if files:
                    try:
                        img = Image.open(os.path.join(c_dir, files[0])).convert("RGB")
                        thumb = img.copy()
                        thumb.thumbnail((120, 120))
                        buf = io.BytesIO()
                        thumb.save(buf, format="PNG")
                        b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
                        buf.close()
                        gallery.append({
                            "class_id": cid,
                            "class_name": class_names.get(str(cid), f"Class {cid}"),
                            "image_b64": b64,
                            "dimensions": f"{img.width}x{img.height}",
                        })
                    except Exception:
                        pass
                    break

    return jsonify({"gallery": gallery})


# ---------------------------------------------------------------------------
# Experiments / Evaluation / Plots
# ---------------------------------------------------------------------------

@app.route("/api/experiments", methods=["GET"])
def get_experiments():
    csv_path = "experiments/experiment_log.csv"
    if not os.path.exists(csv_path):
        return jsonify({"experiments": []})
    experiments = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            experiments.append(row)
    return jsonify({"experiments": experiments, "total_runs": len(experiments)})


@app.route("/api/evaluation", methods=["GET"])
def get_evaluation():
    plots_dir = "experiments/plots"
    available_plots = []
    if os.path.exists(plots_dir):
        available_plots = [
            f for f in os.listdir(plots_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".svg"))
        ]

    benchmark = [
        {"model_id": "custom_cnn",   "model_name": "Custom CNN",
         "test_accuracy": 98.92, "macro_f1": 0.9895, "val_accuracy": 99.05,
         "latency_ms": 3.82,  "params": "2.32M", "size_mb": 2.98,
         "confusion_matrix_plot": "/plots/custom_cnn_confusion_matrix.png",
         "curves_plot": "/plots/custom_cnn_curves.png"},
        {"model_id": "signnet_lite",  "model_name": "SignNet-Lite",
         "test_accuracy": 98.92, "macro_f1": 0.9895, "val_accuracy": 99.05,
         "latency_ms": 1.97,  "params": "2.32M", "size_mb": 2.98,
         "confusion_matrix_plot": "/plots/custom_cnn_confusion_matrix.png",
         "curves_plot": "/plots/custom_cnn_curves.png"},
        {"model_id": "mobilenet_v2", "model_name": "MobileNetV2 (Phase B)",
         "test_accuracy": 98.40, "macro_f1": 0.9829, "val_accuracy": 98.57,
         "latency_ms": 28.53, "params": "2.28M", "size_mb": 8.93,
         "confusion_matrix_plot": "/plots/mobilenet_v2_confusion_matrix.png",
         "curves_plot": "/plots/mobilenet_v2_curves.png"},
        {"model_id": "resnet18",     "model_name": "ResNet18 (Phase B)",
         "test_accuracy": 99.67, "macro_f1": 0.9972, "val_accuracy": 99.42,
         "latency_ms": 10.94, "params": "11.20M", "size_mb": 42.80,
         "confusion_matrix_plot": "/plots/resnet18_confusion_matrix.png",
         "curves_plot": "/plots/resnet18_curves.png"},
    ]

    return jsonify({
        "benchmark": benchmark,
        "plots": [f"/plots/{p}" for p in available_plots],
        "plots_metadata": {
            "model_accuracy_comparison":    "/plots/model_accuracy_comparison.png",
            "model_f1_comparison":          "/plots/model_f1_comparison.png",
            "inference_latency_comparison": "/plots/inference_latency_comparison.png",
            "parameters_vs_accuracy":       "/plots/parameters_vs_accuracy.png",
            "class_distribution":           "/plots/class_distribution.png",
            "image_dimensions":             "/plots/image_dimensions.png",
            "sample_images":                "/plots/sample_images.png",
        },
    })


@app.route("/plots/<path:filename>")
def serve_plot(filename):
    return send_from_directory("experiments/plots", filename)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[RoadSight AI] Starting on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
