import os
import sys
import json
import time
import argparse
from pathlib import Path

# Fix Windows DLL search path for PyTorch C++ binaries
if sys.platform == "win32":
    torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
    if os.path.exists(torch_lib):
        try:
            os.add_dll_directory(torch_lib)
        except Exception:
            pass

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score

from src.dataset import get_dataloaders
from src.models.factory import get_model, get_checkpoint_for_model
from src.utils import log_experiment, get_model_size_mb, count_parameters


def load_class_names():
    class_file = Path("data/class_names.json")
    if class_file.exists():
        with open(class_file, "r") as f:
            return json.load(f)
    return {str(i): f"Class_{i}" for i in range(43)}


def benchmark_cpu_latency(model, device, img_dim=64, num_warmup=10, num_runs=100):
    """
    Measures CPU single-image inference latency with proper warm-up.
    """
    model.eval()
    dummy_input = torch.randn(1, 3, img_dim, img_dim).to(device)

    # Warm-up runs
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)

    # Timed runs
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            t_start = time.perf_counter()
            _ = model(dummy_input)
            t_end = time.perf_counter()
            latencies.append((t_end - t_start) * 1000.0)  # ms

    return float(np.mean(latencies))


def evaluate_model(model_name="custom_cnn", img_dim=64, exp_id="EXP_01", notes="Baseline evaluation", ckpt_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Evaluation Engineer] Evaluating model '{model_name}' on device: {device}", flush=True)

    if ckpt_path and os.path.exists(ckpt_path):
        checkpoint_path = ckpt_path
    else:
        checkpoint_path = get_checkpoint_for_model(model_name)

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}' for '{model_name}'.")

    print(f"[Evaluation Engineer] Loading checkpoint from '{checkpoint_path}'", flush=True)
    input_size = (img_dim, img_dim)
    _, _, test_loader, _, _, n_test = get_dataloaders(batch_size=32, input_size=input_size)

    # Instantiate exact model architecture before loading weights
    model = get_model(model_name=model_name, num_classes=43, pretrained=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    total_params, trainable_params = count_parameters(model)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    # Compute metrics
    acc = accuracy_score(all_targets, all_preds) * 100.0
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)
    
    # Benchmarked CPU inference latency
    avg_latency = benchmark_cpu_latency(model, device, img_dim=img_dim)
    model_size_mb = get_model_size_mb(checkpoint_path)

    print("\n" + "=" * 50, flush=True)
    print(f" EVALUATION REPORT: {model_name.upper()}", flush=True)
    print("=" * 50, flush=True)
    print(f" Test Accuracy      : {acc:.2f}%", flush=True)
    print(f" Macro Precision    : {precision:.4f}", flush=True)
    print(f" Macro Recall       : {recall:.4f}", flush=True)
    print(f" Macro F1-Score     : {f1:.4f}", flush=True)
    print(f" Avg CPU Latency    : {avg_latency:.2f} ms / image", flush=True)
    print(f" Total Parameters   : {total_params:,}", flush=True)
    print(f" Trainable Params   : {trainable_params:,}", flush=True)
    print(f" Model Size on Disk : {model_size_mb:.2f} MB", flush=True)
    print("=" * 50 + "\n", flush=True)

    # Error analysis
    class_names = load_class_names()
    errors = {}
    for true_cls, pred_cls in zip(all_targets, all_preds):
        if true_cls != pred_cls:
            pair = (int(true_cls), int(pred_cls))
            errors[pair] = errors.get(pair, 0) + 1
    
    sorted_errors = sorted(errors.items(), key=lambda x: x[1], reverse=True)[:5]
    print("[Error Analysis] Top 5 Misclassified Sign Pairs:", flush=True)
    for (t_cls, p_cls), count in sorted_errors:
        t_name = class_names.get(str(t_cls), f"Class {t_cls}")
        p_name = class_names.get(str(p_cls), f"Class {p_cls}")
        print(f"  Actual: [{t_cls:02d}] {t_name:<30} -> Predicted: [{p_cls:02d}] {p_name:<30} (Count: {count})", flush=True)
    print()

    # Generate Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    cm_plot_path = f"experiments/plots/{model_name}_confusion_matrix.png"
    Path(cm_plot_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    if HAS_SEABORN:
        sns.heatmap(cm, annot=False, cmap="Blues", fmt="d")
    else:
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
    plt.title(f"Confusion Matrix — {model_name.upper()} (Test Acc: {acc:.1f}%)")
    plt.xlabel("Predicted Class")
    plt.ylabel("Ground Truth Class")
    plt.tight_layout()
    plt.savefig(cm_plot_path, dpi=150)
    plt.close()
    print(f"[Evaluation Engineer] Confusion matrix heatmap saved to {cm_plot_path}", flush=True)

    # Log into experiment table
    exp_record = {
        "exp_id": exp_id,
        "model_name": model_name,
        "input_size": f"{img_dim}x{img_dim}",
        "augmentation": "Rot(12deg)+ColorJitter",
        "lr": 0.001,
        "batch_size": 32,
        "epochs": checkpoint.get("epoch", 5),
        "val_acc": round(checkpoint.get("val_acc", 0.0), 2),
        "test_acc": round(acc, 2),
        "macro_f1": round(f1, 4),
        "inference_time_ms": round(avg_latency, 2),
        "model_size_mb": round(model_size_mb, 2),
        "notes": notes
    }
    log_experiment(exp_record)
    return exp_record, sorted_errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Trained Road Sign Classifier")
    parser.add_argument("--model", type=str, default="custom_cnn", help="Model architecture")
    parser.add_argument("--img_dim", type=int, default=64, help="Input image dimension")
    parser.add_argument("--exp_id", type=str, default="EXP_01", help="Experiment ID for tracking")
    parser.add_argument("--notes", type=str, default="Baseline evaluation", help="Notes for log")
    parser.add_argument("--ckpt", type=str, default=None, help="Explicit checkpoint path")

    args = parser.parse_args()
    evaluate_model(model_name=args.model, img_dim=args.img_dim, exp_id=args.exp_id, notes=args.notes, ckpt_path=args.ckpt)
