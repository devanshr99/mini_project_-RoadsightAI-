import os
import sys
import time
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
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import matplotlib.pyplot as plt

from src.train import train_model
from src.evaluate import evaluate_model


def run_model_comparison_pipeline():
    print("=================================================================", flush=True)
    print("      TRANSFER LEARNING & MODEL COMPARISON SUITE (AGENT 4)", flush=True)
    print("=================================================================\n", flush=True)

    experiments = [
        {
            "exp_id": "EXP-TL-01",
            "model_name": "mobilenet_v2",
            "freeze_backbone": True,
            "unfreeze_top": 0,
            "epochs": 5,
            "lr": 0.001,
            "lr_backbone": 0.0001,
            "notes": "MobileNetV2 Phase A: Frozen ImageNet backbone"
        },
        {
            "exp_id": "EXP-TL-02",
            "model_name": "mobilenet_v2",
            "freeze_backbone": False,
            "unfreeze_top": 4,
            "epochs": 5,
            "lr": 0.001,
            "lr_backbone": 0.0001,
            "notes": "MobileNetV2 Phase B: Unfreeze top 4 residual blocks"
        },
        {
            "exp_id": "EXP-TL-03",
            "model_name": "resnet18",
            "freeze_backbone": True,
            "unfreeze_top": 0,
            "epochs": 5,
            "lr": 0.001,
            "lr_backbone": 0.0001,
            "notes": "ResNet18 Phase A: Frozen ImageNet backbone"
        },
        {
            "exp_id": "EXP-TL-04",
            "model_name": "resnet18",
            "freeze_backbone": False,
            "unfreeze_top": 1,
            "epochs": 5,
            "lr": 0.001,
            "lr_backbone": 0.0001,
            "notes": "ResNet18 Phase B: Unfreeze layer4 + head"
        }
    ]

    for exp in experiments:
        print(f"\n>>> Running Experiment: {exp['exp_id']} ({exp['notes']}) <<<", flush=True)
        ckpt_path, best_val_acc = train_model(
            model_name=exp["model_name"],
            epochs=exp["epochs"],
            batch_size=64,
            lr=exp["lr"],
            lr_backbone=exp["lr_backbone"],
            img_dim=64,
            freeze_backbone=exp["freeze_backbone"],
            unfreeze_top=exp["unfreeze_top"]
        )

        print(f"\n>>> Evaluating Experiment: {exp['exp_id']} on Held-Out Test Set <<<", flush=True)
        evaluate_model(
            model_name=exp["model_name"],
            img_dim=64,
            exp_id=exp["exp_id"],
            notes=exp["notes"],
            ckpt_path=ckpt_path
        )

    print("\n[Comparison Pipeline] Generating comparison visualizations...", flush=True)
    generate_comparison_plots()
    print("[Comparison Pipeline] All experiments completed successfully!", flush=True)


def generate_comparison_plots():
    log_file = Path("experiments/experiment_log.csv")
    if not log_file.exists():
        print("[Plotter Warning] experiment_log.csv not found.")
        return

    df = pd.read_csv(log_file)
    # Deduplicate keeping latest by exp_id / model
    df = df.drop_duplicates(subset=["exp_id"], keep="last")

    plots_dir = Path("experiments/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Model Accuracy Comparison
    plt.figure(figsize=(9, 5))
    bars = plt.bar(df["exp_id"] + "\n(" + df["model_name"] + ")", df["test_acc"], color="#2b5c8f", width=0.55)
    plt.title("Test Accuracy Comparison Across Architectures (%)", fontsize=13, fontweight="bold")
    plt.ylabel("Test Accuracy (%)")
    plt.ylim(0, 105)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.2f}%", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(plots_dir / "model_accuracy_comparison.png", dpi=150)
    plt.close()

    # 2. Model F1 Comparison
    plt.figure(figsize=(9, 5))
    bars = plt.bar(df["exp_id"] + "\n(" + df["model_name"] + ")", df["macro_f1"], color="#2e8b57", width=0.55)
    plt.title("Macro F1-Score Comparison Across Models", fontsize=13, fontweight="bold")
    plt.ylabel("Macro F1-Score")
    plt.ylim(0, 1.1)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.4f}", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(plots_dir / "model_f1_comparison.png", dpi=150)
    plt.close()

    # 3. Parameters vs Accuracy
    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(df["model_size_mb"], df["test_acc"], s=180, c=range(len(df)), cmap="viridis", alpha=0.85, edgecolors="black")
    for i, txt in enumerate(df["exp_id"] + ": " + df["model_name"]):
        plt.annotate(txt, (df["model_size_mb"].iloc[i] + 0.3, df["test_acc"].iloc[i] - 0.5), fontsize=10, fontweight="bold")
    plt.title("Model Checkpoint Size (MB) vs Test Accuracy (%)", fontsize=13, fontweight="bold")
    plt.xlabel("Model Disk Size (MB)")
    plt.ylabel("Test Accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "parameters_vs_accuracy.png", dpi=150)
    plt.close()

    # 4. Inference Latency Comparison
    plt.figure(figsize=(9, 5))
    bars = plt.bar(df["exp_id"] + "\n(" + df["model_name"] + ")", df["inference_time_ms"], color="#d9534f", width=0.55)
    plt.title("Single-Image CPU Inference Latency (ms)", fontsize=13, fontweight="bold")
    plt.ylabel("CPU Latency per Image (ms)")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"{yval:.2f} ms", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(plots_dir / "inference_latency_comparison.png", dpi=150)
    plt.close()

    print("[Comparison Pipeline] Visualizations updated in experiments/plots/", flush=True)


if __name__ == "__main__":
    run_model_comparison_pipeline()
