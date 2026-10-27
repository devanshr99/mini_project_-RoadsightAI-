import os
import random
import json
import csv
from pathlib import Path
import numpy as np
import torch


def set_seed(seed=42):
    """
    Enforces deterministic reproducibility across random, numpy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[Utils] Global random seed set to {seed}")


def save_checkpoint(model, optimizer, epoch, val_loss, val_acc, path):
    """
    Saves PyTorch model state dictionary, optimizer state, and training metadata.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "val_loss": val_loss,
        "val_acc": val_acc
    }
    torch.save(checkpoint, path)
    print(f"[Utils] Checkpoint saved: {path} (Val Acc: {val_acc:.2f}%)")


def count_parameters(model):
    """
    Returns total parameters and trainable parameters count.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model_size_mb(filepath):
    """
    Calculates file footprint of model checkpoint in MB.
    """
    if os.path.exists(filepath):
        return os.path.getsize(filepath) / (1024 * 1024)
    return 0.0


def log_experiment(exp_data, log_path="experiments/experiment_log.csv"):
    """
    Appends an experiment summary entry into the experiment tracking CSV.
    """
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    headers = [
        "exp_id", "model_name", "input_size", "augmentation", "lr",
        "batch_size", "epochs", "val_acc", "test_acc", "macro_f1",
        "inference_time_ms", "model_size_mb", "notes"
    ]
    
    file_exists = log_file.exists()
    with open(log_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(exp_data)
    print(f"[Utils] Logged experiment {exp_data.get('exp_id')} to {log_path}")
