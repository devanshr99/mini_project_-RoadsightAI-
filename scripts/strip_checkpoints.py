"""
Strip optimizer states and unnecessary keys from trained checkpoints.
Reduces ResNet18: 128MB -> ~43MB, MobileNetV2: 26MB -> ~9MB.
Run once locally; commit the resulting .pth files.

Usage:
    python scripts/strip_checkpoints.py
"""
import os
import sys
import gc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch

CHECKPOINTS = [
    "checkpoints/custom_cnn_best.pth",
    "checkpoints/mobilenet_v2_best.pth",
    "checkpoints/resnet18_best.pth",
    "checkpoints/signnet_lite_best.pth",
]

KEEP_KEYS = {"model_state_dict", "epoch", "val_acc", "num_classes"}

def strip_checkpoint(path: str) -> None:
    before_mb = os.path.getsize(path) / 1_048_576
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if "model_state_dict" not in ckpt:
        print(f"  SKIP {path} — no model_state_dict key")
        return

    # Build a minimal checkpoint with only the essential keys
    stripped = {k: v for k, v in ckpt.items() if k in KEEP_KEYS}
    del ckpt
    gc.collect()

    torch.save(stripped, path)
    del stripped
    gc.collect()

    after_mb = os.path.getsize(path) / 1_048_576
    print(f"  {os.path.basename(path)}: {before_mb:.2f} MB -> {after_mb:.2f} MB  (saved {before_mb - after_mb:.2f} MB)")

if __name__ == "__main__":
    print("Stripping optimizer states from checkpoints...")
    for ckpt_path in CHECKPOINTS:
        if os.path.exists(ckpt_path):
            strip_checkpoint(ckpt_path)
        else:
            print(f"  NOT FOUND: {ckpt_path}")
    print("Done. Commit the updated .pth files.")
