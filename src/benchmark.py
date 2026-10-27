import os
import sys
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

import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataset import get_dataloaders
from src.models.factory import get_model
from src.utils import set_seed, count_parameters


def run_benchmark(model_name="custom_cnn", freeze_backbone=True, unfreeze_top=0):
    print(f"[Diagnostic Benchmark] Starting controlled CPU performance benchmark for '{model_name}'...", flush=True)
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    t_start_load = time.time()
    train_loader, val_loader, test_loader, n_train, n_val, n_test = get_dataloaders(
        batch_size=64, input_size=(64, 64), preload=True
    )
    load_duration = time.time() - t_start_load
    print(f"  Dataset Loading & Pre-caching in RAM: {load_duration:.2f}s (Train={n_train:,}, Val={n_val:,}, Test={n_test:,})", flush=True)

    model = get_model(model_name=model_name, num_classes=43, pretrained=True, freeze_backbone=freeze_backbone).to(device)

    if unfreeze_top > 0 and hasattr(model, "unfreeze_layer4"):
        model.unfreeze_layer4()
    elif unfreeze_top > 0 and hasattr(model, "unfreeze_top_blocks"):
        model.unfreeze_top_blocks(n_blocks=unfreeze_top)

    total_params, trainable_params = count_parameters(model)
    print(f"  Model Parameters: Total={total_params:,} | Trainable={trainable_params:,}", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    model.train()
    batch_times = []
    
    t_first_batch_start = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader, start=1):
        b_start = time.time()
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        b_time = (time.time() - b_start) * 1000.0  # ms
        batch_times.append(b_time)

        if batch_idx == 1:
            first_batch_time = time.time() - t_first_batch_start
            print(f"  First Batch Compute Time: {first_batch_time * 1000.0:.2f} ms", flush=True)

        if batch_idx >= 50:
            break

    avg_batch_ms = float(sum(batch_times)) / len(batch_times)
    total_train_batches = len(train_loader)
    est_epoch_train_s = (avg_batch_ms * total_train_batches) / 1000.0

    # Validation speed benchmark
    t_val_start = time.time()
    model.eval()
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            _ = model(inputs)
    val_duration = time.time() - t_val_start
    est_epoch_total_s = est_epoch_train_s + val_duration
    est_5epochs_s = (est_epoch_total_s * 5.0) + load_duration

    print("\n" + "=" * 55, flush=True)
    print(f" BENCHMARK RESULTS ({model_name.upper()}) & TIME ESTIMATION", flush=True)
    print("=" * 55, flush=True)
    print(f" Model Architecture      : {model_name}", flush=True)
    print(f" Total Parameters        : {total_params:,}", flush=True)
    print(f" Trainable Parameters    : {trainable_params:,}", flush=True)
    print(f" Pre-caching in RAM Time : {load_duration:.2f} s", flush=True)
    print(f" First Batch Time        : {first_batch_time * 1000.0:.2f} ms", flush=True)
    print(f" Avg Batch Time (50 runs): {avg_batch_ms:.2f} ms / batch", flush=True)
    print(f" Training Step Time/epoch: {est_epoch_train_s:.2f} s", flush=True)
    print(f" Validation Pass Time    : {val_duration:.2f} s", flush=True)
    print(f" Estimated 1 Epoch Time  : {est_epoch_total_s:.2f} s", flush=True)
    print(f" Estimated 5 Epochs Total: {est_5epochs_s:.2f} s ({est_5epochs_s/60.0:.2f} mins)", flush=True)
    print("=" * 55 + "\n", flush=True)

    return {
        "model_name": model_name,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "load_time_s": load_duration,
        "first_batch_ms": first_batch_time * 1000.0,
        "avg_batch_ms": avg_batch_ms,
        "train_epoch_s": est_epoch_train_s,
        "val_time_s": val_duration,
        "total_epoch_s": est_epoch_total_s,
        "est_5epochs_s": est_5epochs_s
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="custom_cnn")
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument("--unfreeze_top", type=int, default=0)
    args = parser.parse_args()
    run_benchmark(model_name=args.model, freeze_backbone=args.freeze_backbone, unfreeze_top=args.unfreeze_top)
