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

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from src.dataset import get_dataloaders
from src.models.factory import get_model
from src.utils import set_seed, save_checkpoint, count_parameters, log_experiment


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch=1, total_epochs=10):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    num_batches = len(dataloader)
    log_interval = max(1, num_batches // 5)  # Log 5 times per epoch

    print(f"\nEpoch {epoch}/{total_epochs}", flush=True)

    start_t = time.time()

    for batch_idx, (inputs, targets) in enumerate(dataloader, start=1):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == targets.data).item()
        total += targets.size(0)

        if batch_idx % log_interval == 0 or batch_idx == num_batches:
            batch_acc = (correct / total) * 100.0
            batch_loss = running_loss / total
            elapsed = time.time() - start_t
            pct = int(round((batch_idx / num_batches) * 100.0))
            print(f"[{pct:3d}%] Batch {batch_idx}/{num_batches} | Loss: {batch_loss:.4f} | Acc: {batch_acc:.2f}% | Elapsed: {elapsed:.1f}s", flush=True)

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == targets.data).item()
            total += targets.size(0)

    val_loss = running_loss / total
    val_acc = (correct / total) * 100.0
    return val_loss, val_acc


def plot_learning_curves(history, save_path):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["train_loss"], label="Train Loss", color="crimson", linewidth=2)
    ax1.plot(history["val_loss"], label="Val Loss", color="royalblue", linewidth=2, linestyle="--")
    ax1.set_title("Cross-Entropy Loss Curve")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(history["train_acc"], label="Train Acc (%)", color="crimson", linewidth=2)
    ax2.plot(history["val_acc"], label="Val Acc (%)", color="royalblue", linewidth=2, linestyle="--")
    ax2.set_title("Accuracy Curve (%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[ML Engineer] Training curves saved to {save_path}", flush=True)


def train_model(model_name="custom_cnn", epochs=5, batch_size=64, lr=0.001, lr_backbone=0.0001, img_dim=64, freeze_backbone=True, unfreeze_top=0, max_samples=None):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if max_samples:
        print(f"[ML Engineer] Starting PIPELINE SANITY CHECK RUN (max_samples={max_samples}) on device: {device}", flush=True)
    else:
        print(f"[ML Engineer] Starting FULL TRAINING RUN for '{model_name}' on device: {device}", flush=True)

    input_size = (img_dim, img_dim)
    train_loader, val_loader, test_loader, n_train, n_val, n_test = get_dataloaders(
        batch_size=batch_size, input_size=input_size, max_samples=max_samples
    )

    print(f"[ML Engineer] Dataset split sizes: Train={n_train:,} | Val={n_val:,} | Test={n_test:,}", flush=True)

    model = get_model(model_name=model_name, num_classes=43, pretrained=True, freeze_backbone=freeze_backbone).to(device)

    # Configure unfreezing if fine-tuning requested
    if unfreeze_top > 0 and hasattr(model, "unfreeze_layer4"):
        model.unfreeze_layer4()
        print(f"[Transfer Learning Strategy] Selective fine-tuning enabled for ResNet-18 (layer4 + head unfrozen)", flush=True)
    elif unfreeze_top > 0 and hasattr(model, "unfreeze_top_blocks"):
        model.unfreeze_top_blocks(n_blocks=unfreeze_top)
        print(f"[Transfer Learning Strategy] Selective fine-tuning enabled for MobileNetV2 (top {unfreeze_top} blocks + head unfrozen)", flush=True)
    elif freeze_backbone and model_name not in ["custom_cnn", "signnet_lite"]:
        print(f"[Transfer Learning Strategy] Phase A: Frozen pre-trained backbone, training classification head only", flush=True)

    total_params, trainable_params = count_parameters(model)
    print(f"[ML Engineer] Model parameters: Total={total_params:,}, Trainable={trainable_params:,}", flush=True)

    criterion = nn.CrossEntropyLoss()

    # Parameter grouping for differential learning rates during fine-tuning
    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "fc" in name or "classifier" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = [
        {"params": head_params, "lr": lr}
    ]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr_backbone})

    optimizer = optim.AdamW(param_groups, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    
    ckpt_name = "signnet_lite_best.pth" if model_name in ["custom_cnn", "signnet_lite"] else f"{model_name}_best.pth"
    checkpoint_path = f"checkpoints/{ckpt_name}"
    plot_path = f"experiments/plots/{model_name}_curves.png"

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch=epoch, total_epochs=epochs)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        epoch_duration = time.time() - epoch_start
        print(f"\nTrain Loss: {train_loss:.4f}", flush=True)
        print(f"Train Accuracy: {train_acc:.2f}%", flush=True)
        print(f"Val Loss: {val_loss:.4f}", flush=True)
        print(f"Val Accuracy: {val_acc:.2f}%", flush=True)
        print(f"Epoch Time: {epoch_duration:.2f}s\n", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, checkpoint_path)

    total_time = time.time() - start_time
    print(f"[ML Engineer] Training finished in {total_time:.2f}s. Best Val Acc: {best_val_acc:.2f}%", flush=True)

    # Overfitting analysis
    final_train_acc = history["train_acc"][-1]
    gap = final_train_acc - best_val_acc
    print(f"[Overfitting Analysis] Final Train Acc: {final_train_acc:.2f}% | Best Val Acc: {best_val_acc:.2f}% | Gap: {gap:.2f}%", flush=True)
    if gap > 15.0:
        print("[Overfitting Analysis] Note: Moderate overfitting detected (train-val accuracy gap > 15%).", flush=True)
    else:
        print("[Overfitting Analysis] Model generalization is stable.", flush=True)

    plot_learning_curves(history, plot_path)
    return checkpoint_path, best_val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Traffic Sign Classifier")
    parser.add_argument("--model", type=str, default="custom_cnn", help="Model architecture")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial classification head learning rate")
    parser.add_argument("--lr_backbone", type=float, default=0.0001, help="Backbone learning rate for fine-tuning")
    parser.add_argument("--img_dim", type=int, default=64, help="Input image dimension")
    parser.add_argument("--freeze_backbone", action="store_true", default=True, help="Freeze backbone layers")
    parser.add_argument("--unfreeze_top", type=int, default=0, help="Number of top layers/blocks to unfreeze for fine-tuning")
    parser.add_argument("--max_samples", type=int, default=None, help="Cap total samples for fast sanity check run")

    args = parser.parse_args()
    train_model(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_backbone=args.lr_backbone,
        img_dim=args.img_dim,
        freeze_backbone=args.freeze_backbone,
        unfreeze_top=args.unfreeze_top,
        max_samples=args.max_samples
    )
