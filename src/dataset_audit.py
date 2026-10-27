import os
import sys
import json
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
CLASS_NAMES_FILE = DATA_DIR / "class_names.json"
SUMMARY_FILE = DATA_DIR / "dataset_summary.json"
PLOTS_DIR = Path("experiments/plots")


def load_class_names():
    if CLASS_NAMES_FILE.exists():
        with open(CLASS_NAMES_FILE, "r") as f:
            return json.load(f)
    return {str(i): f"Class_{i}" for i in range(43)}


def compute_file_hash(filepath):
    """Computes MD5 hash of a file to detect duplicate images."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()


def audit_dataset():
    """
    Performs a thorough dataset audit on data/raw:
    - Measures total images, valid images, corrupted files, and exact duplicates.
    - Measures resolution ranges (min, max, average width/height).
    - Measures class imbalance ratio (max_class / min_class).
    - Writes exact, measured JSON summary to data/dataset_summary.json.
    """
    print("[Dataset Auditor] Starting dataset audit on data/raw...")
    class_names = load_class_names()
    
    seen_hashes = {}
    duplicate_count = 0
    corrupt_count = 0
    valid_count = 0
    total_files = 0
    
    per_class_counts = {}
    widths = []
    heights = []
    channels_set = set()
    sample_images_per_class = {}

    for class_id_str in sorted(class_names.keys(), key=int):
        class_id = int(class_id_str)
        class_dir = RAW_DIR / f"{class_id:02d}"
        
        count_in_class = 0
        if class_dir.exists():
            for img_path in sorted(class_dir.glob("*.*")):
                if img_path.suffix.lower() not in [".png", ".jpg", ".jpeg", ".bmp", ".ppm"]:
                    continue
                
                total_files += 1
                
                # Check for duplicate
                file_hash = compute_file_hash(img_path)
                if file_hash in seen_hashes:
                    duplicate_count += 1
                else:
                    seen_hashes[file_hash] = str(img_path)

                # Verify PIL read & measure dimensions
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                    with Image.open(img_path) as img:
                        w, h = img.size
                        widths.append(w)
                        heights.append(h)
                        mode = img.mode
                        channels_set.add(mode)
                        valid_count += 1
                        count_in_class += 1
                        
                        if class_id_str not in sample_images_per_class:
                            sample_images_per_class[class_id_str] = str(img_path)

                except Exception as e:
                    corrupt_count += 1
                    print(f"[Warning] Corrupt image detected: {img_path} ({e})")

        per_class_counts[class_id_str] = count_in_class

    counts_list = list(per_class_counts.values())
    min_per_class = int(np.min(counts_list)) if counts_list else 0
    max_per_class = int(np.max(counts_list)) if counts_list else 0
    avg_per_class = float(np.mean(counts_list)) if counts_list else 0.0
    imbalance_ratio = round(max_per_class / max(1, min_per_class), 2)

    summary = {
        "dataset_name": "German Traffic Sign Recognition Benchmark (GTSRB) / Local Benchmark",
        "num_classes": len(class_names),
        "total_images": total_files,
        "valid_images": valid_count,
        "corrupted_images": corrupt_count,
        "duplicate_images": duplicate_count,
        "image_channels": list(channels_set),
        "min_images_per_class": min_per_class,
        "max_images_per_class": max_per_class,
        "average_images_per_class": round(avg_per_class, 2),
        "class_imbalance_ratio": imbalance_ratio,
        "resolution_stats": {
            "min_width": int(np.min(widths)) if widths else 0,
            "min_height": int(np.min(heights)) if heights else 0,
            "max_width": int(np.max(widths)) if widths else 0,
            "max_height": int(np.max(heights)) if heights else 0,
            "avg_width": round(float(np.mean(widths)), 2) if widths else 0.0,
            "avg_height": round(float(np.mean(heights)), 2) if heights else 0.0
        },
        "class_distribution": per_class_counts,
        "splits": {
            "train": int(0.70 * valid_count),
            "val": int(0.15 * valid_count),
            "test": valid_count - int(0.70 * valid_count) - int(0.15 * valid_count)
        }
    }

    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[Dataset Auditor] Saved verified dataset summary to {SUMMARY_FILE}")
    print(f" Total Images: {summary['total_images']} | Valid: {summary['valid_images']} | Duplicates: {summary['duplicate_images']}")
    print(f" Resolution Range: {summary['resolution_stats']['min_width']}x{summary['resolution_stats']['min_height']} to {summary['resolution_stats']['max_width']}x{summary['resolution_stats']['max_height']}")
    return summary, sample_images_per_class


def generate_dataset_plots(summary, sample_images_per_class):
    """
    Generates 3 practical, publication-quality dataset visualizations:
    1. Class distribution bar chart
    2. Sample 4x4 image grid with class labels
    3. Scatter plot of image dimensions (width vs height)
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    class_names = load_class_names()

    # 1. Class Distribution Plot
    plt.figure(figsize=(14, 5))
    class_ids = [int(k) for k in summary["class_distribution"].keys()]
    counts = [summary["class_distribution"][str(k)] for k in class_ids]
    
    plt.bar(class_ids, counts, color="#3b82f6", edgecolor="#1e3a8a", alpha=0.85)
    plt.axhline(y=summary["average_images_per_class"], color="#ef4444", linestyle="--", linewidth=1.5, label=f"Mean ({summary['average_images_per_class']:.1f})")
    
    plt.title("Dataset Class Distribution (43 GTSRB Traffic Sign Classes)", fontsize=13, fontweight="bold")
    plt.xlabel("Class ID (0 to 42)", fontsize=10)
    plt.ylabel("Number of Images", fontsize=10)
    plt.grid(axis="y", linestyle=":", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    
    dist_plot_path = PLOTS_DIR / "class_distribution.png"
    plt.savefig(dist_plot_path, dpi=150)
    plt.close()
    print(f"[Dataset Auditor] Saved class distribution plot to {dist_plot_path}")

    # 2. Sample Images Grid (4x4)
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    
    sample_classes = list(sample_images_per_class.keys())[:16]
    for i, class_id_str in enumerate(sample_classes):
        img_path = sample_images_per_class[class_id_str]
        class_name = class_names.get(class_id_str, f"Class {class_id_str}")
        
        try:
            img = Image.open(img_path)
            axes[i].imshow(img)
            axes[i].set_title(f"Class {class_id_str}: {class_name[:18]}", fontsize=8, fontweight="bold")
        except Exception:
            pass
        axes[i].axis("off")

    plt.suptitle("Sample Traffic Sign Images per Category", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    sample_plot_path = PLOTS_DIR / "sample_images.png"
    plt.savefig(sample_plot_path, dpi=150)
    plt.close()
    print(f"[Dataset Auditor] Saved sample image grid plot to {sample_plot_path}")

    # 3. Image Dimensions Scatter Plot
    widths = []
    heights = []
    for class_dir in RAW_DIR.glob("*"):
        if class_dir.is_dir():
            for img_p in class_dir.glob("*.*"):
                try:
                    with Image.open(img_p) as img:
                        w, h = img.size
                        widths.append(w)
                        heights.append(h)
                except Exception:
                    pass

    plt.figure(figsize=(7, 5))
    plt.scatter(widths, heights, alpha=0.4, c="#10b981", edgecolors="none", s=25)
    plt.axvline(x=64, color="#ef4444", linestyle="--", label="Target Input Size (64x64)")
    plt.axhline(y=64, color="#ef4444", linestyle="--")
    
    plt.title("Raw Image Dimension Scatter Plot (Width vs Height)", fontsize=11, fontweight="bold")
    plt.xlabel("Image Width (px)", fontsize=10)
    plt.ylabel("Image Height (px)", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    dim_plot_path = PLOTS_DIR / "image_dimensions.png"
    plt.savefig(dim_plot_path, dpi=150)
    plt.close()
    print(f"[Dataset Auditor] Saved image dimension plot to {dim_plot_path}")


if __name__ == "__main__":
    summary, sample_imgs = audit_dataset()
    generate_dataset_plots(summary, sample_imgs)
