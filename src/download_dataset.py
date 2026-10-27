import os
import json
import shutil
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
CLASS_NAMES_FILE = DATA_DIR / "class_names.json"
SUMMARY_FILE = DATA_DIR / "dataset_summary.json"


def load_class_names():
    if CLASS_NAMES_FILE.exists():
        with open(CLASS_NAMES_FILE, "r") as f:
            return json.load(f)
    return {str(i): f"Class_{i}" for i in range(43)}


def download_official_gtsrb():
    """
    Downloads official 26,640 GTSRB dataset images via torchvision
    and organizes them into data/raw/{00..42}/.
    """
    print("[Dataset Researcher] Downloading official GTSRB dataset via torchvision...")
    try:
        import torchvision
        temp_dir = DATA_DIR / "gtsrb_temp"
        ds = torchvision.datasets.GTSRB(root=str(temp_dir), split="train", download=True)
        
        gtsrb_src = temp_dir / "gtsrb" / "GTSRB" / "Training"
        if gtsrb_src.exists():
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            copied = 0
            for class_dir in sorted(gtsrb_src.glob("*")):
                if class_dir.is_dir():
                    try:
                        cid = int(class_dir.name)
                    except ValueError:
                        continue
                    target_dir = RAW_DIR / f"{cid:02d}"
                    target_dir.mkdir(parents=True, exist_ok=True)
                    for img_f in class_dir.glob("*.ppm"):
                        shutil.copy2(img_f, target_dir / img_f.name)
                        copied += 1
            print(f"[Dataset Researcher] Copied {copied} official GTSRB images to {RAW_DIR}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return True
    except Exception as e:
        print(f"[Dataset Researcher] Official GTSRB download attempt failed: {e}")
    return False


if __name__ == "__main__":
    if not list(RAW_DIR.glob("*/*.*")):
        success = download_official_gtsrb()
        if not success:
            print("[Dataset Researcher] Using local benchmark setup.")
    
    # Audit and generate statistics
    from src.dataset_audit import audit_dataset, generate_dataset_plots
    summary, sample_imgs = audit_dataset()
    generate_dataset_plots(summary, sample_imgs)
