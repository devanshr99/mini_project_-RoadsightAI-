import os
import shutil
from pathlib import Path

GTSRB_SRC_DIR = Path("data/gtsrb_test/gtsrb/GTSRB/Training")
RAW_DIR = Path("data/raw")


def ingest_gtsrb():
    """
    Ingests official 26,640 GTSRB training images from data/gtsrb_test into data/raw/
    mapping '00000' -> '00', '00001' -> '01', ..., '00042' -> '42'.
    """
    if not GTSRB_SRC_DIR.exists():
        print(f"[Ingest] GTSRB source directory '{GTSRB_SRC_DIR}' not found.")
        return

    print("[Ingest] Ingesting official GTSRB dataset images into data/raw/ ...")
    
    # Remove synthetic placeholders
    if RAW_DIR.exists():
        shutil.rmtree(RAW_DIR)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    total_ingested = 0
    for class_dir in sorted(GTSRB_SRC_DIR.glob("*")):
        if class_dir.is_dir():
            try:
                class_id = int(class_dir.name)
            except ValueError:
                continue

            target_class_dir = RAW_DIR / f"{class_id:02d}"
            target_class_dir.mkdir(parents=True, exist_ok=True)

            for img_file in class_dir.glob("*.ppm"):
                shutil.copy2(img_file, target_class_dir / img_file.name)
                total_ingested += 1

    print(f"[Ingest] Ingestion complete: {total_ingested} official GTSRB images copied into {RAW_DIR}")


if __name__ == "__main__":
    ingest_gtsrb()
