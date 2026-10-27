import os
import sys
import json
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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path

# Mean and Std for Normalization (Standard ImageNet / RGB stats)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def get_transforms(input_size=(64, 64), is_training=True):
    """
    Builds data preprocessing and augmentation pipeline.
    
    VIVA NOTE ON AUGMENTATION:
    We use RandomRotation (+/- 12 deg), ColorJitter (brightness/contrast), and Shift.
    We STRICTLY EXCLUDE horizontal/vertical flips because directional traffic signs
    (e.g., 'Turn Left', 'Keep Right') change their semantic meaning when flipped!
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomRotation(degrees=12),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
        ])


def preprocess_image(image, input_size=(64, 64), device=None):
    """
    Unified, reusable preprocessing function for inference and Grad-CAM explainability.
    
    Guarantees:
    - Input is converted to RGB PIL Image
    - Resized strictly to input_size (default: 64x64)
    - Normalized with training dataset statistics (NORM_MEAN, NORM_STD)
    - Batch dimension added -> shape [1, 3, 64, 64]
    - Explicitly verifies input tensor shape before returning
    """
    if not isinstance(image, Image.Image):
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        else:
            image = Image.fromarray(image)

    # Convert to RGB (handles RGBA, grayscale, palette, etc.)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize to exact dimensions (64, 64)
    image_resized = image.resize(input_size, Image.BILINEAR)

    # Transform and normalize
    transform_pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])
    tensor = transform_pipeline(image_resized).unsqueeze(0)  # Shape: [1, 3, H, W]

    # Verify tensor shape
    expected_shape = (1, 3, input_size[0], input_size[1])
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"Preprocessed tensor shape mismatch: expected {expected_shape}, but got {tuple(tensor.shape)}"
        )

    if device is not None:
        tensor = tensor.to(device)

    return tensor


class RoadSignDataset(Dataset):
    """
    Custom PyTorch Dataset for Road Sign Image Classification.
    Loads pre-resized images from data/processed/ or data/raw/.
    Supports in-memory caching (preload=True) for high-performance CPU training.
    """
    def __init__(self, samples, transform=None, target_size=(64, 64), preload=True):
        self.samples = samples
        self.transform = transform
        self.preload = preload
        self.cached_images = []

        if self.preload:
            print(f"[Dataset] Pre-caching {len(self.samples):,} images into RAM ({target_size[0]}x{target_size[1]})...", flush=True)
            for img_path, _ in self.samples:
                try:
                    with Image.open(img_path) as img:
                        resized_img = img.convert("RGB").resize(target_size, Image.BILINEAR)
                        self.cached_images.append(resized_img)
                except Exception:
                    self.cached_images.append(Image.new("RGB", target_size, (0, 0, 0)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        if self.preload and idx < len(self.cached_images):
            image = self.cached_images[idx]
        else:
            image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def load_dataset_splits(data_dir="data", seed=42, max_samples=None):
    """
    Scans processed (or raw) dataset directory and builds deterministic train, val, test split tuples.
    If max_samples is provided, caps total samples for fast sanity check experiments.
    """
    base_dir = Path(data_dir)
    target_dir = base_dir / "processed" if (base_dir / "processed").exists() else base_dir / "raw"
    all_samples = []

    for class_dir in sorted(target_dir.glob("*")):
        if class_dir.is_dir():
            try:
                class_id = int(class_dir.name)
            except ValueError:
                continue

            for img_path in sorted(class_dir.glob("*.*")):
                if img_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".ppm", ".bmp"]:
                    all_samples.append((str(img_path), class_id))

    if not all_samples:
        raise RuntimeError(f"No image files found in {target_dir}. Please run src/download_dataset.py first.")

    # Shuffle deterministically
    import random
    rng = random.Random(seed)
    rng.shuffle(all_samples)

    if max_samples and max_samples < len(all_samples):
        all_samples = all_samples[:max_samples]

    total = len(all_samples)
    train_end = int(0.70 * total)
    val_end = int(0.85 * total)

    train_samples = all_samples[:train_end]
    val_samples = all_samples[train_end:val_end]
    test_samples = all_samples[val_end:]

    return train_samples, val_samples, test_samples


def get_dataloaders(data_dir="data", batch_size=32, input_size=(64, 64), num_workers=0, max_samples=None, preload=True):
    """
    Constructs PyTorch DataLoaders for Train, Validation, and Test sets.
    Default preload=True avoids redundant disk I/O per epoch.
    """
    train_samples, val_samples, test_samples = load_dataset_splits(data_dir=data_dir, max_samples=max_samples)

    train_dataset = RoadSignDataset(train_samples, transform=get_transforms(input_size, is_training=True), target_size=input_size, preload=preload)
    val_dataset = RoadSignDataset(val_samples, transform=get_transforms(input_size, is_training=False), target_size=input_size, preload=preload)
    test_dataset = RoadSignDataset(test_samples, transform=get_transforms(input_size, is_training=False), target_size=input_size, preload=preload)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, len(train_samples), len(val_samples), len(test_samples)
