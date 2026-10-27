import os
import json
import pytest
import torch
from PIL import Image
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataset import get_dataloaders, RoadSignDataset, load_dataset_splits, get_transforms


def test_class_names_mapping_integrity():
    class_names_file = Path("data/class_names.json")
    assert class_names_file.exists(), "data/class_names.json must exist"

    with open(class_names_file, "r") as f:
        mapping = json.load(f)

    assert len(mapping) == 43, f"Expected 43 classes, got {len(mapping)}"
    for i in range(43):
        assert str(i) in mapping, f"Class ID {i} missing from mapping"
        assert isinstance(mapping[str(i)], str), f"Class label for {i} must be a string"
        assert len(mapping[str(i)]) > 0, f"Class label for {i} must not be empty"


def test_load_dataset_splits():
    train_s, val_s, test_s = load_dataset_splits()
    assert len(train_s) > 0, "Train split should not be empty"
    assert len(val_s) > 0, "Validation split should not be empty"
    assert len(test_s) > 0, "Test split should not be empty"

    # Verify no data leakage across splits
    train_set = set(x[0] for x in train_s)
    val_set = set(x[0] for x in val_s)
    test_set = set(x[0] for x in test_s)

    assert len(train_set.intersection(val_set)) == 0, "Data leakage detected between train and val splits!"
    assert len(train_set.intersection(test_set)) == 0, "Data leakage detected between train and test splits!"
    assert len(val_set.intersection(test_set)) == 0, "Data leakage detected between val and test splits!"


def test_dataloaders_shape():
    train_loader, val_loader, test_loader, n_train, n_val, n_test = get_dataloaders(batch_size=8, input_size=(64, 64))
    images, labels = next(iter(train_loader))

    assert images.shape == (8, 3, 64, 64), f"Expected batch tensor shape (8, 3, 64, 64), got {images.shape}"
    assert labels.shape == (8,), f"Expected labels shape (8,), got {labels.shape}"
    assert labels.dtype == torch.long, "Labels must be long integer tensor"


def test_train_vs_val_transforms_differ():
    train_tf = get_transforms(input_size=(64, 64), is_training=True)
    val_tf = get_transforms(input_size=(64, 64), is_training=False)

    dummy_img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    
    # Train transform should apply random variations
    t1 = train_tf(dummy_img)
    t2 = train_tf(dummy_img)
    # Val transform should be strictly deterministic
    v1 = val_tf(dummy_img)
    v2 = val_tf(dummy_img)

    assert torch.allclose(v1, v2), "Validation transform must be deterministic!"
    assert not torch.allclose(t1, t2), "Train transform should apply stochastic augmentation!"


def test_preprocessing_tensor_value_range():
    val_tf = get_transforms(input_size=(64, 64), is_training=False)
    dummy_img = Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8))
    tensor = val_tf(dummy_img)

    assert tensor.shape == (3, 64, 64), "Output tensor shape must be (3, 64, 64)"
    assert not torch.isnan(tensor).any(), "Tensor contains NaN values!"
    assert not torch.isinf(tensor).any(), "Tensor contains Inf values!"
