import os
import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.custom_cnn import SignNetLite, build_custom_cnn
from src.models.factory import get_model
from src.utils import count_parameters, save_checkpoint


def test_signnet_lite_initialization_and_forward():
    model = build_custom_cnn(num_classes=43)
    dummy_input = torch.randn(4, 3, 64, 64)
    logits = model(dummy_input)

    assert logits.shape == (4, 43), f"Expected logits shape (4, 43), got {logits.shape}"
    
    total, trainable = count_parameters(model)
    assert total > 500000, f"SignNet-Lite parameters ({total}) should be > 500k"
    assert total == trainable, "All parameters in SignNet-Lite baseline must be trainable"


def test_signnet_lite_configurable_classes():
    model_10 = SignNetLite(num_classes=10)
    dummy = torch.randn(2, 3, 64, 64)
    out = model_10(dummy)
    assert out.shape == (2, 10), "Model should support configurable number of classes"


def test_mobilenet_v2_factory_and_freezing():
    model = get_model(model_name="mobilenet_v2", num_classes=43, pretrained=False, freeze_backbone=True)
    dummy = torch.randn(2, 3, 64, 64)
    logits = model(dummy)
    assert logits.shape == (2, 43)

    total, trainable = count_parameters(model)
    assert trainable < total, "Frozen backbone should have fewer trainable parameters than total"


def test_resnet18_factory_and_freezing():
    model = get_model(model_name="resnet18", num_classes=43, pretrained=False, freeze_backbone=True)
    dummy = torch.randn(2, 3, 64, 64)
    logits = model(dummy)
    assert logits.shape == (2, 43)

    total, trainable = count_parameters(model)
    assert trainable < total, "Frozen backbone should have fewer trainable parameters than total"


def test_loss_computation_and_backward():
    model = SignNetLite(num_classes=43)
    criterion = nn.CrossEntropyLoss()
    
    inputs = torch.randn(4, 3, 64, 64)
    targets = torch.tensor([0, 14, 25, 42], dtype=torch.long)
    
    logits = model(inputs)
    loss = criterion(logits, targets)
    
    assert not torch.isnan(loss), "Loss computed to NaN"
    assert loss.item() > 0, "Loss must be positive"
    
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient for {name} was not computed"


def test_checkpoint_save_and_load(tmp_path):
    model = SignNetLite(num_classes=43)
    ckpt_file = tmp_path / "test_signnet.pth"
    
    save_checkpoint(model, None, epoch=1, val_loss=0.5, val_acc=85.0, path=str(ckpt_file))
    assert ckpt_file.exists(), "Checkpoint file was not created"
    
    new_model = SignNetLite(num_classes=43)
    checkpoint = torch.load(ckpt_file)
    new_model.load_state_dict(checkpoint["model_state_dict"])
    
    model.eval()
    new_model.eval()
    
    dummy = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out1 = model(dummy)
        out2 = new_model(dummy)
    
    assert torch.allclose(out1, out2, atol=1e-5), "Loaded model outputs do not match original model"
