import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2

from src.dataset import preprocess_image, get_transforms, NORM_MEAN, NORM_STD


class GradCAM:
    """
    Grad-CAM (Gradient-Weighted Class Activation Mapping) Generator.
    Produces visual heatmaps explaining CNN model predictions.
    Safely registers and removes PyTorch hooks to prevent accumulation.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handles = []

        # Ensure target layer parameters require gradients so backward hook reliably captures gradients
        for param in self.target_layer.parameters():
            param.requires_grad = True

        # Register forward and backward hooks and record handles
        self.handles.append(self.target_layer.register_forward_hook(self._save_activations))
        self.handles.append(self.target_layer.register_full_backward_hook(self._save_gradients))

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        """Removes registered PyTorch hooks to avoid leaks and conflicts."""
        for handle in self.handles:
            try:
                handle.remove()
            except Exception:
                pass
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()

    def __del__(self):
        self.remove_hooks()

    def generate_heatmap(self, input_tensor, target_class=None):
        """
        Computes Grad-CAM activation heatmap for a given input tensor and target class.
        Strictly validates input tensor shape before executing forward pass.
        """
        if input_tensor.dim() != 4 or tuple(input_tensor.shape[1:]) != (3, 64, 64):
            raise ValueError(
                f"Grad-CAM input_tensor shape must be [B, 3, 64, 64], but got {list(input_tensor.shape)}"
            )

        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Target class score
        score = output[0, target_class]
        score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM failed to capture gradients or activations from target layer.")

        # Pool gradients across spatial channels: shape (C,)
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        # Weight activations by pooled gradients
        activations = self.activations[0] # shape (C, H, W)
        for i in range(activations.size(0)):
            activations[i] *= pooled_gradients[i]

        # Compute average across channel dimension
        heatmap = torch.mean(activations, dim=0).cpu().numpy()

        # Apply ReLU to keep only positive contributions
        heatmap = np.maximum(heatmap, 0)

        # Normalize to [0, 1]
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)

        return heatmap, target_class, F.softmax(output, dim=1)[0].detach().cpu().numpy()


def overlay_heatmap(heatmap, original_pil_image, alpha=0.5):
    """
    Blends Grad-CAM heatmap with original PIL Image using OpenCV Jet colormap.
    """
    w, h = original_pil_image.size
    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Convert to 8-bit heatmap image
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Apply Jet color map
    color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)

    # Blend original image with heatmap
    orig_np = np.array(original_pil_image)
    blended = np.uint8(alpha * orig_np + (1.0 - alpha) * color_heatmap)

    return Image.fromarray(blended), Image.fromarray(color_heatmap)


def explain_prediction(model, image_pil, target_layer=None, input_tensor=None, input_size=(64, 64), device="cpu"):
    """
    High-level API for model prediction explanation.
    Uses unified preprocess_image to ensure input tensor is strictly [1, 3, 64, 64].
    Returns: (predicted_class_id, top_3_probs, overlay_image, heatmap_image)
    """
    if target_layer is None:
        if hasattr(model, "get_last_conv_layer"):
            target_layer = model.get_last_conv_layer()
        else:
            raise ValueError("Target layer must be specified for Grad-CAM.")

    with GradCAM(model, target_layer) as grad_cam:
        if input_tensor is None:
            input_tensor = preprocess_image(image_pil, input_size=input_size, device=device)
        else:
            if input_tensor.dim() != 4 or tuple(input_tensor.shape) != (1, 3, input_size[0], input_size[1]):
                raise ValueError(
                    f"Invalid input_tensor shape for inference: expected (1, 3, {input_size[0]}, {input_size[1]}), "
                    f"but got {tuple(input_tensor.shape)}"
                )
            input_tensor = input_tensor.to(device)

        heatmap, predicted_class, probs = grad_cam.generate_heatmap(input_tensor)

    # Generate overlay image matching the exact input_size
    overlay_img, heatmap_img = overlay_heatmap(
        heatmap,
        image_pil.convert("RGB").resize(input_size, Image.BILINEAR)
    )

    # Top-3 predictions
    top3_indices = np.argsort(probs)[::-1][:3]
    top3_preds = [(int(idx), float(probs[idx])) for idx in top3_indices]

    return predicted_class, top3_preds, overlay_img, heatmap_img
