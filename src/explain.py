import gc
import os
import sys
import warnings
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2

from src.dataset import preprocess_image, NORM_MEAN, NORM_STD


class GradCAM:
    """
    Grad-CAM (Gradient-Weighted Class Activation Mapping) generator.

    Memory contract:
    - Activations and gradients captured by hooks are stored temporarily.
    - generate_heatmap() explicitly deletes them after computing the heatmap.
    - Hooks are removed when the context manager exits or the object is deleted.
    - The backward pass uses a minimal local gradient context so the full
      computation graph is not retained beyond the backward() call.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handles = []

        # Ensure target layer parameters require gradients so backward hook fires
        for param in self.target_layer.parameters():
            param.requires_grad_(True)

        self.handles.append(self.target_layer.register_forward_hook(self._save_activations))
        self.handles.append(self.target_layer.register_full_backward_hook(self._save_gradients))

    # --- Hooks -----------------------------------------------------------

    def _save_activations(self, module, input, output):
        # Detach immediately — we only need the values, not the graph
        self.activations = output.detach().clone()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach().clone()

    # --- Lifecycle -------------------------------------------------------

    def remove_hooks(self):
        """Remove all registered PyTorch hooks and clear cached tensors."""
        for handle in self.handles:
            try:
                handle.remove()
            except Exception:
                pass
        self.handles = []
        # Explicitly release hook-captured tensors
        self.activations = None
        self.gradients = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()

    def __del__(self):
        self.remove_hooks()

    # --- Core Grad-CAM ---------------------------------------------------

    def generate_heatmap(self, input_tensor, target_class=None):
        """
        Compute a Grad-CAM heatmap for the given input tensor.

        Returns:
            heatmap   : np.ndarray [H, W] in [0, 1]
            target_class : int
            probs     : np.ndarray [num_classes] softmax probabilities
        """
        if input_tensor.dim() != 4 or tuple(input_tensor.shape[1:]) != (3, 64, 64):
            raise ValueError(
                f"Grad-CAM input_tensor must be [B, 3, 64, 64], got {list(input_tensor.shape)}"
            )

        self.model.eval()

        # --- Forward pass (need grad enabled for backward to fire the hook) ---
        # We suppress the benign "Full backward hook is firing" UserWarning
        # (it still fires correctly because target layer params have requires_grad=True).
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Full backward hook is firing",
                category=UserWarning,
            )
            output = self.model(input_tensor)

        # Determine target class from argmax if not specified
        if target_class is None:
            target_class = int(torch.argmax(output, dim=1).item())

        # Compute softmax probabilities before backward (detached, no grad needed)
        with torch.no_grad():
            probs = F.softmax(output.detach(), dim=1)[0].cpu().numpy().copy()

        # Zero previous gradients and backprop through the target class score only
        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()

        # --- Extract heatmap from captured activations and gradients ---
        if self.gradients is None or self.activations is None:
            raise RuntimeError(
                "Grad-CAM failed to capture gradients or activations from target layer."
            )

        # Pool gradients across spatial dims → shape (C,)
        pooled_gradients = self.gradients.mean(dim=[0, 2, 3])  # (C,)

        # Weight each activation channel by its pooled gradient
        activations = self.activations[0].clone()   # (C, H, W) — clone to avoid in-place on hook tensor
        for i in range(activations.size(0)):
            activations[i] *= pooled_gradients[i]

        # Average across channels and apply ReLU
        heatmap = activations.mean(dim=0).cpu().numpy()          # (H, W)
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        # --- Explicitly release intermediate tensors ---
        del pooled_gradients, activations, score, output
        self.activations = None
        self.gradients = None
        gc.collect()

        return heatmap, target_class, probs


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------

def overlay_heatmap(heatmap: np.ndarray, display_pil: Image.Image, alpha: float = 0.5):
    """
    Blend a Grad-CAM heatmap with a PIL image.

    display_pil should already be resized to model input size (64×64) to keep
    intermediate NumPy arrays small and avoid up-sampling large upload images.

    Returns: (overlay_pil, heatmap_pil) — both PIL Images at display_pil size.
    """
    w, h = display_pil.size
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)

    orig_np = np.array(display_pil, dtype=np.uint8)
    blended = np.uint8(alpha * orig_np + (1.0 - alpha) * color_heatmap)

    overlay_pil = Image.fromarray(blended)
    heatmap_pil = Image.fromarray(color_heatmap)

    # Free large NumPy arrays
    del orig_np, blended, color_heatmap, heatmap_resized, heatmap_uint8

    return overlay_pil, heatmap_pil


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def explain_prediction(
    model,
    image_pil: Image.Image,
    target_layer=None,
    input_tensor=None,
    input_size=(64, 64),
    device="cpu",
):
    """
    Run Grad-CAM and return prediction results.

    Key memory optimisation: overlay_heatmap is called at model input_size (64×64),
    not at the original upload resolution (which can be up to 4096×4096).

    Returns: (predicted_class_id, top3_preds, overlay_image, heatmap_image)
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
                    f"Invalid input_tensor shape: expected (1, 3, {input_size[0]}, {input_size[1]}), "
                    f"got {tuple(input_tensor.shape)}"
                )
            input_tensor = input_tensor.to(device)

        heatmap, predicted_class, probs = grad_cam.generate_heatmap(input_tensor)
        # grad_cam context exits here — hooks removed, tensors cleared

    # Build display image at model input size (64×64) to keep NumPy arrays tiny
    display_pil = image_pil.convert("RGB").resize(input_size, Image.BILINEAR)
    overlay_img, heatmap_img = overlay_heatmap(heatmap, display_pil)
    del display_pil, heatmap

    # Top-3 predictions sorted by probability
    top3_indices = np.argsort(probs)[::-1][:3]
    top3_preds = [(int(idx), float(probs[idx])) for idx in top3_indices]

    return predicted_class, top3_preds, overlay_img, heatmap_img
