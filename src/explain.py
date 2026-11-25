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


# ---------------------------------------------------------------------------
# GradCAM — hooks registered only during one heatmap call, removed immediately
# ---------------------------------------------------------------------------

class GradCAM:
    """
    Minimal Grad-CAM implementation designed for tight memory on CPU-only hosts.

    Key properties:
    - Forward pass uses a local no_grad context for all layers EXCEPT the final
      score→backward() path.  This keeps the graph minimal.
    - Hook-captured tensors (activations, gradients) are cloned-and-detached
      immediately in the hook, so the full layer output graph is not retained.
    - generate_heatmap() explicitly deletes all intermediate tensors after use.
    - Hooks are removed inside the context-manager __exit__ and also in __del__.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self._activations = None
        self._gradients = None
        self._handles = []

        # Ensure target layer params participate in backward so the hook fires
        for p in self.target_layer.parameters():
            p.requires_grad_(True)

        self._handles.append(
            self.target_layer.register_forward_hook(self._hook_activations)
        )
        self._handles.append(
            self.target_layer.register_full_backward_hook(self._hook_gradients)
        )

    # --- Hooks (clone+detach immediately to free graph references) -----------

    def _hook_activations(self, module, inp, output):
        self._activations = output.detach().clone()

    def _hook_gradients(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach().clone()

    # --- Lifecycle -----------------------------------------------------------

    def _remove_hooks(self):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()
        self._activations = None
        self._gradients = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._remove_hooks()

    def __del__(self):
        self._remove_hooks()

    # --- Heatmap generation --------------------------------------------------

    def generate_heatmap(self, input_tensor, target_class=None):
        """
        Returns (heatmap_np [H,W] in [0,1], target_class int, probs_np [C]).

        Memory strategy:
        1. Run forward pass.  Suppress the benign UserWarning from full_backward_hook.
        2. Compute softmax probabilities with no_grad on a detached copy.
        3. Backprop only the target class score; graph is freed after backward().
        4. Compute heatmap from hook-captured tensors; delete them immediately.
        """
        if input_tensor.dim() != 4 or tuple(input_tensor.shape[1:]) != (3, 64, 64):
            raise ValueError(
                f"input_tensor must be [B,3,64,64], got {list(input_tensor.shape)}"
            )

        self.model.eval()
        self.model.zero_grad()

        # Forward pass — suppress benign warning about full_backward_hook
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Full backward hook is firing", category=UserWarning
            )
            output = self.model(input_tensor)   # shape [1, num_classes]

        # Choose target class
        if target_class is None:
            target_class = int(output.argmax(dim=1).item())

        # Probabilities — detached, no_grad, computed separately from backward graph
        with torch.no_grad():
            probs = F.softmax(output.detach(), dim=1)[0].cpu().numpy().copy()

        # Backprop through target class score only
        score = output[0, target_class]
        score.backward()

        # --- Heatmap from hooks ---
        if self._gradients is None or self._activations is None:
            raise RuntimeError("Grad-CAM hooks did not capture tensors — check target layer.")

        # Pooled gradients: mean across spatial dims → (C,)
        pooled = self._gradients.mean(dim=[0, 2, 3])  # (C,)

        # Weight activation channels, sum across channels
        acts = self._activations[0]  # (C, H, W)
        weighted = (acts * pooled[:, None, None]).sum(dim=0)  # (H, W)

        heatmap = weighted.cpu().numpy()
        heatmap = np.maximum(heatmap, 0)   # ReLU
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        # --- Explicit cleanup of all large intermediates ---
        del pooled, acts, weighted, score, output
        self._activations = None
        self._gradients = None
        gc.collect()

        return heatmap, target_class, probs


# ---------------------------------------------------------------------------
# Overlay helpers — always works at model input size (64×64), NOT upload size
# ---------------------------------------------------------------------------

def overlay_heatmap(heatmap: np.ndarray, display_pil: Image.Image, alpha: float = 0.45):
    """
    Blend Grad-CAM heatmap with display_pil (should be 64×64).
    Working at 64×64 keeps NumPy arrays tiny (~12 KB each).
    """
    w, h = display_pil.size
    hmap = cv2.resize(heatmap.astype(np.float32), (w, h))
    hmap_u8 = np.uint8(255 * hmap)

    color = cv2.applyColorMap(hmap_u8, cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

    orig = np.array(display_pil, dtype=np.float32)
    blended = np.clip(alpha * orig + (1.0 - alpha) * color.astype(np.float32), 0, 255).astype(np.uint8)

    overlay = Image.fromarray(blended)
    heatmap_img = Image.fromarray(color)

    del orig, blended, color, hmap, hmap_u8
    return overlay, heatmap_img


# ---------------------------------------------------------------------------
# High-level explain_prediction API
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
    Run Grad-CAM and return (predicted_class_id, top3_preds, overlay_img, heatmap_img).

    Overlay images are generated at model input_size (64×64) — NOT at upload resolution —
    to keep intermediate NumPy arrays small (~12 KB vs up to 50 MB for a 4096×4096 upload).
    """
    if target_layer is None:
        if hasattr(model, "get_last_conv_layer"):
            target_layer = model.get_last_conv_layer()
        else:
            raise ValueError("Pass target_layer or use a model with get_last_conv_layer().")

    with GradCAM(model, target_layer) as cam:
        if input_tensor is None:
            input_tensor = preprocess_image(image_pil, input_size=input_size, device=device)
        else:
            input_tensor = input_tensor.to(device)

        heatmap, predicted_class, probs = cam.generate_heatmap(input_tensor)
        # GradCAM context exits here — hooks removed, hook tensors freed

    # Build 64×64 display image — avoids creating large arrays for big uploads
    display_pil = image_pil.convert("RGB").resize(input_size, Image.BILINEAR)
    overlay_img, heatmap_img = overlay_heatmap(heatmap, display_pil)
    del display_pil, heatmap
    gc.collect()

    top3_indices = np.argsort(probs)[::-1][:3]
    top3_preds = [(int(i), float(probs[i])) for i in top3_indices]

    return predicted_class, top3_preds, overlay_img, heatmap_img
