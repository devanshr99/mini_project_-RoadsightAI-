"""
TrafficSignPredictor — inference engine for RoadSight AI.

Memory contract (designed for Render free tier: 512 MB total):
  - Single model resident at a time (enforced by PredictorCache in app.py).
  - Checkpoint dict freed immediately after state_dict copy.
  - input_tensor deleted in finally block after Grad-CAM.
  - PIL images and BytesIO buffers closed after base64 encoding.
  - gc.collect() called at end of every predict() call.
"""
import gc
import os
import json
import base64
from io import BytesIO

import torch
from PIL import Image

from src.models.factory import get_model, get_checkpoint_for_model, resolve_model_key
from src.dataset import preprocess_image, get_transforms
from src.explain import explain_prediction


class TrafficSignPredictor:
    def __init__(self, model_name="custom_cnn", checkpoint_path=None, img_dim=64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = resolve_model_key(model_name)
        self.img_dim = img_dim
        self.input_size = (img_dim, img_dim)

        if checkpoint_path is None:
            checkpoint_path = get_checkpoint_for_model(self.model_name)

        self.checkpoint_path = str(checkpoint_path)
        self.class_names = self._load_class_names()
        self.model = self._load_model()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load_class_names(self) -> dict:
        path = "data/class_names.json"
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return {str(i): f"Class {i}" for i in range(43)}

    def _load_model(self):
        """
        Load model weights with minimal peak RAM:
          1. Instantiate empty model on CPU (no pretrained weights → no download).
          2. torch.load() the checkpoint.
          3. load_state_dict() — only the model_state_dict subtensor is copied.
          4. del checkpoint dict immediately → frees optimizer state, scheduler, etc.
          5. gc.collect() to return freed pages to OS promptly.
        """
        num_classes = len(self.class_names)
        model = get_model(
            model_name=self.model_name,
            num_classes=num_classes,
            pretrained=False,
        ).to(self.device)

        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint '{self.checkpoint_path}' not found for '{self.model_name}'."
            )

        # weights_only=False required for older checkpoint dicts that contain non-tensor values
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        if "model_state_dict" not in checkpoint:
            raise KeyError(f"'model_state_dict' missing in '{self.checkpoint_path}'")

        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        print(
            f"[Predictor] Loaded '{self.model_name}' "
            f"from '{os.path.basename(self.checkpoint_path)}' (strict=True, device={self.device})"
        )

        # Free the checkpoint dict — releases optimizer states and any extra keys.
        # This is the single most impactful RAM saving (up to 128 MB for ResNet18).
        del checkpoint
        gc.collect()

        model.eval()
        # Freeze all parameters so Grad-CAM only builds a backward graph from
        # the target convolutional layer onward instead of all 11M+ parameters.
        for param in model.parameters():
            param.requires_grad_(False)
        return model

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def predict(self, image_pil: Image.Image) -> dict:
        """
        Full prediction + Grad-CAM for one image.

        Every large intermediate (input_tensor, PIL images, BytesIO buffers) is
        explicitly deleted and gc.collect() is called at the end.
        """
        if not isinstance(image_pil, Image.Image):
            image_pil = Image.fromarray(image_pil)
        if image_pil.mode != "RGB":
            image_pil = image_pil.convert("RGB")

        # Preprocess → [1,3,64,64] on self.device
        input_tensor = preprocess_image(
            image_pil, input_size=self.input_size, device=self.device
        )

        expected = (1, 3, self.input_size[0], self.input_size[1])
        if tuple(input_tensor.shape) != expected:
            raise ValueError(
                f"Tensor shape {tuple(input_tensor.shape)} != expected {expected}"
            )

        overlay_img = heatmap_img = None
        try:
            predicted_class_id, top3_preds, overlay_img, heatmap_img = explain_prediction(
                model=self.model,
                image_pil=image_pil,
                input_tensor=input_tensor,
                input_size=self.input_size,
                device=self.device,
            )
        finally:
            del input_tensor
            gc.collect()

        # Class name and confidence
        top_name = self.class_names.get(str(predicted_class_id), f"Class {predicted_class_id}")
        top_conf = round(top3_preds[0][1] * 100.0, 2)

        formatted_top3 = [
            {
                "class_id": class_id,
                "class_name": self.class_names.get(str(class_id), f"Class {class_id}"),
                "confidence": round(prob * 100.0, 2),
            }
            for class_id, prob in top3_preds
        ]

        # Encode to base64 and immediately release PIL images + BytesIO buffers
        gradcam_b64 = self._encode_b64(overlay_img)
        heatmap_b64 = self._encode_b64(heatmap_img)
        del overlay_img, heatmap_img
        gc.collect()

        return {
            "predicted_class_id": predicted_class_id,
            "predicted_class_name": top_name,
            "confidence": top_conf,
            "top3_predictions": formatted_top3,
            "gradcam_overlay_b64": gradcam_b64,
            "heatmap_b64": heatmap_b64,
            "model_name": self.model_name,
            "checkpoint_loaded": self.checkpoint_path,
        }

    @staticmethod
    def _encode_b64(pil_img: Image.Image) -> str:
        """PNG-encode a PIL image to a data-URL base64 string, then close the buffer."""
        buf = BytesIO()
        try:
            pil_img.save(buf, format="PNG", optimize=True)
            return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        finally:
            buf.close()
