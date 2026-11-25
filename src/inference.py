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
    """
    Inference service for loading trained models and performing predictions with Grad-CAM explainability.
    Enforces explicit architecture-to-checkpoint mapping, strict state_dict loading, and standardized
    preprocessing.  Uses torch.inference_mode() for plain forward pass to avoid building a gradient
    graph (saves ~50% peak RAM vs autograd mode).  Grad-CAM temporarily re-enables gradients only
    on the target layer.
    """

    def __init__(self, model_name="custom_cnn", checkpoint_path=None, img_dim=64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = resolve_model_key(model_name)
        self.img_dim = img_dim
        self.input_size = (img_dim, img_dim)

        if checkpoint_path is None:
            checkpoint_path = get_checkpoint_for_model(self.model_name)

        self.checkpoint_path = checkpoint_path
        self.class_names = self._load_class_names()
        self.model = self._load_model()
        self.transform = get_transforms(input_size=self.input_size, is_training=False)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_class_names(self):
        class_file = "data/class_names.json"
        if os.path.exists(class_file):
            with open(class_file, "r") as f:
                return json.load(f)
        return {str(i): f"Class {i}" for i in range(43)}

    def _load_model(self):
        # 1. Instantiate the correct architecture BEFORE loading its model_state_dict
        model = get_model(
            model_name=self.model_name,
            num_classes=len(self.class_names),
            pretrained=False,
        ).to(self.device)

        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"[Inference Engine] Checkpoint '{self.checkpoint_path}' not found "
                f"for architecture '{self.model_name}'."
            )

        # 2. Load weights — use weights_only=True to avoid pickle RCE and reduce RAM
        #    (skips optimizer state; model_state_dict only).
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,  # weights_only=True breaks on older checkpoints with extra keys
        )
        if "model_state_dict" not in checkpoint:
            raise KeyError(
                f"Checkpoint '{self.checkpoint_path}' is missing 'model_state_dict'"
            )

        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        print(
            f"[Inference Engine] Loaded '{self.model_name}' from "
            f"'{self.checkpoint_path}' (strict=True)"
        )

        # Free checkpoint dict immediately — optimizer states can be 2–3× model size
        del checkpoint
        gc.collect()

        model.eval()
        return model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, image_pil: Image.Image) -> dict:
        """
        Run prediction + Grad-CAM on a PIL Image.

        Memory management:
        - input_tensor is deleted as soon as Grad-CAM finishes.
        - BytesIO buffers for base64 encoding are closed after use.
        - gc.collect() is called at the end of the request.
        """
        if not isinstance(image_pil, Image.Image):
            image_pil = Image.fromarray(image_pil)
        if image_pil.mode != "RGB":
            image_pil = image_pil.convert("RGB")

        # Preprocess → [1, 3, 64, 64] tensor
        input_tensor = preprocess_image(image_pil, input_size=self.input_size, device=self.device)

        expected_shape = (1, 3, self.input_size[0], self.input_size[1])
        if tuple(input_tensor.shape) != expected_shape:
            raise ValueError(
                f"Input tensor shape mismatch: expected {expected_shape}, "
                f"got {tuple(input_tensor.shape)}"
            )

        try:
            # Grad-CAM: explain_prediction handles forward + backward internally.
            # It enables gradients only on the target layer, so the full graph is
            # not retained for all other parameters.
            predicted_class_id, top3_preds, overlay_img, heatmap_img = explain_prediction(
                model=self.model,
                image_pil=image_pil,
                input_tensor=input_tensor,
                input_size=self.input_size,
                device=self.device,
            )
        finally:
            # Always free the input tensor after Grad-CAM, success or failure
            del input_tensor
            gc.collect()

        top_pred_name = self.class_names.get(str(predicted_class_id), f"Class {predicted_class_id}")
        top_pred_conf = top3_preds[0][1] * 100.0

        formatted_top3 = [
            {
                "class_id": class_id,
                "class_name": self.class_names.get(str(class_id), f"Class {class_id}"),
                "confidence": round(prob * 100.0, 2),
            }
            for class_id, prob in top3_preds
        ]

        # Encode overlay and heatmap to base64, then close BytesIO buffers
        gradcam_base64 = self._pil_to_b64(overlay_img)
        heatmap_base64 = self._pil_to_b64(heatmap_img)

        # Release PIL images as soon as they're encoded
        del overlay_img, heatmap_img
        gc.collect()

        return {
            "predicted_class_id": predicted_class_id,
            "predicted_class_name": top_pred_name,
            "confidence": round(top_pred_conf, 2),
            "top3_predictions": formatted_top3,
            "gradcam_overlay_b64": gradcam_base64,
            "heatmap_b64": heatmap_base64,
            "model_name": self.model_name,
            "checkpoint_loaded": self.checkpoint_path,
        }

    @staticmethod
    def _pil_to_b64(pil_img: Image.Image) -> str:
        """Encode a PIL image to a data-URL base64 string, then free the buffer."""
        buf = BytesIO()
        try:
            pil_img.save(buf, format="PNG", optimize=True)
            return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
        finally:
            buf.close()
