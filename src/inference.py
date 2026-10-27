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
    Enforces explicit architecture-to-checkpoint mapping, strict state_dict loading, and standardized preprocessing.
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
            pretrained=False
        ).to(self.device)
        
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"[Inference Engine] Checkpoint '{self.checkpoint_path}' not found for architecture '{self.model_name}'."
            )

        # 2. Load state_dict using strict=True to detect any architecture mismatch
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        if "model_state_dict" not in checkpoint:
            raise KeyError(f"Checkpoint '{self.checkpoint_path}' is missing 'model_state_dict'")

        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        print(f"[Inference Engine] Successfully loaded '{self.model_name}' weights from '{self.checkpoint_path}' (strict=True)")
        
        model.eval()
        return model

    def predict(self, image_pil):
        """
        Runs prediction on a PIL Image and returns top predictions + Grad-CAM heatmap overlay.
        Always processes through unified preprocess_image to ensure strictly [1, 3, 64, 64] tensor.
        """
        if not isinstance(image_pil, Image.Image):
            image_pil = Image.fromarray(image_pil)
        if image_pil.mode != "RGB":
            image_pil = image_pil.convert("RGB")

        # 1. Reusable preprocessing: ensures resize to 64x64, normalization, batch dim
        input_tensor = preprocess_image(image_pil, input_size=self.input_size, device=self.device)

        # 2. Strict shape verification before inference: raise clear error if not [1, 3, 64, 64]
        expected_shape = (1, 3, self.input_size[0], self.input_size[1])
        if tuple(input_tensor.shape) != expected_shape:
            raise ValueError(
                f"Input tensor shape mismatch: expected {expected_shape}, but got {tuple(input_tensor.shape)}"
            )

        # 3. Grad-CAM explanation using the exact same preprocessed tensor
        predicted_class_id, top3_preds, overlay_img, heatmap_img = explain_prediction(
            model=self.model,
            image_pil=image_pil,
            input_tensor=input_tensor,
            input_size=self.input_size,
            device=self.device
        )

        top_pred_name = self.class_names.get(str(predicted_class_id), f"Class {predicted_class_id}")
        top_pred_conf = top3_preds[0][1] * 100.0

        formatted_top3 = []
        for class_id, prob in top3_preds:
            formatted_top3.append({
                "class_id": class_id,
                "class_name": self.class_names.get(str(class_id), f"Class {class_id}"),
                "confidence": round(prob * 100.0, 2)
            })

        # Convert overlay image to Base64 data URL
        buf_overlay = BytesIO()
        overlay_img.save(buf_overlay, format="PNG")
        gradcam_base64 = "data:image/png;base64," + base64.b64encode(buf_overlay.getvalue()).decode("utf-8")

        # Convert standalone heatmap image to Base64 data URL
        buf_heatmap = BytesIO()
        heatmap_img.save(buf_heatmap, format="PNG")
        heatmap_base64 = "data:image/png;base64," + base64.b64encode(buf_heatmap.getvalue()).decode("utf-8")

        return {
            "predicted_class_id": predicted_class_id,
            "predicted_class_name": top_pred_name,
            "confidence": round(top_pred_conf, 2),
            "top3_predictions": formatted_top3,
            "gradcam_overlay_b64": gradcam_base64,
            "heatmap_b64": heatmap_base64,
            "model_name": self.model_name,
            "checkpoint_loaded": self.checkpoint_path
        }
