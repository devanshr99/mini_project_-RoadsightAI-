"""
Profile peak RAM usage for each model's load + inference + Grad-CAM cycle.
Run locally before committing to verify memory budget fits in Render's 512 MB.

Usage:
    python scripts/profile_memory.py
"""
import os
import sys
import gc
import io
import tracemalloc
import base64
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image

from src.models.factory import get_model, get_checkpoint_for_model, resolve_model_key
from src.dataset import preprocess_image
from src.explain import explain_prediction


MODELS = ["custom_cnn", "signnet_lite", "mobilenet_v2", "resnet18"]


def make_test_image(size=(128, 128)) -> Image.Image:
    """Create a synthetic test image."""
    return Image.new("RGB", size, (220, 50, 50))


def mb(n: int) -> float:
    return n / 1_048_576


def profile_model(model_name: str, test_img: Image.Image) -> dict:
    gc.collect()
    tracemalloc.stop()
    tracemalloc.start()

    key = resolve_model_key(model_name)
    ckpt_path = get_checkpoint_for_model(key)
    ckpt_size = os.path.getsize(ckpt_path) / 1_048_576

    print(f"\n{'='*55}")
    print(f"Model : {model_name}")
    print(f"Ckpt  : {os.path.basename(ckpt_path)} ({ckpt_size:.2f} MB on disk)")
    print(f"{'='*55}")

    # --- Load checkpoint dict ---
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cur, peak = tracemalloc.get_traced_memory()
    print(f"  After torch.load():    cur={mb(cur):.1f} MB  peak={mb(peak):.1f} MB")

    # --- Instantiate model and copy weights ---
    model = get_model(model_name=key, num_classes=43, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    del ckpt
    gc.collect()
    cur, peak = tracemalloc.get_traced_memory()
    print(f"  After load_state_dict: cur={mb(cur):.1f} MB  peak={mb(peak):.1f} MB")
    model.eval()

    # --- Preprocess ---
    input_tensor = preprocess_image(test_img, input_size=(64, 64), device="cpu")
    cur, peak = tracemalloc.get_traced_memory()
    print(f"  After preprocess:      cur={mb(cur):.1f} MB  peak={mb(peak):.1f} MB")

    # --- Grad-CAM ---
    try:
        pred_cls, top3, overlay, heatmap = explain_prediction(
            model=model, image_pil=test_img,
            input_tensor=input_tensor, input_size=(64, 64), device="cpu"
        )
        cur, peak = tracemalloc.get_traced_memory()
        print(f"  After Grad-CAM:        cur={mb(cur):.1f} MB  peak={mb(peak):.1f} MB")
        print(f"  Prediction: class={pred_cls}  conf={top3[0][1]*100:.1f}%")

        # Encode to base64
        def to_b64(img):
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            r = base64.b64encode(buf.getvalue()).decode()
            buf.close()
            return r

        b64_o = to_b64(overlay)
        b64_h = to_b64(heatmap)
        cur, peak = tracemalloc.get_traced_memory()
        print(f"  After base64 encode:   cur={mb(cur):.1f} MB  peak={mb(peak):.1f} MB")
        print(f"  base64 overlay size: {len(b64_o)//1024} KB")
        del overlay, heatmap, b64_o, b64_h
    except Exception as e:
        print(f"  Grad-CAM FAILED: {e}")

    # --- Cleanup ---
    del input_tensor, model
    gc.collect()
    cur, peak = tracemalloc.get_traced_memory()
    print(f"  After cleanup:         cur={mb(cur):.1f} MB  peak={mb(peak):.1f} MB")

    _, session_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"model": model_name, "ckpt_mb": ckpt_size, "session_peak_mb": mb(session_peak)}


if __name__ == "__main__":
    test_img = make_test_image()
    results = []
    for m in MODELS:
        try:
            r = profile_model(m, test_img)
            results.append(r)
        except Exception as e:
            print(f"  ERROR for {m}: {e}")
        gc.collect()

    print(f"\n{'='*55}")
    print("SUMMARY (tracemalloc — Python heap only, not torch internals):")
    for r in results:
        print(f"  {r['model']:<20} ckpt={r['ckpt_mb']:.2f} MB  session_peak={r['session_peak_mb']:.1f} MB")
    print("\nNote: Render free tier = 512 MB total RSS.")
    print("Torch tensor memory is not fully captured by tracemalloc;")
    print("check /api/memory on Render for live RSS after deployment.")
