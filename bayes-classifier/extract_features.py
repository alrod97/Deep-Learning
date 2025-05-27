import os
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

# -------------------------------------------------------------------
# Load model
# -------------------------------------------------------------------
MODEL_NAME = "facebook/dinov2-giant"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# -------------------------------------------------------------------
# Feature extraction function
# -------------------------------------------------------------------
@torch.inference_mode()
def extract_features(img: Image.Image, pool="cls"):
    inputs = processor(images=img.convert("RGB"), return_tensors="pt").to(DEVICE)
    outputs = model(**inputs)
    tokens = outputs.last_hidden_state

    if pool == "cls":
        feats = tokens[:, 0, :]
    elif pool == "mean":
        feats = tokens.mean(dim=1)
    else:
        raise ValueError("Invalid pooling type")

    feats = F.normalize(feats, p=2, dim=-1)
    return feats.squeeze(0).cpu().numpy()

# -------------------------------------------------------------------
# Main feature extraction logic
# -------------------------------------------------------------------
def process_dataset(dataset_dir: Path):
    for class_dir in sorted(dataset_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        feature_dir = class_dir / "features_dinov2-giant"
        feature_dir.mkdir(exist_ok=True)

        image_paths = [p for p in class_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
        print(f"🔍 Processing class '{class_name}' with {len(image_paths)} images...")

        for img_path in image_paths:
            img_name = img_path.stem
            out_path = feature_dir / f"{class_name}_{img_name}.npy"
            if out_path.exists():
                continue  # Skip existing

            try:
                img = Image.open(img_path)
                vec = extract_features(img)
                np.save(out_path, vec)
            except Exception as e:
                print(f"⚠️ Error processing {img_path.name}: {e}")

# -------------------------------------------------------------------
# Entry point with argparse
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract DINOv2 features from images.")
    parser.add_argument("dataset_dir", type=str, help="Path to dataset directory")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_dir)
    assert dataset_path.exists(), f"❌ Directory not found: {dataset_path}"

    process_dataset(dataset_path)

