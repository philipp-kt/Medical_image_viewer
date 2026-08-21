"""
Chest X-ray Analysis using ianpan/chest-x-ray-basic model.

Applies lung/heart segmentation and classification (view, age, sex)
to a DICOM chest X-ray image.

Usage:
    python chest_xray_analysis.py [path_to_dicom]

If no path is given, defaults to the ST000002/SE000000/CR000000 file.

Requirements:
    pip install torch torchvision transformers pydicom numpy opencv-python matplotlib
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from transformers import AutoModel
import pydicom

try:
    from config import DEFAULT_DICOM_PATH
except ImportError:
    # No local config.py present — see config.example.py to set a default
    # file. Otherwise pass a path as a command-line argument.
    DEFAULT_DICOM_PATH = ""

VIEW_LABELS = {0: "AP", 1: "PA", 2: "Lateral"}


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    model = AutoModel.from_pretrained("ianpan/chest-x-ray-basic", trust_remote_code=True)
    model = model.eval().to(device)
    print("Model loaded.")
    return model, device


def load_dicom_image(model, dicom_path):
    """Load DICOM image using the model's built-in loader, with fallback."""
    try:
        img = model.load_image_from_dicom(str(dicom_path))
        print(f"Loaded DICOM via model loader: shape {img.shape}")
        return img
    except Exception:
        pass

    # Fallback: load with pydicom and convert to grayscale uint8
    ds = pydicom.dcmread(str(dicom_path), force=True)
    pixel_array = ds.pixel_array.astype(np.float64)
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    pixel_array = pixel_array * slope + intercept
    # Normalize to 0-255
    pmin, pmax = pixel_array.min(), pixel_array.max()
    if pmax > pmin:
        pixel_array = (pixel_array - pmin) / (pmax - pmin) * 255.0
    img = pixel_array.astype(np.uint8)
    print(f"Loaded DICOM via pydicom fallback: shape {img.shape}")
    return img


def run_inference(model, device, img):
    x = model.preprocess(img)
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float()
    with torch.inference_mode():
        out = model(x.to(device))
    return out


def print_results(out):
    # View classification
    view_idx = out["view"].argmax(1).item()
    print(f"\nView type : {VIEW_LABELS.get(view_idx, view_idx)}")

    # Age prediction
    age = out["age"].item()
    print(f"Predicted age : {age:.1f} years")

    # Sex prediction
    female_score = out["female"].item()
    sex = "Female" if female_score >= 0.5 else "Male"
    print(f"Predicted sex : {sex} (female score: {female_score:.3f})")


def visualize(img, out):
    mask = out["mask"].argmax(1).squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Original image
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Segmentation mask
    seg_cmap = ListedColormap(["black", "steelblue", "cornflowerblue", "indianred"])
    axes[1].imshow(mask, cmap=seg_cmap, vmin=0, vmax=3)
    axes[1].set_title("Segmentation\n(blue=lungs, red=heart)")
    axes[1].axis("off")

    # Overlay
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else img.copy()
    # Resize mask to match image
    mask_resized = cv2.resize(mask.astype(np.uint8), (img_rgb.shape[1], img_rgb.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
    overlay = img_rgb.astype(np.float32).copy()
    # Right lung - blue
    overlay[mask_resized == 1] = overlay[mask_resized == 1] * 0.5 + np.array([70, 130, 180]) * 0.5
    # Left lung - blue
    overlay[mask_resized == 2] = overlay[mask_resized == 2] * 0.5 + np.array([100, 149, 237]) * 0.5
    # Heart - red
    overlay[mask_resized == 3] = overlay[mask_resized == 3] * 0.5 + np.array([205, 92, 92]) * 0.5

    axes[2].imshow(overlay.astype(np.uint8))
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    # Add classification results as text
    view_idx = out["view"].argmax(1).item()
    age = out["age"].item()
    female_score = out["female"].item()
    sex = "Female" if female_score >= 0.5 else "Male"

    fig.suptitle(
        f"View: {VIEW_LABELS.get(view_idx, view_idx)}  |  "
        f"Age: {age:.1f} yrs  |  "
        f"Sex: {sex} ({female_score:.2f})",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) > 1:
        dicom_path = Path(sys.argv[1])
    elif DEFAULT_DICOM_PATH:
        dicom_path = Path(DEFAULT_DICOM_PATH)
    else:
        print("Error: no DICOM path given. Pass one as an argument, or set "
              "DEFAULT_DICOM_PATH in config.py (see config.example.py).")
        sys.exit(1)

    if not dicom_path.exists():
        print(f"Error: file not found: {dicom_path}")
        sys.exit(1)

    print(f"DICOM file: {dicom_path}")

    model, device = load_model()
    img = load_dicom_image(model, dicom_path)
    out = run_inference(model, device, img)
    print_results(out)
    visualize(img, out)


if __name__ == "__main__":
    main()
