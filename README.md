# DICOM Medical Image Viewer

A lightweight desktop application for browsing and displaying medical images in DICOM format.

## Features

- **File Browser** — Navigate DICOM directory structures (Study / Series / Image) via a tree view
- **Image Display** — Renders DICOM pixel data with proper rescale slope and intercept handling
- **Zoom & Pan** — Mouse wheel zoom with percentage relative to native image size, click-drag panning, Fit and 1:1 buttons
- **Window / Level Controls** — Adjust brightness and contrast with sliders; automatically initializes from DICOM tags when available
- **Export Image** — Save the displayed image (with current window/level) as PNG, JPEG, TIFF, or BMP via File > Export Image
- **Metadata Panel** — Displays key DICOM attributes including patient name, modality, study date, and image dimensions
- **Multi-format Support** — Handles grayscale, RGB, and multi-frame DICOM images

## Chest X-Ray Analysis

Automated analysis of chest X-ray DICOM images using the [ianpan/chest-x-ray-basic](https://huggingface.co/ianpan/chest-x-ray-basic) model (EfficientNetV2-S + U-Net decoder, 22.2M parameters).

### Capabilities

- **Lung & Heart Segmentation** — Produces pixel-level masks for right lung, left lung, and heart
- **View Classification** — Identifies AP, PA, or lateral projection
- **Age Prediction** — Estimates patient age from radiographic features
- **Sex Prediction** — Binary male/female classification

### Output

Displays a three-panel matplotlib visualization:
1. Original grayscale image
2. Segmentation mask (blue = lungs, red = heart)
3. Overlay of segmentation on the original image

Classification results (view, age, sex) are printed to the console and shown in the figure title.

## Requirements

### DICOM Viewer

- Python 3.10+
- [pydicom](https://pydicom.github.io/) — DICOM file reading
- [NumPy](https://numpy.org/) — Pixel data processing
- [Pillow](https://pillow.readthedocs.io/) — Image rendering
- tkinter — GUI framework (included with Python)

### Chest X-Ray Analysis

- Python 3.10+
- [PyTorch](https://pytorch.org/)
- [torchvision](https://pytorch.org/vision/)
- [transformers](https://huggingface.co/docs/transformers/) (v4.x)
- [timm](https://huggingface.co/docs/timm/)
- [albumentations](https://albumentations.ai/)
- [OpenCV](https://opencv.org/) (`opencv-python`)
- [matplotlib](https://matplotlib.org/)
- pydicom, NumPy

## Installation

```bash
# DICOM Viewer
pip install pydicom numpy pillow

# Chest X-Ray Analysis
pip install torch torchvision transformers timm albumentations pydicom numpy opencv-python matplotlib
```

## Usage

```bash
# DICOM Viewer
python dicom_viewer.py

# Chest X-Ray Analysis (default DICOM path)
python chest_xray_analysis.py

# Chest X-Ray Analysis (custom DICOM file)
python chest_xray_analysis.py path/to/dicom/file
```

The viewer automatically loads the default DICOM directory if configured. To open a different folder, use **File > Open Folder**.

### Optional: default paths

Copy `config.example.py` to `config.py` (gitignored) and set `DEFAULT_DICOM_DIR` / `DEFAULT_DICOM_PATH` to auto-load your own data on startup. Without it, both scripts work fine — just pick a folder/file manually.
