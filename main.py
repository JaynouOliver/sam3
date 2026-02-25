#!/usr/bin/env python3
"""
Autodistill SAM3 — ready to run on Lightning Studio (GPU required).

Setup (run these in terminal first):
    pip install autodistill-sam3 autodistill supervision inference roboflow scikit-learn sam3
    export ROBOFLOW_API_KEY="6mPyaZWFhvbmBKftxcq7"

GPU: A10G (24GB) recommended. T4 (16GB) may work. L4 is also fine.
"""

import os
import subprocess
import sys

# ── 0. Install deps if missing ──────────────────────────────────────────────
def install_deps():
    packages = [
        "autodistill-sam3",
        "autodistill",
        "supervision",
        "inference",
        "roboflow",
        "scikit-learn",
        "sam3",
        "torch",
        "torchvision",
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + packages)

install_deps()

# ── 1. Set API key ──────────────────────────────────────────────────────────
os.environ["ROBOFLOW_API_KEY"] = "6mPyaZWFhvbmBKftxcq7"

# ── 2. Imports ──────────────────────────────────────────────────────────────
import numpy as np
import torch
import cv2
import supervision as sv
from autodistill_sam3 import SegmentAnything3
from autodistill.detection import CaptionOntology
from autodistill.helpers import load_image
from urllib.request import urlretrieve

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:             {torch.cuda.get_device_name(0)}")
    print(f"VRAM:            {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# ── 3. Download a sample image ──────────────────────────────────────────────
IMAGE_PATH = "sample_fruit.jpg"
if not os.path.exists(IMAGE_PATH):
    print("Downloading sample image...")
    urlretrieve(
        "https://images.unsplash.com/photo-1619566636858-adf3ef46400b?w=800",
        IMAGE_PATH,
    )
    print(f"Saved to {IMAGE_PATH}")

# ── 4. Build model with ontology ────────────────────────────────────────────
print("\nLoading SAM3 model...")
base_model = SegmentAnything3(
    ontology=CaptionOntology(
        {
            "fruit": "fruit",
            "leaf": "leaf",
        }
    )
)

# ── 5. Run inference ────────────────────────────────────────────────────────
print("Running inference...")
detections = base_model.predict(IMAGE_PATH)

print(f"\nDetections found: {len(detections)}")
for i, (xyxy, conf, cls_id) in enumerate(
    zip(detections.xyxy, detections.confidence, detections.class_id)
):
    label = base_model.ontology.classes()[cls_id]
    print(f"  [{i}] {label}: confidence={conf:.3f}, bbox={xyxy.astype(int).tolist()}")

# ── 6. Visualize & save ────────────────────────────────────────────────────
image = load_image(IMAGE_PATH, return_format="cv2")

label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
mask_annotator = sv.MaskAnnotator()

annotated = mask_annotator.annotate(scene=image.copy(), detections=detections)
annotated = label_annotator.annotate(
    scene=annotated,
    detections=detections,
    labels=[base_model.ontology.classes()[cid] for cid in detections.class_id],
)

OUTPUT_PATH = "output_annotated.jpg"
cv2.imwrite(OUTPUT_PATH, annotated)
print(f"\nAnnotated image saved to: {OUTPUT_PATH}")
print("Done!")
