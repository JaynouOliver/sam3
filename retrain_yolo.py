"""
Retrain YOLO on existing SAM3-labeled dataset (skip labeling).
Uses pipeline_dataset/ which already has 840 train / 149 val images.
"""

import time
from pathlib import Path
from ultralytics import YOLO

# ── Config ──────────────────────────────────────────────────────────
MODEL = "yolov8l-seg.pt"       # Large model
DATASET_YAML = "pipeline_dataset/dataset.yaml"
TRAIN_DIR = "pipeline_training"
RUN_NAME = "sam3_distilled_l"   # Change per run to avoid overwriting
EPOCHS = 150
BATCH = 16
PATIENCE = 25
# ────────────────────────────────────────────────────────────────────

print(f"Training {MODEL} on {DATASET_YAML}")
print(f"Output: {TRAIN_DIR}/{RUN_NAME}/")
print(f"Epochs: {EPOCHS}, Batch: {BATCH}\n")

start = time.time()

yolo = YOLO(MODEL)
results = yolo.train(
    data=DATASET_YAML,
    epochs=EPOCHS,
    imgsz=640,
    batch=BATCH,
    patience=PATIENCE,
    project=TRAIN_DIR,
    name=RUN_NAME,
    device=0,
    workers=4,
    verbose=True,
    # Augmentation (same as original pipeline)
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    flipud=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.15,
    scale=0.5,
    translate=0.1,
    degrees=10.0,
)

elapsed = time.time() - start
print(f"\nDone in {elapsed:.1f}s ({elapsed/60:.1f} min)")
print(f"Best weights: {TRAIN_DIR}/{RUN_NAME}/weights/best.pt")
