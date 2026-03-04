"""
End-to-end training pipeline: SAM3 auto-label → YOLO dataset → YOLOv8n-seg training.
Exports SAM3 masks as segmentation polygons (not bounding boxes).
No filters applied — all raw SAM3 detections are used as training labels.
Change NUM_IMAGES to control how many images to process (10 for testing, 1000 for full run).
"""
import os, sys, time, glob, json, random, shutil
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Thread
from queue import Queue

os.environ['ROBOFLOW_API_KEY'] = '6mPyaZWFhvbmBKftxcq7'

# ═══════════════════════════════════════════════════════════════════
# CONFIG — Change NUM_IMAGES here
# ═══════════════════════════════════════════════════════════════════
NUM_IMAGES = 10  # Set to 1000 for full training run

IMAGE_DIR = "/teamspace/studios/this_studio/room_images"
LABEL_DIR = "/teamspace/studios/this_studio/pipeline_labels"
DATASET_DIR = "/teamspace/studios/this_studio/pipeline_dataset"
TRAIN_DIR = "/teamspace/studios/this_studio/pipeline_training"
SPLIT_RATIO = 0.85
SEED = 42
PRELOAD_WORKERS = 4
SAVE_WORKERS = 4

ONTOLOGY = {
    "ceiling": "ceilings",
    "curtain": "curtains",
    "decorative object": "decor",
    "floor": "floors",
    "upholstered furniture": "upholstery",
    "wall": "walls",
    "countertop surface": "worktop_surface",
    "board accessory": "board_accessory",
    "faucet": "faucet_tap",
    "light fixture": "fixtures",
    "door handle": "handle",
    "cabinet knob": "knob",
    "hardware fitting": "other_hardware",
    "outdoor fabric": "outdoor_fabric",
    "outdoor paving stone": "outdoor_paver",
    "stair rod": "stair_rod",
    "light switch": "switch",
    "wallpaper": "wallpaper_wallcovering",
    "background": "na",
}

# ═══════════════════════════════════════════════════════════════════
# STEP 1: SAM3 AUTO-LABELING
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print(f"STEP 1: SAM3 Auto-Labeling ({NUM_IMAGES} images)")
print("=" * 70)

from autodistill_sam3 import SegmentAnything3
from autodistill.detection import CaptionOntology
from autodistill.helpers import load_image

print("Loading SAM3...")
t0 = time.time()
ontology = CaptionOntology(ONTOLOGY)
model = SegmentAnything3(ontology=ontology)
classes = ontology.classes()
print(f"Loaded in {time.time()-t0:.1f}s | {len(classes)} classes\n")

# Find and sample images
extensions = ["*.png", "*.jpg", "*.jpeg"]
all_images = []
for ext in extensions:
    all_images.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
all_images.sort()

random.seed(SEED)
image_paths = random.sample(all_images, min(NUM_IMAGES, len(all_images)))
print(f"Found {len(all_images)} total images, selected {len(image_paths)}\n")

# Clean output dirs
for d in [LABEL_DIR, DATASET_DIR]:
    if os.path.exists(d):
        shutil.rmtree(d)
os.makedirs(os.path.join(LABEL_DIR, "labels"), exist_ok=True)


def mask_to_polygon(mask, img_w, img_h, max_points=100):
    """Convert a binary mask to a normalized polygon (YOLO seg format).
    Returns list of normalized (x, y) floats, or None if mask is invalid."""
    mask_uint8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # Take the largest contour
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return None
    # Simplify if too many points
    if len(contour) > max_points:
        epsilon = 0.001 * cv2.arcLength(contour, True)
        contour = cv2.approxPolyDP(contour, epsilon, True)
        while len(contour) > max_points and epsilon < 0.05 * cv2.arcLength(contour, True):
            epsilon *= 1.5
            contour = cv2.approxPolyDP(contour, epsilon, True)
    if len(contour) < 3:
        return None
    # Normalize to [0, 1]
    points = []
    for pt in contour.squeeze():
        points.append(pt[0] / img_w)
        points.append(pt[1] / img_h)
    return points


def detections_to_yolo_seg(detections, img_w, img_h):
    """Convert detections with masks to YOLO segmentation format.
    Format: class_id x1 y1 x2 y2 x3 y3 ... (normalized polygon points)
    Falls back to bbox polygon if mask is unavailable."""
    lines = []
    has_masks = detections.mask is not None and len(detections.mask) > 0
    for i in range(len(detections)):
        class_id = int(detections.class_id[i])
        polygon = None
        if has_masks:
            polygon = mask_to_polygon(detections.mask[i], img_w, img_h)
        # Fallback: convert bbox to a 4-point polygon
        if polygon is None:
            x1, y1, x2, y2 = detections.xyxy[i]
            polygon = [
                x1 / img_w, y1 / img_h,
                x2 / img_w, y1 / img_h,
                x2 / img_w, y2 / img_h,
                x1 / img_w, y2 / img_h,
            ]
        coords = " ".join(f"{v:.6f}" for v in polygon)
        lines.append(f"{class_id} {coords}")
    return lines


def save_label(stem, yolo_lines):
    path = os.path.join(LABEL_DIR, "labels", f"{stem}.txt")
    with open(path, "w") as f:
        f.write("\n".join(yolo_lines))


# Preload images on CPU while GPU infers
preload_queue = Queue(maxsize=PRELOAD_WORKERS * 3)


def preloader():
    def load_one(p):
        try:
            return p, load_image(p, return_format="cv2")
        except:
            return p, None
    with ThreadPoolExecutor(max_workers=PRELOAD_WORKERS) as pool:
        batch = PRELOAD_WORKERS * 3
        for i in range(0, len(image_paths), batch):
            futs = [pool.submit(load_one, p) for p in image_paths[i:i+batch]]
            for f in as_completed(futs):
                preload_queue.put(f.result())
    preload_queue.put(None)


preload_thread = Thread(target=preloader, daemon=True)
preload_thread.start()

save_pool = ThreadPoolExecutor(max_workers=SAVE_WORKERS)
save_futures = []
total_start = time.time()
processed = 0
total_raw = 0
valid_pairs = []
class_counts = {}

while True:
    item = preload_queue.get()
    if item is None:
        break

    img_path, image = item
    fname = os.path.basename(img_path)
    stem = os.path.splitext(fname)[0]
    processed += 1

    if image is None:
        print(f"  [{processed}/{len(image_paths)}] {fname} ... SKIP (load failed)")
        continue

    try:
        t1 = time.time()
        detections = model.predict(img_path)
        infer_time = time.time() - t1
        raw_count = len(detections)
        total_raw += raw_count

        # No filters — use raw SAM3 output as-is

        # Track class distribution
        if len(detections) > 0:
            for cid in detections.class_id:
                cls_name = classes[cid]
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

        img_h, img_w = image.shape[:2]
        yolo_lines = detections_to_yolo_seg(detections, img_w, img_h)
        save_futures.append(save_pool.submit(save_label, stem, yolo_lines))

        if raw_count > 0:
            valid_pairs.append((img_path, os.path.join(LABEL_DIR, "labels", f"{stem}.txt")))

        print(f"  [{processed}/{len(image_paths)}] {fname} ... {infer_time:.1f}s | detections={raw_count}")

    except Exception as e:
        print(f"  [{processed}/{len(image_paths)}] {fname} ... ERROR: {e}")

for f in save_futures:
    f.result()
save_pool.shutdown()

label_time = time.time() - total_start
print(f"\nLabeling done: {processed} images in {label_time:.1f}s ({label_time/max(processed,1):.1f}s/img)")
print(f"Valid images (with detections): {len(valid_pairs)}/{processed}")
print(f"Total detections: {total_raw} (no filters applied)")
print(f"\nClass distribution:")
for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
    print(f"  {cls:25s}: {cnt}")

# Save metadata
with open(os.path.join(LABEL_DIR, "classes.txt"), "w") as f:
    f.write("\n".join(classes))
with open(os.path.join(LABEL_DIR, "summary.json"), "w") as f:
    json.dump({
        "num_images": processed,
        "valid_images": len(valid_pairs),
        "total_detections": total_raw,
        "filters": "none — raw SAM3 output",
        "class_counts": class_counts,
        "label_time_seconds": round(label_time, 1),
        "config": {
            "num_classes": len(classes),
        },
    }, f, indent=2)

if len(valid_pairs) < 2:
    print("\nERROR: Not enough valid images to create train/val split. Exiting.")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════
# STEP 2: PREPARE YOLO DATASET
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("STEP 2: Prepare YOLO Dataset")
print("=" * 70)

import yaml

for split in ["train", "val"]:
    os.makedirs(os.path.join(DATASET_DIR, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, split, "labels"), exist_ok=True)

random.seed(SEED)
random.shuffle(valid_pairs)
split_idx = max(int(len(valid_pairs) * SPLIT_RATIO), 1)
train_pairs = valid_pairs[:split_idx]
val_pairs = valid_pairs[split_idx:]

# Ensure val has at least 1 image
if len(val_pairs) == 0 and len(train_pairs) > 1:
    val_pairs = [train_pairs.pop()]

print(f"Train: {len(train_pairs)} | Val: {len(val_pairs)}")

for split_name, pairs in [("train", train_pairs), ("val", val_pairs)]:
    for img_path, lbl_path in pairs:
        img_fname = os.path.basename(img_path)
        lbl_fname = os.path.basename(lbl_path)
        dst_img = os.path.join(DATASET_DIR, split_name, "images", img_fname)
        dst_lbl = os.path.join(DATASET_DIR, split_name, "labels", lbl_fname)
        if not os.path.exists(dst_img):
            os.symlink(os.path.abspath(img_path), dst_img)
        shutil.copy2(lbl_path, dst_lbl)

dataset_yaml_path = os.path.join(DATASET_DIR, "dataset.yaml")
dataset_config = {
    "path": DATASET_DIR,
    "train": "train/images",
    "val": "val/images",
    "nc": len(classes),
    "names": classes,
}
with open(dataset_yaml_path, "w") as f:
    yaml.dump(dataset_config, f, default_flow_style=False)

print(f"Dataset YAML: {dataset_yaml_path}")
print("Dataset ready\n")

# ═══════════════════════════════════════════════════════════════════
# STEP 3: TRAIN YOLOv8n-seg
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 3: Train YOLOv8n-seg Student Model")
print("=" * 70)

from ultralytics import YOLO

# Scale epochs: short for test runs, full for production
train_epochs = 20 if NUM_IMAGES <= 50 else 150

yolo = YOLO("yolov8n-seg.pt")
results = yolo.train(
    data=dataset_yaml_path,
    epochs=train_epochs,
    imgsz=640,
    batch=16,
    patience=25,
    project=TRAIN_DIR,
    name="sam3_distilled",
    device=0,
    workers=4,
    verbose=True,
    # Augmentation
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

total_time = time.time() - total_start
print(f"\n{'='*70}")
print("PIPELINE COMPLETE")
print(f"{'='*70}")
print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
print(f"  Labeling:  {label_time:.1f}s")
print(f"  Dataset+Training: {total_time - label_time:.1f}s")
print(f"Images: {NUM_IMAGES} | Valid: {len(valid_pairs)} | Train: {len(train_pairs)} | Val: {len(val_pairs)}")
print(f"Epochs: {train_epochs}")
print(f"Best model: {TRAIN_DIR}/sam3_distilled/weights/best.pt")
print(f"\nTo run full training, set NUM_IMAGES = 1000 and re-run.")
