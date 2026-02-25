"""
Parallel SAM3 Auto-Labeling Pipeline.

Since SAM3 runs locally on GPU, there's no API rate limit.
The bottleneck is GPU memory. Strategy:
  - SAM3 model is loaded ONCE (GPU memory)
  - Image loading + preprocessing done in parallel on CPU (ThreadPool)
  - Inference is sequential on GPU (can't parallelize single-GPU inference)
  - Post-processing + saving done in parallel on CPU

This is a producer-consumer pipeline:
  CPU threads (load images) -> GPU (inference) -> CPU threads (save labels)

For true parallel inference, we'd need multiple GPUs or batch inference.
The speedup comes from overlapping I/O with computation.
"""
import os
import sys
import time
import glob
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from threading import Thread

os.environ['ROBOFLOW_API_KEY'] = '6mPyaZWFhvbmBKftxcq7'

from autodistill_sam3 import SegmentAnything3
from autodistill.detection import CaptionOntology
from autodistill.helpers import load_image
import supervision as sv

# ── Config ──
IMAGE_DIR = "/teamspace/studios/this_studio/product_images"
OUTPUT_DIR = "/teamspace/studios/this_studio/sam3_labels_1k"
MIN_CONFIDENCE = 0.7
MIN_AREA = 500
PRELOAD_WORKERS = 4  # threads for image preloading
SAVE_WORKERS = 4     # threads for saving results

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

os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels"), exist_ok=True)

# ── Load model once ──
print("Loading SAM3 model...")
t0 = time.time()
ontology = CaptionOntology(ONTOLOGY)
model = SegmentAnything3(ontology=ontology)
classes = ontology.classes()
print(f"Model loaded in {time.time()-t0:.1f}s | {len(classes)} classes\n")

# ── Find images ──
extensions = ["*.png", "*.jpg", "*.jpeg"]
image_paths = []
for ext in extensions:
    image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
image_paths.sort()
print(f"Found {len(image_paths)} images to label\n")

if not image_paths:
    print("No images found! Run download_images.py first.")
    sys.exit(1)


def detections_to_yolo(detections, img_w, img_h):
    lines = []
    for xyxy, conf, class_id in zip(detections.xyxy, detections.confidence, detections.class_id):
        x1, y1, x2, y2 = xyxy
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        lines.append(f"{int(class_id)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def save_result(stem, fname, img_path, yolo_lines):
    """Save label file and symlink image (runs on CPU thread)."""
    label_path = os.path.join(OUTPUT_DIR, "labels", f"{stem}.txt")
    with open(label_path, "w") as f:
        f.write("\n".join(yolo_lines))
    img_dest = os.path.join(OUTPUT_DIR, "images", fname)
    if not os.path.exists(img_dest):
        os.symlink(img_path, img_dest)


# ── Preload image queue ──
# Preload images on CPU threads while GPU does inference
preload_queue = Queue(maxsize=PRELOAD_WORKERS * 2)


def preloader():
    """Preload images into memory using threads."""
    def load_one(path):
        try:
            img = load_image(path, return_format="cv2")
            return path, img
        except Exception as e:
            return path, None

    with ThreadPoolExecutor(max_workers=PRELOAD_WORKERS) as pool:
        # Submit in batches to keep queue fed
        batch_size = PRELOAD_WORKERS * 2
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i + batch_size]
            futures = [pool.submit(load_one, p) for p in batch]
            for f in as_completed(futures):
                preload_queue.put(f.result())
    # Sentinel to signal done
    preload_queue.put(None)


# Start preloader thread
preload_thread = Thread(target=preloader, daemon=True)
preload_thread.start()

# ── Main inference loop ──
save_pool = ThreadPoolExecutor(max_workers=SAVE_WORKERS)
save_futures = []
results = []
total_start = time.time()
processed = 0
total_raw = 0
total_filtered = 0

print(f"Starting pipeline: preload({PRELOAD_WORKERS}) -> GPU inference -> save({SAVE_WORKERS})")
print("-" * 70)

while True:
    item = preload_queue.get()
    if item is None:
        break

    img_path, image = item
    fname = os.path.basename(img_path)
    stem = os.path.splitext(fname)[0]
    processed += 1

    if image is None:
        print(f"[{processed}/{len(image_paths)}] {fname} SKIP (load failed)")
        results.append({"file": fname, "error": "load failed"})
        continue

    try:
        t1 = time.time()
        detections = model.predict(img_path)
        infer_time = time.time() - t1
        raw_count = len(detections)
        total_raw += raw_count

        # Filter
        if len(detections) > 0:
            areas = (detections.xyxy[:, 2] - detections.xyxy[:, 0]) * \
                    (detections.xyxy[:, 3] - detections.xyxy[:, 1])
            mask = (detections.confidence >= MIN_CONFIDENCE) & (areas >= MIN_AREA)
            detections = detections[mask]
            if len(detections) > 0:
                detections = detections.with_nms(threshold=0.5, class_agnostic=True)

        filtered_count = len(detections)
        total_filtered += filtered_count

        # Save asynchronously
        img_h, img_w = image.shape[:2]
        yolo_lines = detections_to_yolo(detections, img_w, img_h)
        save_futures.append(save_pool.submit(save_result, stem, fname, img_path, yolo_lines))

        if processed % 50 == 0 or processed == len(image_paths):
            elapsed = time.time() - total_start
            rate = processed / elapsed
            eta = (len(image_paths) - processed) / rate if rate > 0 else 0
            print(f"[{processed}/{len(image_paths)}] {infer_time:.1f}s | "
                  f"raw={raw_count} filt={filtered_count} | "
                  f"rate={rate:.1f} img/s | ETA={eta:.0f}s")

        results.append({
            "file": fname,
            "inference_time": round(infer_time, 2),
            "raw": raw_count,
            "filtered": filtered_count,
        })

    except Exception as e:
        print(f"[{processed}/{len(image_paths)}] {fname} ERROR: {e}")
        results.append({"file": fname, "error": str(e)})

# Wait for all saves
for f in save_futures:
    f.result()
save_pool.shutdown()

total_time = time.time() - total_start

# ── Save summary ──
summary = {
    "total_images": len(image_paths),
    "processed": processed,
    "total_time_seconds": round(total_time, 1),
    "avg_time_per_image": round(total_time / max(processed, 1), 2),
    "images_per_second": round(processed / max(total_time, 1), 2),
    "total_raw_detections": total_raw,
    "total_filtered_detections": total_filtered,
    "classes": classes,
    "config": {"min_confidence": MIN_CONFIDENCE, "min_area": MIN_AREA},
    "results": results,
}

with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

with open(os.path.join(OUTPUT_DIR, "classes.txt"), "w") as f:
    f.write("\n".join(classes))

print(f"\n{'='*70}")
print(f"Done! {processed} images in {total_time:.1f}s ({total_time/max(processed,1):.1f}s/img)")
print(f"Raw detections: {total_raw} | Filtered: {total_filtered}")
print(f"Output: {OUTPUT_DIR}/")
