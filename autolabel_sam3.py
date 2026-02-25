"""
SAM3 Auto-Labeling Pipeline
Runs SAM3 on all firestore_gen images using 19 material concepts.
Outputs YOLO-format labels for distillation training.
"""
import os, time, json, glob
import numpy as np

os.environ['ROBOFLOW_API_KEY'] = '6mPyaZWFhvbmBKftxcq7'

from autodistill_sam3 import SegmentAnything3
from autodistill.detection import CaptionOntology
from autodistill.helpers import load_image
import supervision as sv

# ── 19 Material Concepts ──
# Keys = SAM3 prompts (descriptive for better detection)
# Values = class labels (clean names for the student model)
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

# ── Config ──
IMAGE_DIR = "/teamspace/studios/this_studio/firestore_gen"
OUTPUT_DIR = "/teamspace/studios/this_studio/sam3_labels"
MIN_CONFIDENCE = 0.7
MIN_AREA = 500  # pixels

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels"), exist_ok=True)

# ── Load model ──
print("Loading SAM3 model...")
t0 = time.time()
ontology = CaptionOntology(ONTOLOGY)
model = SegmentAnything3(ontology=ontology)
classes = ontology.classes()
print(f"Model loaded in {time.time()-t0:.1f}s")
print(f"Classes ({len(classes)}): {classes}\n")

# ── Find all images ──
extensions = ["*.png", "*.jpg", "*.jpeg"]
image_paths = []
for ext in extensions:
    image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
image_paths.sort()
print(f"Found {len(image_paths)} images\n")


def detections_to_yolo(detections, img_width, img_height):
    """Convert sv.Detections to YOLO format lines."""
    lines = []
    for xyxy, conf, class_id in zip(detections.xyxy, detections.confidence, detections.class_id):
        x1, y1, x2, y2 = xyxy
        cx = ((x1 + x2) / 2) / img_width
        cy = ((y1 + y2) / 2) / img_height
        w = (x2 - x1) / img_width
        h = (y2 - y1) / img_height
        lines.append(f"{int(class_id)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


# ── Run inference on all images ──
results_summary = []
total_start = time.time()

for idx, img_path in enumerate(image_paths):
    fname = os.path.basename(img_path)
    stem = os.path.splitext(fname)[0]
    print(f"[{idx+1}/{len(image_paths)}] {fname} ...", end=" ", flush=True)

    try:
        t1 = time.time()
        detections = model.predict(img_path)
        infer_time = time.time() - t1
        raw_count = len(detections)

        # Filter: confidence + area
        areas = (detections.xyxy[:, 2] - detections.xyxy[:, 0]) * \
                (detections.xyxy[:, 3] - detections.xyxy[:, 1])
        mask = (detections.confidence >= MIN_CONFIDENCE) & (areas >= MIN_AREA)
        detections = detections[mask]

        # Class-agnostic NMS to remove cross-class overlaps
        detections = detections.with_nms(threshold=0.5, class_agnostic=True)

        # Get image dimensions
        image = load_image(img_path, return_format="cv2")
        img_h, img_w = image.shape[:2]

        # Save YOLO labels
        yolo_lines = detections_to_yolo(detections, img_w, img_h)
        label_path = os.path.join(OUTPUT_DIR, "labels", f"{stem}.txt")
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines))

        # Symlink image
        img_dest = os.path.join(OUTPUT_DIR, "images", fname)
        if not os.path.exists(img_dest):
            os.symlink(img_path, img_dest)

        print(f"{infer_time:.1f}s | raw={raw_count} filtered={len(detections)}")
        results_summary.append({
            "file": fname,
            "inference_time": round(infer_time, 2),
            "raw_detections": raw_count,
            "filtered_detections": len(detections),
        })

    except Exception as e:
        print(f"ERROR: {e}")
        results_summary.append({"file": fname, "error": str(e)})

total_time = time.time() - total_start

# ── Save summary ──
summary = {
    "total_images": len(image_paths),
    "total_time_seconds": round(total_time, 1),
    "avg_time_per_image": round(total_time / len(image_paths), 2),
    "classes": classes,
    "num_classes": len(classes),
    "config": {"min_confidence": MIN_CONFIDENCE, "min_area": MIN_AREA},
    "results": results_summary,
}

summary_path = os.path.join(OUTPUT_DIR, "summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

# Save classes.txt for YOLO training
classes_path = os.path.join(OUTPUT_DIR, "classes.txt")
with open(classes_path, "w") as f:
    f.write("\n".join(classes))

print(f"\n{'='*50}")
print(f"Done! {len(image_paths)} images in {total_time:.1f}s ({total_time/len(image_paths):.1f}s/img)")
print(f"Labels saved to: {OUTPUT_DIR}/labels/")
print(f"Classes file: {classes_path}")
print(f"Summary: {summary_path}")
