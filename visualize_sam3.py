"""
SAM3 Visualization Script
Runs SAM3 inference on a sample of room images, saves annotated images
with segmentation masks, and generates an index.html for comparison.
"""
import os, sys, time, json, base64, random, glob
import numpy as np

os.environ['ROBOFLOW_API_KEY'] = '6mPyaZWFhvbmBKftxcq7'

import cv2
import supervision as sv
from autodistill_sam3 import SegmentAnything3
from autodistill.detection import CaptionOntology
from autodistill.helpers import load_image

# ── Config ──
IMAGE_DIR = "/teamspace/studios/this_studio/room_images"
OUTPUT_DIR = "/teamspace/studios/this_studio/sam3_viz"
NUM_IMAGES = 15
SEED = 123
MIN_CONFIDENCE = 0.65
MIN_AREA = 400

# 19-class ontology (same as pipeline)
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

# Distinct colors per class (BGR for OpenCV)
CLASS_COLORS = [
    (66, 135, 245),   # ceilings - blue
    (75, 25, 230),    # curtains - red
    (60, 180, 75),    # decor - green
    (0, 130, 200),    # floors - dark blue
    (245, 130, 48),   # upholstery - orange
    (145, 30, 180),   # walls - purple
    (70, 240, 240),   # worktop_surface - cyan
    (240, 50, 230),   # board_accessory - magenta
    (210, 245, 60),   # faucet_tap - lime
    (0, 128, 128),    # fixtures - teal
    (220, 190, 255),  # handle - lavender
    (170, 110, 40),   # knob - brown
    (128, 128, 0),    # other_hardware - olive
    (255, 215, 180),  # outdoor_fabric - coral
    (0, 0, 128),      # outdoor_paver - navy
    (128, 128, 128),  # stair_rod - grey
    (255, 250, 200),  # switch - beige
    (170, 255, 195),  # wallpaper - mint
    (200, 200, 200),  # na - light grey
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Select diverse images ──
all_images = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.*")))
all_images = [p for p in all_images if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
print(f"Found {len(all_images)} images in {IMAGE_DIR}")

random.seed(SEED)
selected = random.sample(all_images, min(NUM_IMAGES, len(all_images)))
print(f"Selected {len(selected)} images for visualization\n")

# ── Load SAM3 ──
print("Loading SAM3 model...")
t0 = time.time()
ontology = CaptionOntology(ONTOLOGY)
model = SegmentAnything3(ontology=ontology)
classes = ontology.classes()
print(f"Model loaded in {time.time()-t0:.1f}s")
print(f"Classes ({len(classes)}): {classes}\n")

# ── Annotators ──
mask_annotator = sv.MaskAnnotator(opacity=0.4)
label_annotator = sv.LabelAnnotator(
    text_position=sv.Position.CENTER,
    text_scale=0.5,
    text_thickness=1,
)
box_annotator = sv.BoxAnnotator(thickness=2)

# ── Run inference and save results ──
results = []

for idx, img_path in enumerate(selected):
    fname = os.path.basename(img_path)
    stem = os.path.splitext(fname)[0]
    print(f"[{idx+1}/{len(selected)}] {fname} ...", end=" ", flush=True)

    try:
        t1 = time.time()
        image = load_image(img_path, return_format="cv2")
        img_h, img_w = image.shape[:2]

        # Run SAM3 inference
        detections = model.predict(img_path)
        infer_time = time.time() - t1
        raw_count = len(detections)

        # Filter by confidence and area
        if len(detections) > 0:
            areas = (detections.xyxy[:, 2] - detections.xyxy[:, 0]) * \
                    (detections.xyxy[:, 3] - detections.xyxy[:, 1])
            mask = (detections.confidence >= MIN_CONFIDENCE) & (areas >= MIN_AREA)
            detections = detections[mask]
            # NMS
            detections = detections.with_nms(threshold=0.5, class_agnostic=True)

        filtered_count = len(detections)
        print(f"{infer_time:.1f}s | raw={raw_count} filtered={filtered_count}")

        # Build labels with confidence
        labels = []
        det_info = []
        for xyxy, conf, cls_id in zip(detections.xyxy, detections.confidence, detections.class_id):
            label = classes[cls_id]
            labels.append(f"{label} {conf:.2f}")
            area = int((xyxy[2]-xyxy[0]) * (xyxy[3]-xyxy[1]))
            det_info.append({
                "class": label,
                "class_id": int(cls_id),
                "confidence": round(float(conf), 3),
                "area_px": area,
            })

        # Annotate: masks + boxes + labels
        annotated = mask_annotator.annotate(scene=image.copy(), detections=detections)
        annotated = box_annotator.annotate(scene=annotated, detections=detections)
        annotated = label_annotator.annotate(
            scene=annotated, detections=detections, labels=labels
        )

        # Save original (resized for web) and annotated
        max_dim = 800
        scale = min(max_dim / img_w, max_dim / img_h, 1.0)
        new_w, new_h = int(img_w * scale), int(img_h * scale)

        orig_resized = cv2.resize(image, (new_w, new_h))
        ann_resized = cv2.resize(annotated, (new_w, new_h))

        orig_path = os.path.join(OUTPUT_DIR, f"{stem}_original.jpg")
        ann_path = os.path.join(OUTPUT_DIR, f"{stem}_annotated.jpg")
        cv2.imwrite(orig_path, orig_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
        cv2.imwrite(ann_path, ann_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])

        results.append({
            "filename": fname,
            "stem": stem,
            "width": img_w,
            "height": img_h,
            "inference_time": round(infer_time, 2),
            "raw_detections": raw_count,
            "filtered_detections": filtered_count,
            "detections": det_info,
            "original_img": f"{stem}_original.jpg",
            "annotated_img": f"{stem}_annotated.jpg",
        })

    except Exception as e:
        print(f"ERROR: {e}")
        results.append({"filename": fname, "stem": stem, "error": str(e)})

# ── Save results JSON ──
with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

# ── Generate index.html ──
print("\nGenerating index.html...")

# Collect class stats across all images
all_class_counts = {}
for r in results:
    if "detections" in r:
        for d in r["detections"]:
            cls = d["class"]
            all_class_counts[cls] = all_class_counts.get(cls, 0) + 1

html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SAM3 Segmentation Visualization — Mattoboard</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f0f0f; color: #e0e0e0; padding: 20px; }
  h1 { text-align: center; margin-bottom: 8px; font-size: 1.8em; color: #fff; }
  .subtitle { text-align: center; color: #888; margin-bottom: 30px; font-size: 0.95em; }
  .stats { display: flex; justify-content: center; gap: 30px; margin-bottom: 30px; flex-wrap: wrap; }
  .stat-card { background: #1a1a1a; border: 1px solid #333; border-radius: 10px; padding: 16px 24px; text-align: center; }
  .stat-card .number { font-size: 2em; font-weight: 700; color: #4fc3f7; }
  .stat-card .label { font-size: 0.85em; color: #888; margin-top: 4px; }
  .class-legend { display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; margin-bottom: 30px; padding: 16px; background: #1a1a1a; border-radius: 10px; border: 1px solid #333; }
  .class-chip { display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; border-radius: 20px; font-size: 0.8em; background: #222; border: 1px solid #444; }
  .class-chip .dot { width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }
  .image-card { background: #1a1a1a; border: 1px solid #333; border-radius: 12px; margin-bottom: 30px; overflow: hidden; }
  .image-card-header { padding: 16px 20px; border-bottom: 1px solid #333; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; }
  .image-card-header h3 { font-size: 1em; color: #fff; }
  .image-card-header .meta { font-size: 0.8em; color: #888; }
  .image-card-header .meta span { margin-left: 12px; }
  .image-pair { display: grid; grid-template-columns: 1fr 1fr; }
  .image-pair .col { position: relative; }
  .image-pair .col img { width: 100%; display: block; cursor: pointer; }
  .image-pair .col .label-tag { position: absolute; top: 10px; left: 10px; background: rgba(0,0,0,0.7); color: #fff; padding: 4px 10px; border-radius: 6px; font-size: 0.75em; font-weight: 600; }
  .det-tags { padding: 12px 20px; display: flex; flex-wrap: wrap; gap: 6px; border-top: 1px solid #333; }
  .det-tag { font-size: 0.75em; padding: 3px 8px; border-radius: 12px; background: #222; border: 1px solid #555; }
  .det-tag .conf { color: #4fc3f7; margin-left: 4px; }
  @media (max-width: 768px) { .image-pair { grid-template-columns: 1fr; } }

  /* Fullscreen overlay */
  .overlay { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.95); z-index: 1000; justify-content: center; align-items: center; cursor: zoom-out; }
  .overlay.active { display: flex; }
  .overlay img { max-width: 95%; max-height: 95%; object-fit: contain; }
</style>
</head>
<body>

<h1>SAM3 Segmentation Visualization</h1>
<p class="subtitle">Mattoboard — 19 Material Classes · SegmentAnything3 Teacher Model</p>

<div class="stats">
  <div class="stat-card">
    <div class="number">""" + str(len([r for r in results if "error" not in r])) + """</div>
    <div class="label">Images Processed</div>
  </div>
  <div class="stat-card">
    <div class="number">""" + str(sum(r.get("filtered_detections", 0) for r in results)) + """</div>
    <div class="label">Total Detections</div>
  </div>
  <div class="stat-card">
    <div class="number">""" + str(len(all_class_counts)) + """</div>
    <div class="label">Classes Detected</div>
  </div>
  <div class="stat-card">
    <div class="number">""" + f"{sum(r.get('inference_time', 0) for r in results) / max(len(results), 1):.1f}s" + """</div>
    <div class="label">Avg Inference Time</div>
  </div>
</div>

<div class="class-legend">
"""

# Add class legend chips with colors
CLASS_COLORS_HEX = [
    "#F58742", "#E61964", "#3CB44B", "#0082C8", "#F5822D",
    "#911EB4", "#F0F046", "#F032E6", "#3CF5DC", "#008080",
    "#FFBEDE", "#AA6E28", "#808000", "#FFD8B4", "#000080",
    "#808080", "#FFFAC8", "#AAFFC3", "#C8C8C8",
]

for i, cls_name in enumerate(classes):
    count = all_class_counts.get(cls_name, 0)
    color = CLASS_COLORS_HEX[i % len(CLASS_COLORS_HEX)]
    html += f'  <div class="class-chip"><span class="dot" style="background:{color}"></span>{cls_name} ({count})</div>\n'

html += "</div>\n\n"

# Add image cards
for r in results:
    if "error" in r:
        continue
    html += f"""<div class="image-card">
  <div class="image-card-header">
    <h3>{r['filename']}</h3>
    <div class="meta">
      <span>{r['width']}×{r['height']}</span>
      <span>⏱ {r['inference_time']}s</span>
      <span>Raw: {r['raw_detections']}</span>
      <span>Filtered: {r['filtered_detections']}</span>
    </div>
  </div>
  <div class="image-pair">
    <div class="col">
      <div class="label-tag">Original</div>
      <img src="{r['original_img']}" alt="Original" onclick="openFullscreen(this)">
    </div>
    <div class="col">
      <div class="label-tag">SAM3 Segmentation</div>
      <img src="{r['annotated_img']}" alt="Annotated" onclick="openFullscreen(this)">
    </div>
  </div>
  <div class="det-tags">
"""
    for d in r["detections"]:
        html += f'    <span class="det-tag">{d["class"]}<span class="conf">{d["confidence"]:.2f}</span></span>\n'
    html += "  </div>\n</div>\n\n"

html += """
<div class="overlay" id="overlay" onclick="closeFullscreen()">
  <img id="overlay-img" src="" alt="Fullscreen">
</div>

<script>
function openFullscreen(el) {
  document.getElementById('overlay-img').src = el.src;
  document.getElementById('overlay').classList.add('active');
}
function closeFullscreen() {
  document.getElementById('overlay').classList.remove('active');
}
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeFullscreen(); });
</script>

</body>
</html>
"""

html_path = os.path.join(OUTPUT_DIR, "index.html")
with open(html_path, "w") as f:
    f.write(html)

print(f"\nDone! Results saved to {OUTPUT_DIR}/")
print(f"Open {html_path} in a browser to view.")
print(f"\nSummary:")
print(f"  Images: {len([r for r in results if 'error' not in r])}")
print(f"  Total detections: {sum(r.get('filtered_detections', 0) for r in results)}")
print(f"  Classes found: {sorted(all_class_counts.keys())}")
