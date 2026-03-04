"""
Compare SAM3 vs YOLOv8n-seg outputs on a single image.
Generates an HTML file showing both side-by-side.
"""
import os, sys, time, json, base64
from io import BytesIO

os.environ['ROBOFLOW_API_KEY'] = '6mPyaZWFhvbmBKftxcq7'

import cv2
import numpy as np
from PIL import Image

IMAGE_PATH = "/teamspace/studios/this_studio/room_images/0091ff0b-717a-460c-acc3-989b556caf29_room.png"
OUTPUT_HTML = "/teamspace/studios/this_studio/compare_outputs.html"

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


def cv2_to_base64(img):
    """Convert OpenCV image to base64 string for HTML embedding."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    buf = BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def generate_colors(n):
    """Generate n distinct colors."""
    colors = []
    for i in range(n):
        hue = int(180 * i / n)
        c = np.array([[[hue, 200, 255]]], dtype=np.uint8)
        rgb = cv2.cvtColor(c, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(x) for x in rgb))
    return colors


# Load original image
print(f"Loading image: {IMAGE_PATH}")
original = cv2.imread(IMAGE_PATH)
if original is None:
    print("ERROR: Could not load image")
    sys.exit(1)
img_h, img_w = original.shape[:2]
print(f"Image size: {img_w}x{img_h}")

# ═══════════════════════════════════════════════════════════════════
# SAM3 INFERENCE
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("Running SAM3...")
print("=" * 50)

from autodistill_sam3 import SegmentAnything3
from autodistill.detection import CaptionOntology
import supervision as sv

ontology = CaptionOntology(ONTOLOGY)
sam3_model = SegmentAnything3(ontology=ontology)
sam3_classes = ontology.classes()

t0 = time.time()
sam3_detections = sam3_model.predict(IMAGE_PATH)
sam3_time = time.time() - t0
print(f"SAM3 inference: {sam3_time:.2f}s")
print(f"SAM3 detections: {len(sam3_detections)}")

# Check what SAM3 actually returns
sam3_has_masks = sam3_detections.mask is not None and len(sam3_detections.mask) > 0
print(f"SAM3 has masks: {sam3_has_masks}")
print(f"SAM3 has boxes: {sam3_detections.xyxy is not None and len(sam3_detections.xyxy) > 0}")

# Draw SAM3 output
sam3_colors = generate_colors(len(sam3_classes))
sam3_vis = original.copy()
sam3_details = []

if len(sam3_detections) > 0:
    for i in range(len(sam3_detections)):
        class_id = int(sam3_detections.class_id[i])
        conf = float(sam3_detections.confidence[i])
        class_name = sam3_classes[class_id] if class_id < len(sam3_classes) else f"class_{class_id}"
        color = sam3_colors[class_id % len(sam3_colors)]

        # Draw mask if available
        if sam3_has_masks:
            mask = sam3_detections.mask[i]
            overlay = sam3_vis.copy()
            overlay[mask] = color
            sam3_vis = cv2.addWeighted(sam3_vis, 0.55, overlay, 0.45, 0)

        # Draw bbox
        x1, y1, x2, y2 = sam3_detections.xyxy[i].astype(int)
        cv2.rectangle(sam3_vis, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(sam3_vis, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(sam3_vis, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Mask pixel count
        mask_pixels = int(np.sum(mask)) if sam3_has_masks else 0
        sam3_details.append({
            "class": class_name,
            "confidence": round(conf, 3),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "has_mask": sam3_has_masks,
            "mask_pixels": mask_pixels,
        })

print(f"SAM3 drawn with {'masks + boxes' if sam3_has_masks else 'boxes only'}")

# ═══════════════════════════════════════════════════════════════════
# YOLOv8n-seg INFERENCE (pretrained on COCO)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("Running YOLOv8n-seg (COCO pretrained)...")
print("=" * 50)

from ultralytics import YOLO

yolo_seg = YOLO("yolov8n-seg.pt")

t0 = time.time()
yolo_results = yolo_seg.predict(IMAGE_PATH, verbose=False)
yolo_time = time.time() - t0
print(f"YOLOv8n-seg inference: {yolo_time:.2f}s")

result = yolo_results[0]
yolo_names = result.names
yolo_has_masks = result.masks is not None
print(f"YOLOv8n-seg detections: {len(result.boxes)}")
print(f"YOLOv8n-seg has masks: {yolo_has_masks}")

# Draw YOLOv8-seg output
yolo_colors = generate_colors(80)  # COCO has 80 classes
yolo_vis = original.copy()
yolo_details = []

if len(result.boxes) > 0:
    for i in range(len(result.boxes)):
        class_id = int(result.boxes.cls[i])
        conf = float(result.boxes.conf[i])
        class_name = yolo_names[class_id]
        color = yolo_colors[class_id % len(yolo_colors)]

        # Draw mask if available
        if yolo_has_masks and i < len(result.masks.data):
            mask_tensor = result.masks.data[i].cpu().numpy()
            # Resize mask to image size
            mask = cv2.resize(mask_tensor, (img_w, img_h)) > 0.5
            overlay = yolo_vis.copy()
            overlay[mask] = color
            yolo_vis = cv2.addWeighted(yolo_vis, 0.55, overlay, 0.45, 0)
            mask_pixels = int(np.sum(mask))
        else:
            mask_pixels = 0

        # Draw bbox
        x1, y1, x2, y2 = result.boxes.xyxy[i].cpu().numpy().astype(int)
        cv2.rectangle(yolo_vis, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(yolo_vis, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(yolo_vis, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        yolo_details.append({
            "class": class_name,
            "confidence": round(conf, 3),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "has_mask": yolo_has_masks,
            "mask_pixels": mask_pixels,
        })

print(f"YOLOv8n-seg drawn with {'masks + boxes' if yolo_has_masks else 'boxes only'}")

# ═══════════════════════════════════════════════════════════════════
# ALSO RUN YOLOv8n (detection only) for comparison
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("Running YOLOv8n (detection only, COCO pretrained)...")
print("=" * 50)

yolo_det = YOLO("yolov8n.pt")

t0 = time.time()
yolo_det_results = yolo_det.predict(IMAGE_PATH, verbose=False)
yolo_det_time = time.time() - t0
print(f"YOLOv8n inference: {yolo_det_time:.2f}s")

det_result = yolo_det_results[0]
yolo_det_vis = original.copy()
yolo_det_details = []

if len(det_result.boxes) > 0:
    for i in range(len(det_result.boxes)):
        class_id = int(det_result.boxes.cls[i])
        conf = float(det_result.boxes.conf[i])
        class_name = yolo_names.get(class_id, f"class_{class_id}")
        color = yolo_colors[class_id % len(yolo_colors)]

        x1, y1, x2, y2 = det_result.boxes.xyxy[i].cpu().numpy().astype(int)
        cv2.rectangle(yolo_det_vis, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(yolo_det_vis, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(yolo_det_vis, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        yolo_det_details.append({
            "class": class_name,
            "confidence": round(conf, 3),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "has_mask": False,
            "mask_pixels": 0,
        })

# ═══════════════════════════════════════════════════════════════════
# GENERATE HTML
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("Generating HTML comparison...")
print("=" * 50)

orig_b64 = cv2_to_base64(original)
sam3_b64 = cv2_to_base64(sam3_vis)
yolo_seg_b64 = cv2_to_base64(yolo_vis)
yolo_det_b64 = cv2_to_base64(yolo_det_vis)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SAM3 vs YOLOv8-seg vs YOLOv8 — Output Comparison</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f0f0f; color: #e0e0e0; padding: 20px; }}
  h1 {{ text-align: center; margin-bottom: 8px; font-size: 1.8em; color: #fff; }}
  .subtitle {{ text-align: center; color: #888; margin-bottom: 30px; font-size: 0.95em; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; max-width: 1800px; margin: 0 auto; }}
  .card {{ background: #1a1a1a; border-radius: 12px; overflow: hidden; border: 1px solid #333; }}
  .card-header {{ padding: 16px 20px; border-bottom: 1px solid #333; display: flex; justify-content: space-between; align-items: center; }}
  .card-header h2 {{ font-size: 1.1em; }}
  .badge {{ padding: 4px 12px; border-radius: 20px; font-size: 0.8em; font-weight: 600; }}
  .badge-blue {{ background: #1e3a5f; color: #60a5fa; }}
  .badge-green {{ background: #1a3f2a; color: #4ade80; }}
  .badge-orange {{ background: #3f2a1a; color: #fb923c; }}
  .badge-gray {{ background: #2a2a2a; color: #999; }}
  .card img {{ width: 100%; display: block; }}
  .stats {{ padding: 16px 20px; }}
  .stat-row {{ display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #222; font-size: 0.9em; }}
  .stat-row:last-child {{ border: none; }}
  .stat-label {{ color: #888; }}
  .stat-value {{ font-weight: 600; }}
  .detections {{ padding: 12px 20px 16px; }}
  .detections h3 {{ font-size: 0.85em; color: #888; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px; }}
  .det-list {{ max-height: 200px; overflow-y: auto; }}
  .det-item {{ display: flex; justify-content: space-between; padding: 4px 8px; font-size: 0.85em; border-radius: 4px; margin-bottom: 2px; }}
  .det-item:nth-child(odd) {{ background: #222; }}
  .highlight {{ color: #fbbf24; font-weight: 600; }}
  .comparison-table {{ max-width: 900px; margin: 30px auto; background: #1a1a1a; border-radius: 12px; border: 1px solid #333; overflow: hidden; }}
  .comparison-table table {{ width: 100%; border-collapse: collapse; }}
  .comparison-table th {{ background: #222; padding: 12px 16px; text-align: left; font-size: 0.9em; color: #888; text-transform: uppercase; letter-spacing: 1px; }}
  .comparison-table td {{ padding: 10px 16px; border-top: 1px solid #2a2a2a; font-size: 0.95em; }}
  .comparison-table tr:hover td {{ background: #222; }}
  .winner {{ color: #4ade80; font-weight: 600; }}
</style>
</head>
<body>

<h1>SAM3 vs YOLOv8-seg vs YOLOv8 Detection</h1>
<p class="subtitle">Same image, three models — comparing output types and quality</p>

<!-- Summary Table -->
<div class="comparison-table">
<table>
<tr>
  <th>Metric</th>
  <th>SAM3 (Teacher)</th>
  <th>YOLOv8n-seg</th>
  <th>YOLOv8n (det only)</th>
</tr>
<tr>
  <td>Inference Time</td>
  <td>{sam3_time:.2f}s</td>
  <td class="winner">{yolo_time*1000:.0f}ms</td>
  <td class="winner">{yolo_det_time*1000:.0f}ms</td>
</tr>
<tr>
  <td>Detections</td>
  <td>{len(sam3_detections)}</td>
  <td>{len(result.boxes)}</td>
  <td>{len(det_result.boxes)}</td>
</tr>
<tr>
  <td>Output Type</td>
  <td class="winner">Masks + Boxes</td>
  <td class="winner">Masks + Boxes</td>
  <td>Boxes only</td>
</tr>
<tr>
  <td>Has Segmentation Masks</td>
  <td class="winner">{"Yes" if sam3_has_masks else "No"}</td>
  <td class="winner">{"Yes" if yolo_has_masks else "No"}</td>
  <td>No</td>
</tr>
<tr>
  <td>Classes</td>
  <td>19 custom (your ontology)</td>
  <td>80 COCO (generic)</td>
  <td>80 COCO (generic)</td>
</tr>
<tr>
  <td>Trained On Your Data?</td>
  <td>Zero-shot (no training)</td>
  <td>No (COCO pretrained)</td>
  <td>No (COCO pretrained)</td>
</tr>
</table>
</div>

<div class="grid">

<!-- Original -->
<div class="card">
  <div class="card-header">
    <h2>Original Image</h2>
    <span class="badge badge-gray">{img_w}x{img_h}</span>
  </div>
  <img src="data:image/png;base64,{orig_b64}">
</div>

<!-- SAM3 -->
<div class="card">
  <div class="card-header">
    <h2>SAM3 (Teacher)</h2>
    <span class="badge badge-blue">{sam3_time:.2f}s &middot; {len(sam3_detections)} detections &middot; {"MASKS" if sam3_has_masks else "BOXES"}</span>
  </div>
  <img src="data:image/png;base64,{sam3_b64}">
  <div class="stats">
    <div class="stat-row"><span class="stat-label">Inference</span><span class="stat-value">{sam3_time:.2f}s</span></div>
    <div class="stat-row"><span class="stat-label">Detections</span><span class="stat-value">{len(sam3_detections)}</span></div>
    <div class="stat-row"><span class="stat-label">Has Masks</span><span class="stat-value highlight">{"Yes — pixel-level segmentation" if sam3_has_masks else "No — boxes only"}</span></div>
    <div class="stat-row"><span class="stat-label">Classes</span><span class="stat-value">19 custom (your ontology)</span></div>
  </div>
  <div class="detections">
    <h3>Detections</h3>
    <div class="det-list">
      {"".join(f'<div class="det-item"><span>{d["class"]}</span><span>{d["confidence"]:.2f} {"| " + str(d["mask_pixels"]) + "px mask" if d["mask_pixels"] > 0 else ""}</span></div>' for d in sam3_details)}
    </div>
  </div>
</div>

<!-- YOLOv8n-seg -->
<div class="card">
  <div class="card-header">
    <h2>YOLOv8n-seg (COCO)</h2>
    <span class="badge badge-green">{yolo_time*1000:.0f}ms &middot; {len(result.boxes)} detections &middot; {"MASKS" if yolo_has_masks else "BOXES"}</span>
  </div>
  <img src="data:image/png;base64,{yolo_seg_b64}">
  <div class="stats">
    <div class="stat-row"><span class="stat-label">Inference</span><span class="stat-value">{yolo_time*1000:.0f}ms</span></div>
    <div class="stat-row"><span class="stat-label">Detections</span><span class="stat-value">{len(result.boxes)}</span></div>
    <div class="stat-row"><span class="stat-label">Has Masks</span><span class="stat-value highlight">{"Yes — instance segmentation" if yolo_has_masks else "No — boxes only"}</span></div>
    <div class="stat-row"><span class="stat-label">Classes</span><span class="stat-value">80 COCO (not your classes)</span></div>
  </div>
  <div class="detections">
    <h3>Detections</h3>
    <div class="det-list">
      {"".join(f'<div class="det-item"><span>{d["class"]}</span><span>{d["confidence"]:.2f} {"| " + str(d["mask_pixels"]) + "px mask" if d["mask_pixels"] > 0 else ""}</span></div>' for d in yolo_details)}
    </div>
  </div>
</div>

<!-- YOLOv8n detection -->
<div class="card">
  <div class="card-header">
    <h2>YOLOv8n Detection (COCO)</h2>
    <span class="badge badge-orange">{yolo_det_time*1000:.0f}ms &middot; {len(det_result.boxes)} detections &middot; BOXES ONLY</span>
  </div>
  <img src="data:image/png;base64,{yolo_det_b64}">
  <div class="stats">
    <div class="stat-row"><span class="stat-label">Inference</span><span class="stat-value">{yolo_det_time*1000:.0f}ms</span></div>
    <div class="stat-row"><span class="stat-label">Detections</span><span class="stat-value">{len(det_result.boxes)}</span></div>
    <div class="stat-row"><span class="stat-label">Has Masks</span><span class="stat-value">No — bounding boxes only</span></div>
    <div class="stat-row"><span class="stat-label">Classes</span><span class="stat-value">80 COCO (not your classes)</span></div>
  </div>
  <div class="detections">
    <h3>Detections</h3>
    <div class="det-list">
      {"".join(f'<div class="det-item"><span>{d["class"]}</span><span>{d["confidence"]:.2f}</span></div>' for d in yolo_det_details)}
    </div>
  </div>
</div>

</div>

<div class="comparison-table" style="margin-top: 30px;">
<table>
<tr><th colspan="2">Key Takeaway</th></tr>
<tr><td colspan="2" style="line-height: 1.8; padding: 20px;">
<strong>SAM3</strong> gives you <span class="winner">pixel-perfect segmentation masks</span> with your 19 custom classes — but it's slow ({sam3_time:.1f}s).<br>
<strong>YOLOv8n-seg</strong> gives masks too and is <span class="winner">{sam3_time/yolo_time:.0f}x faster</span> — but it only knows COCO's 80 generic classes (couch, chair, tv...), not your material categories.<br>
<strong>YOLOv8n</strong> (detection) gives <strong>boxes only</strong> — no segmentation masks at all.<br><br>
<strong>The plan:</strong> Use SAM3 to auto-label your 1,000 room images <em>with segmentation polygons</em>, then train YOLOv8n-seg on that data.
The result: a model with SAM3's custom classes + mask quality, at YOLOv8's speed.
</td></tr>
</table>
</div>

</body>
</html>"""

with open(OUTPUT_HTML, "w") as f:
    f.write(html)

print(f"\nHTML saved to: {OUTPUT_HTML}")
print("Open it in your browser to see the comparison.")
