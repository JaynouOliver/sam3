"""
Visualize how YOLO sees SAM3's output.
Shows: Original → SAM3 masks → YOLO seg polygons → Old YOLO bbox format
Outputs compare_outputs.html
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

MIN_CONFIDENCE = 0.65
MIN_AREA = 400


def cv2_to_base64(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    buf = BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def generate_colors(n):
    colors = []
    for i in range(n):
        hue = int(180 * i / n)
        c = np.array([[[hue, 200, 255]]], dtype=np.uint8)
        rgb = cv2.cvtColor(c, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(x) for x in rgb))
    return colors


def mask_to_polygon(mask, img_w, img_h, max_points=100):
    mask_uint8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return None, None
    if len(contour) > max_points:
        epsilon = 0.001 * cv2.arcLength(contour, True)
        contour = cv2.approxPolyDP(contour, epsilon, True)
        while len(contour) > max_points and epsilon < 0.05 * cv2.arcLength(contour, True):
            epsilon *= 1.5
            contour = cv2.approxPolyDP(contour, epsilon, True)
    if len(contour) < 3:
        return None, None
    # Normalized points for YOLO label
    norm_points = []
    for pt in contour.squeeze():
        norm_points.append(pt[0] / img_w)
        norm_points.append(pt[1] / img_h)
    # Pixel points for drawing
    pixel_points = contour.squeeze().astype(np.int32)
    return norm_points, pixel_points


# ═══════════════════════════════════════════════════════════════════
# Load image and run SAM3
# ═══════════════════════════════════════════════════════════════════
print(f"Loading image: {IMAGE_PATH}")
original = cv2.imread(IMAGE_PATH)
img_h, img_w = original.shape[:2]
print(f"Image size: {img_w}x{img_h}")

print("\nLoading SAM3...")
from autodistill_sam3 import SegmentAnything3
from autodistill.detection import CaptionOntology

ontology = CaptionOntology(ONTOLOGY)
sam3_model = SegmentAnything3(ontology=ontology)
classes = ontology.classes()

print("Running SAM3 inference...")
t0 = time.time()
detections = sam3_model.predict(IMAGE_PATH)
sam3_time = time.time() - t0
print(f"SAM3: {len(detections)} raw detections in {sam3_time:.1f}s")

# Filter
if len(detections) > 0:
    areas = (detections.xyxy[:, 2] - detections.xyxy[:, 0]) * \
            (detections.xyxy[:, 3] - detections.xyxy[:, 1])
    mask = (detections.confidence >= MIN_CONFIDENCE) & (areas >= MIN_AREA)
    detections = detections[mask]
    if len(detections) > 0:
        detections = detections.with_nms(threshold=0.5, class_agnostic=True)

print(f"After filtering: {len(detections)} detections")
has_masks = detections.mask is not None and len(detections.mask) > 0
print(f"Has masks: {has_masks}")

colors = generate_colors(len(classes))

# ═══════════════════════════════════════════════════════════════════
# View 1: SAM3 raw pixel masks (what SAM3 actually outputs)
# ═══════════════════════════════════════════════════════════════════
print("\nDrawing SAM3 raw masks...")
sam3_raw_vis = original.copy()
sam3_details = []

for i in range(len(detections)):
    cid = int(detections.class_id[i])
    conf = float(detections.confidence[i])
    cname = classes[cid]
    color = colors[cid % len(colors)]

    if has_masks:
        m = detections.mask[i]
        overlay = sam3_raw_vis.copy()
        overlay[m] = color
        sam3_raw_vis = cv2.addWeighted(sam3_raw_vis, 0.5, overlay, 0.5, 0)
        # Draw mask boundary
        mask_uint8 = m.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(sam3_raw_vis, contours, -1, color, 2)
        mask_pixels = int(np.sum(m))
    else:
        mask_pixels = 0

    x1, y1, x2, y2 = detections.xyxy[i].astype(int)
    label = f"{cname} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(sam3_raw_vis, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
    cv2.putText(sam3_raw_vis, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    sam3_details.append({"class": cname, "conf": conf, "mask_pixels": mask_pixels})

# ═══════════════════════════════════════════════════════════════════
# View 2: YOLO seg polygon labels (what YOLO-seg training sees)
# ═══════════════════════════════════════════════════════════════════
print("Drawing YOLO seg polygon view...")
yolo_seg_vis = original.copy()
yolo_label_lines = []
polygon_details = []

for i in range(len(detections)):
    cid = int(detections.class_id[i])
    conf = float(detections.confidence[i])
    cname = classes[cid]
    color = colors[cid % len(colors)]

    norm_pts, pixel_pts = None, None
    if has_masks:
        norm_pts, pixel_pts = mask_to_polygon(detections.mask[i], img_w, img_h)

    if pixel_pts is not None and norm_pts is not None:
        # Draw filled polygon with transparency
        overlay = yolo_seg_vis.copy()
        cv2.fillPoly(overlay, [pixel_pts], color)
        yolo_seg_vis = cv2.addWeighted(yolo_seg_vis, 0.55, overlay, 0.45, 0)
        # Draw polygon outline
        cv2.polylines(yolo_seg_vis, [pixel_pts], isClosed=True, color=color, thickness=2)
        # Draw vertices
        for pt in pixel_pts:
            cv2.circle(yolo_seg_vis, tuple(pt), 3, (255, 255, 255), -1)
            cv2.circle(yolo_seg_vis, tuple(pt), 3, color, 1)

        coords_str = " ".join(f"{v:.6f}" for v in norm_pts)
        yolo_label_lines.append(f"{cid} {coords_str}")
        num_pts = len(norm_pts) // 2
    else:
        # Fallback to bbox polygon
        x1, y1, x2, y2 = detections.xyxy[i]
        pts = np.array([[int(x1), int(y1)], [int(x2), int(y1)],
                        [int(x2), int(y2)], [int(x1), int(y2)]], np.int32)
        overlay = yolo_seg_vis.copy()
        cv2.fillPoly(overlay, [pts], color)
        yolo_seg_vis = cv2.addWeighted(yolo_seg_vis, 0.55, overlay, 0.45, 0)
        cv2.polylines(yolo_seg_vis, [pts], True, color, 2)
        norm_pts = [x1/img_w, y1/img_h, x2/img_w, y1/img_h,
                    x2/img_w, y2/img_h, x1/img_w, y2/img_h]
        coords_str = " ".join(f"{v:.6f}" for v in norm_pts)
        yolo_label_lines.append(f"{cid} {coords_str}")
        num_pts = 4

    # Label
    bx1, by1 = int(detections.xyxy[i][0]), int(detections.xyxy[i][1])
    label = f"{cname} ({num_pts} pts)"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(yolo_seg_vis, (bx1, by1 - th - 6), (bx1 + tw, by1), color, -1)
    cv2.putText(yolo_seg_vis, label, (bx1, by1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    polygon_details.append({"class": cname, "conf": conf, "num_points": num_pts})

# ═══════════════════════════════════════════════════════════════════
# View 3: Old YOLO bbox format (what the pipeline USED to export)
# ═══════════════════════════════════════════════════════════════════
print("Drawing old YOLO bbox view...")
yolo_bbox_vis = original.copy()
bbox_label_lines = []

for i in range(len(detections)):
    cid = int(detections.class_id[i])
    conf = float(detections.confidence[i])
    cname = classes[cid]
    color = colors[cid % len(colors)]

    x1, y1, x2, y2 = detections.xyxy[i].astype(int)
    cv2.rectangle(yolo_bbox_vis, (x1, y1), (x2, y2), color, 2)

    # YOLO bbox label
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    bbox_label_lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    label = f"{cname} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(yolo_bbox_vis, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
    cv2.putText(yolo_bbox_vis, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

# ═══════════════════════════════════════════════════════════════════
# Generate HTML
# ═══════════════════════════════════════════════════════════════════
print("\nGenerating HTML...")

orig_b64 = cv2_to_base64(original)
sam3_b64 = cv2_to_base64(sam3_raw_vis)
yolo_seg_b64 = cv2_to_base64(yolo_seg_vis)
yolo_bbox_b64 = cv2_to_base64(yolo_bbox_vis)

# Build label file previews (show first 8 lines)
seg_label_preview = "\n".join(yolo_label_lines[:8])
if len(yolo_label_lines) > 8:
    seg_label_preview += f"\n... ({len(yolo_label_lines)} total lines)"

bbox_label_preview = "\n".join(bbox_label_lines[:8])
if len(bbox_label_lines) > 8:
    bbox_label_preview += f"\n... ({len(bbox_label_lines)} total lines)"

# Class legend
legend_html = ""
for i, cname in enumerate(classes):
    c = colors[i % len(colors)]
    count = sum(1 for d in sam3_details if d["class"] == cname)
    if count > 0:
        legend_html += f'<span style="display:inline-block;margin:3px 6px;padding:3px 10px;border-radius:4px;font-size:0.8em;background:rgb({c[2]},{c[1]},{c[0]});color:#fff;font-weight:600;">{cname} ({count})</span>'

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>How YOLO Sees SAM3's Output</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0a0a0a; color: #e0e0e0; padding: 20px; }}
  h1 {{ text-align: center; margin-bottom: 6px; font-size: 1.8em; color: #fff; }}
  .subtitle {{ text-align: center; color: #888; margin-bottom: 10px; }}
  .legend {{ text-align: center; margin-bottom: 25px; padding: 10px; }}
  .flow {{ text-align: center; margin-bottom: 30px; padding: 16px; background: #151515; border-radius: 12px; border: 1px solid #333; max-width: 1000px; margin-left: auto; margin-right: auto; }}
  .flow-arrow {{ color: #4ade80; font-size: 1.5em; margin: 0 8px; }}
  .flow-step {{ display: inline-block; padding: 8px 16px; background: #1a1a1a; border-radius: 8px; border: 1px solid #333; font-size: 0.9em; }}
  .flow-step.active {{ border-color: #4ade80; color: #4ade80; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; max-width: 1800px; margin: 0 auto; }}
  .card {{ background: #1a1a1a; border-radius: 12px; overflow: hidden; border: 1px solid #333; }}
  .card.highlight {{ border-color: #4ade80; }}
  .card.old {{ border-color: #f87171; opacity: 0.85; }}
  .card-header {{ padding: 14px 20px; border-bottom: 1px solid #333; display: flex; justify-content: space-between; align-items: center; }}
  .card-header h2 {{ font-size: 1.05em; }}
  .badge {{ padding: 4px 12px; border-radius: 20px; font-size: 0.75em; font-weight: 600; }}
  .badge-green {{ background: #1a3f2a; color: #4ade80; }}
  .badge-blue {{ background: #1e3a5f; color: #60a5fa; }}
  .badge-red {{ background: #3f1a1a; color: #f87171; }}
  .badge-gray {{ background: #2a2a2a; color: #999; }}
  .card img {{ width: 100%; display: block; }}
  .stats {{ padding: 14px 20px; }}
  .stat-row {{ display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid #222; font-size: 0.85em; }}
  .stat-row:last-child {{ border: none; }}
  .stat-label {{ color: #888; }}
  .stat-value {{ font-weight: 600; }}
  .label-preview {{ padding: 12px 20px 16px; }}
  .label-preview h3 {{ font-size: 0.8em; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }}
  .label-preview pre {{ background: #111; padding: 12px; border-radius: 8px; font-size: 0.75em; overflow-x: auto; line-height: 1.5; color: #ccc; max-height: 180px; overflow-y: auto; }}
  .winner {{ color: #4ade80; }}
  .loser {{ color: #f87171; }}
  .comparison {{ max-width: 1000px; margin: 30px auto; background: #1a1a1a; border-radius: 12px; border: 1px solid #333; overflow: hidden; }}
  .comparison table {{ width: 100%; border-collapse: collapse; }}
  .comparison th {{ background: #222; padding: 10px 16px; text-align: left; font-size: 0.85em; color: #888; text-transform: uppercase; }}
  .comparison td {{ padding: 10px 16px; border-top: 1px solid #222; font-size: 0.9em; }}
</style>
</head>
<body>

<h1>How YOLO Sees SAM3's Output</h1>
<p class="subtitle">SAM3 pixel masks → converted to polygon labels → fed to YOLO-seg training</p>

<div class="flow">
  <span class="flow-step">SAM3 predicts pixel masks</span>
  <span class="flow-arrow">&rarr;</span>
  <span class="flow-step active">Masks converted to polygons</span>
  <span class="flow-arrow">&rarr;</span>
  <span class="flow-step">Saved as .txt label files</span>
  <span class="flow-arrow">&rarr;</span>
  <span class="flow-step">YOLOv8n-seg trains on them</span>
</div>

<div class="legend">{legend_html}</div>

<div class="grid">

<!-- Original -->
<div class="card">
  <div class="card-header">
    <h2>1. Original Image</h2>
    <span class="badge badge-gray">{img_w}x{img_h}</span>
  </div>
  <img src="data:image/png;base64,{orig_b64}">
  <div class="stats">
    <div class="stat-row"><span class="stat-label">Input to SAM3</span><span class="stat-value">Room scene photo</span></div>
    <div class="stat-row"><span class="stat-label">SAM3 prompts</span><span class="stat-value">{len(ONTOLOGY)} text queries</span></div>
  </div>
</div>

<!-- SAM3 raw masks -->
<div class="card">
  <div class="card-header">
    <h2>2. SAM3 Output: Pixel Masks</h2>
    <span class="badge badge-blue">{len(detections)} detections &middot; {sam3_time:.1f}s</span>
  </div>
  <img src="data:image/png;base64,{sam3_b64}">
  <div class="stats">
    <div class="stat-row"><span class="stat-label">Output type</span><span class="stat-value">Binary pixel mask per detection</span></div>
    <div class="stat-row"><span class="stat-label">Mask resolution</span><span class="stat-value">{img_w}x{img_h} (full image)</span></div>
    <div class="stat-row"><span class="stat-label">Every pixel labeled</span><span class="stat-value winner">Yes — exact boundaries</span></div>
    <div class="stat-row"><span class="stat-label">Total mask pixels</span><span class="stat-value">{sum(d["mask_pixels"] for d in sam3_details):,}</span></div>
  </div>
</div>

<!-- YOLO seg polygons (NEW) -->
<div class="card highlight">
  <div class="card-header">
    <h2>3. YOLO Sees: Segmentation Polygons (NEW)</h2>
    <span class="badge badge-green">Polygon format &middot; {len(yolo_label_lines)} objects</span>
  </div>
  <img src="data:image/png;base64,{yolo_seg_b64}">
  <div class="stats">
    <div class="stat-row"><span class="stat-label">Format</span><span class="stat-value winner">class_id x1 y1 x2 y2 x3 y3 ...</span></div>
    <div class="stat-row"><span class="stat-label">Shape fidelity</span><span class="stat-value winner">High — follows mask contour</span></div>
    <div class="stat-row"><span class="stat-label">Avg polygon points</span><span class="stat-value">{sum(d["num_points"] for d in polygon_details) / max(len(polygon_details), 1):.0f} vertices per object</span></div>
    <div class="stat-row"><span class="stat-label">White dots = polygon vertices</span><span class="stat-value">What YOLO trains on</span></div>
  </div>
  <div class="label-preview">
    <h3>Actual .txt label file (YOLO seg format)</h3>
    <pre>{seg_label_preview}</pre>
  </div>
</div>

<!-- Old YOLO bbox (OLD) -->
<div class="card old">
  <div class="card-header">
    <h2>4. YOLO Used to See: Bounding Boxes (OLD)</h2>
    <span class="badge badge-red">Bbox format &middot; {len(bbox_label_lines)} objects</span>
  </div>
  <img src="data:image/png;base64,{yolo_bbox_b64}">
  <div class="stats">
    <div class="stat-row"><span class="stat-label">Format</span><span class="stat-value loser">class_id cx cy w h</span></div>
    <div class="stat-row"><span class="stat-label">Shape fidelity</span><span class="stat-value loser">Low — rectangles only</span></div>
    <div class="stat-row"><span class="stat-label">Mask info</span><span class="stat-value loser">Completely lost</span></div>
    <div class="stat-row"><span class="stat-label">Overlap issues</span><span class="stat-value loser">Boxes overlap heavily</span></div>
  </div>
  <div class="label-preview">
    <h3>Old .txt label file (YOLO bbox format)</h3>
    <pre>{bbox_label_preview}</pre>
  </div>
</div>

</div>

<!-- Comparison table -->
<div class="comparison" style="margin-top: 30px;">
<table>
<tr><th>What changed</th><th>Before (bbox)</th><th>After (seg polygon)</th></tr>
<tr>
  <td>Label format</td>
  <td class="loser">class cx cy w h (5 values)</td>
  <td class="winner">class x1 y1 x2 y2 ... (variable polygon)</td>
</tr>
<tr>
  <td>Model</td>
  <td class="loser">yolov8n.pt (detection)</td>
  <td class="winner">yolov8n-seg.pt (segmentation)</td>
</tr>
<tr>
  <td>Output at inference</td>
  <td class="loser">Bounding boxes only</td>
  <td class="winner">Segmentation masks + boxes</td>
</tr>
<tr>
  <td>SAM3 mask quality preserved</td>
  <td class="loser">No — reduced to rectangles</td>
  <td class="winner">Yes — polygon approximation of mask contour</td>
</tr>
<tr>
  <td>Shape accuracy</td>
  <td class="loser">Coarse (lots of background in box)</td>
  <td class="winner">Fine (follows object boundary)</td>
</tr>
</table>
</div>

<div class="comparison" style="margin-top: 20px; margin-bottom: 40px;">
<table>
<tr><th colspan="2">What This Means</th></tr>
<tr><td colspan="2" style="line-height: 1.8; padding: 20px;">
The <span class="winner">green-bordered panel (3)</span> is what YOLOv8n-seg will now train on.
Each SAM3 pixel mask is converted to a polygon contour (white dots = vertices),
and saved in YOLO segmentation label format. The trained model will output
<strong>segmentation masks</strong> at inference time — not just boxes.<br><br>
The <span class="loser">red-bordered panel (4)</span> shows what the old pipeline did:
threw away all mask detail and reduced everything to rectangles.
That information is now preserved.
</td></tr>
</table>
</div>

</body>
</html>"""

with open(OUTPUT_HTML, "w") as f:
    f.write(html)

print(f"\nHTML saved to: {OUTPUT_HTML}")
print("Open it in your browser to preview.")
