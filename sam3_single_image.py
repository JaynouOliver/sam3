"""
Run SAM3 on a single image with the 19-class ontology and generate HTML visualization.
"""

import os
import time
import base64
import colorsys
import io

import cv2
import numpy as np
from PIL import Image, ImageDraw

os.environ['ROBOFLOW_API_KEY'] = '6mPyaZWFhvbmBKftxcq7'

IMAGE_PATH = "room_images/bead687f-9d5a-4385-b36e-947bb9910189_room.png"
OUTPUT_HTML = "sam3_single_result.html"
MIN_CONFIDENCE = 0.65
MIN_AREA = 400

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

CLASS_NAMES = list(ONTOLOGY.values())

def get_color(class_id, num_classes=19):
    hue = class_id / num_classes
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return (int(r * 255), int(g * 255), int(b * 255))

COLORS = {name: get_color(i) for i, name in enumerate(CLASS_NAMES)}


def img_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    from autodistill_sam3 import SegmentAnything3
    from autodistill.detection import CaptionOntology
    from autodistill.helpers import load_image
    import supervision as sv

    print(f"Loading SAM3 with {len(ONTOLOGY)} classes...")
    t0 = time.time()
    ontology = CaptionOntology(ONTOLOGY)
    model = SegmentAnything3(ontology=ontology)
    classes = ontology.classes()
    print(f"Loaded in {time.time()-t0:.1f}s")

    # Load image
    print(f"\nRunning inference on: {IMAGE_PATH}")
    img = cv2.imread(IMAGE_PATH)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    t0 = time.time()
    result = model.predict(IMAGE_PATH)
    inference_time = time.time() - t0
    print(f"Inference time: {inference_time:.2f}s")

    # Filter detections
    detections = []
    kept_indices = []
    if result and len(result) > 0:
        for i in range(len(result)):
            conf = result.confidence[i] if result.confidence is not None else 1.0
            cls_id = int(result.class_id[i])
            cls_name = classes[cls_id] if cls_id < len(classes) else f"class_{cls_id}"

            # Area filter
            if result.mask is not None:
                area = result.mask[i].sum()
            else:
                x1, y1, x2, y2 = result.xyxy[i]
                area = (x2 - x1) * (y2 - y1)

            if conf >= MIN_CONFIDENCE and area >= MIN_AREA:
                det = {
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": float(conf),
                    "bbox": result.xyxy[i].tolist(),
                    "area": int(area),
                }
                detections.append(det)
                kept_indices.append(i)

    print(f"Total raw detections: {len(result) if result else 0}")
    print(f"After filtering (conf>={MIN_CONFIDENCE}, area>={MIN_AREA}): {len(detections)}")

    # Draw on image
    img_pil = Image.fromarray(img_rgb).convert("RGBA")
    overlay = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for j, idx in enumerate(kept_indices):
        det = detections[j]
        color = COLORS.get(det["class_name"], (255, 255, 255))
        rgba = color + (80,)

        # Draw mask
        if result.mask is not None:
            mask = result.mask[idx].astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if len(contour) >= 3:
                    poly = [(int(p[0][0]), int(p[0][1])) for p in contour]
                    draw.polygon(poly, fill=rgba, outline=color + (200,), width=2)

        # Draw bbox
        bbox = det["bbox"]
        draw.rectangle(bbox, outline=color + (220,), width=2)

    img_composited = Image.alpha_composite(img_pil, overlay).convert("RGB")

    # Draw labels
    label_draw = ImageDraw.Draw(img_composited)
    for det in detections:
        color = COLORS.get(det["class_name"], (255, 255, 255))
        bbox = det["bbox"]
        label = f'{det["class_name"]} {det["confidence"]:.0%}'
        x, y = bbox[0], max(bbox[1] - 16, 0)
        tb = label_draw.textbbox((x, y), label)
        label_draw.rectangle([tb[0] - 2, tb[1] - 1, tb[2] + 2, tb[3] + 1], fill=color)
        label_draw.text((x, y), label, fill=(255, 255, 255))

    # Per-class summary
    from collections import Counter
    class_counts = Counter(d["class_name"] for d in detections)

    # Detection list for HTML
    det_items = ""
    for d in sorted(detections, key=lambda x: -x["confidence"]):
        c = COLORS.get(d["class_name"], (255, 255, 255))
        det_items += (
            f'<div class="det-item">'
            f'<span><span style="display:inline-block;width:10px;height:10px;'
            f'border-radius:50%;background:rgb({c[0]},{c[1]},{c[2]});margin-right:8px;">'
            f'</span>{d["class_name"]}</span>'
            f'<span>{d["confidence"]:.0%} — {d["area"]:,}px</span></div>'
        )

    # Class summary rows
    class_rows = ""
    for cls_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        c = COLORS.get(cls_name, (255, 255, 255))
        avg_conf = np.mean([d["confidence"] for d in detections if d["class_name"] == cls_name])
        class_rows += (
            f'<tr><td><span style="display:inline-block;width:10px;height:10px;'
            f'border-radius:50%;background:rgb({c[0]},{c[1]},{c[2]});margin-right:8px;">'
            f'</span>{cls_name}</td><td>{count}</td><td>{avg_conf:.0%}</td></tr>'
        )

    # Legend
    legend_html = ""
    for name in CLASS_NAMES:
        c = COLORS[name]
        legend_html += (
            f'<span style="display:inline-block;margin:2px 4px;padding:2px 8px;'
            f'border-radius:4px;font-size:0.75em;background:rgb({c[0]},{c[1]},{c[2]});'
            f'color:#fff;font-weight:600;">{name}</span>'
        )

    orig_b64 = img_to_b64(Image.fromarray(img_rgb))
    annotated_b64 = img_to_b64(img_composited)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SAM3 Segmentation — {os.path.basename(IMAGE_PATH)}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0a0a0a; color: #e0e0e0; padding: 20px; }}
  h1 {{ text-align: center; margin-bottom: 6px; font-size: 1.8em; color: #fff; }}
  .subtitle {{ text-align: center; color: #888; margin-bottom: 20px; }}
  .legend {{ text-align: center; margin-bottom: 20px; padding: 10px; }}
  .summary {{ max-width: 700px; margin: 0 auto 30px; background: #1a1a1a; border-radius: 12px; border: 1px solid #333; overflow: hidden; }}
  .summary table {{ width: 100%; border-collapse: collapse; }}
  .summary th {{ background: #222; padding: 10px 16px; text-align: left; font-size: 0.85em; color: #888; text-transform: uppercase; }}
  .summary td {{ padding: 10px 16px; border-top: 1px solid #222; font-size: 0.95em; }}
  .highlight {{ color: #4ade80; font-weight: 600; }}
  .dual-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; max-width: 1400px; margin: 0 auto 30px; }}
  .card {{ background: #1a1a1a; border-radius: 10px; overflow: hidden; border: 1px solid #333; }}
  .card-header {{ padding: 10px 14px; border-bottom: 1px solid #333; display: flex; justify-content: space-between; align-items: center; }}
  .card-header h3 {{ font-size: 0.9em; }}
  .badge {{ padding: 3px 10px; border-radius: 20px; font-size: 0.7em; font-weight: 600; }}
  .badge-purple {{ background: #2d1a4e; color: #c084fc; }}
  .card img {{ width: 100%; display: block; }}
  .det-list {{ padding: 10px 14px; max-height: 400px; overflow-y: auto; }}
  .det-item {{ display: flex; justify-content: space-between; padding: 4px 8px; font-size: 0.82em; border-radius: 3px; margin-bottom: 2px; }}
  .det-item:nth-child(odd) {{ background: #222; }}
  .two-tables {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; max-width: 1400px; margin: 0 auto 30px; }}
</style>
</head>
<body>

<h1>SAM3 (Teacher) — Single Image Segmentation</h1>
<p class="subtitle">{os.path.basename(IMAGE_PATH)} — {len(ONTOLOGY)} class prompts</p>

<div class="summary">
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Model</td><td>SegmentAnything3 (Roboflow API)</td></tr>
<tr><td>Inference Time</td><td class="highlight">{inference_time:.2f}s</td></tr>
<tr><td>Image Size</td><td>{w} x {h}</td></tr>
<tr><td>Raw Detections</td><td>{len(result) if result else 0}</td></tr>
<tr><td>After Filtering</td><td class="highlight">{len(detections)}</td></tr>
<tr><td>Confidence Threshold</td><td>{MIN_CONFIDENCE}</td></tr>
<tr><td>Min Area</td><td>{MIN_AREA}px</td></tr>
<tr><td>Classes Detected</td><td>{len(class_counts)} / {len(CLASS_NAMES)}</td></tr>
</table>
</div>

<div class="legend">{legend_html}</div>

<div class="dual-grid">
  <div class="card">
    <div class="card-header"><h3>Original</h3></div>
    <img src="data:image/jpeg;base64,{orig_b64}">
  </div>
  <div class="card">
    <div class="card-header">
      <h3>SAM3 Segmentation</h3>
      <span class="badge badge-purple">{len(detections)} detections</span>
    </div>
    <img src="data:image/jpeg;base64,{annotated_b64}">
  </div>
</div>

<div class="two-tables">
  <div class="summary">
    <table>
    <tr><th>Class</th><th>Count</th><th>Avg Conf</th></tr>
    {class_rows}
    </table>
  </div>
  <div class="summary">
    <table>
    <tr><th colspan="2">All Detections (sorted by confidence)</th></tr>
    </table>
    <div class="det-list">{det_items}</div>
  </div>
</div>

</body>
</html>"""

    with open(OUTPUT_HTML, "w") as f:
        f.write(html)
    print(f"\nHTML output: {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
