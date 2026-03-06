"""
Run YOLOv8l-seg inference on room images and generate HTML visualization.
"""

import base64
import colorsys
import io
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

# ── Config ──────────────────────────────────────────────────────────
MODEL_PATH = "runs/segment/pipeline_training/sam3_distilled_l/weights/best.pt"
IMAGE_DIR = Path("room_images")
NUM_IMAGES = 15
CONF_THRESHOLD = 0.3
OUTPUT_HTML = "inference_results.html"
# ────────────────────────────────────────────────────────────────────

CLASS_NAMES = [
    "ceilings", "curtains", "decor", "floors", "upholstery", "walls",
    "worktop_surface", "board_accessory", "faucet_tap", "fixtures",
    "handle", "knob", "other_hardware", "outdoor_fabric", "outdoor_paver",
    "stair_rod", "switch", "wallpaper_wallcovering", "na"
]


def get_color(class_id, num_classes=19):
    hue = class_id / num_classes
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return (int(r * 255), int(g * 255), int(b * 255))


COLORS = {name: get_color(i) for i, name in enumerate(CLASS_NAMES)}


def draw_detections(img_pil, result):
    """Draw segmentation masks and labels on the image."""
    img = img_pil.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    names = result.names
    detections = []

    if result.boxes is not None:
        for i, box in enumerate(result.boxes):
            cls_id = int(box.cls[0])
            cls_name = names[cls_id]
            conf = float(box.conf[0])
            color = COLORS.get(cls_name, (255, 255, 255))
            rgba = color + (80,)

            # Draw polygon mask
            if result.masks is not None and i < len(result.masks):
                mask_xy = result.masks[i].xy
                if len(mask_xy) > 0 and len(mask_xy[0]) >= 3:
                    poly = [tuple(p) for p in mask_xy[0]]
                    draw.polygon(poly, fill=rgba, outline=color + (200,), width=2)

            # Draw bbox
            bbox = box.xyxy[0].tolist()
            draw.rectangle(bbox, outline=color + (220,), width=2)

            detections.append({
                "class_name": cls_name,
                "confidence": conf,
                "bbox": bbox,
            })

    img = Image.alpha_composite(img, overlay).convert("RGB")

    # Draw labels
    label_draw = ImageDraw.Draw(img)
    if result.boxes is not None:
        for i, box in enumerate(result.boxes):
            cls_name = names[int(box.cls[0])]
            conf = float(box.conf[0])
            color = COLORS.get(cls_name, (255, 255, 255))
            bbox = box.xyxy[0].tolist()
            label = f'{cls_name} {conf:.0%}'
            x, y = bbox[0], max(bbox[1] - 16, 0)
            tb = label_draw.textbbox((x, y), label)
            label_draw.rectangle([tb[0] - 2, tb[1] - 1, tb[2] + 2, tb[3] + 1], fill=color)
            label_draw.text((x, y), label, fill=(255, 255, 255))

    return img, detections


def img_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # Warmup
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(dummy, verbose=False)

    all_images = sorted(IMAGE_DIR.glob("*.*"))
    selected = random.sample(all_images, min(NUM_IMAGES, len(all_images)))
    print(f"Running inference on {len(selected)} images (conf={CONF_THRESHOLD})...\n")

    results_data = []
    times = []

    for i, img_path in enumerate(selected):
        img_pil = Image.open(img_path).convert("RGB")

        t0 = time.time()
        result = model.predict(np.array(img_pil), conf=CONF_THRESHOLD, verbose=False)[0]
        t_inf = time.time() - t0
        times.append(t_inf)

        n_det = len(result.boxes) if result.boxes is not None else 0
        print(f"  [{i+1}/{len(selected)}] {img_path.name}: {n_det} det, {t_inf*1000:.1f}ms")

        annotated, detections = draw_detections(img_pil, result)

        results_data.append({
            "filename": img_path.name,
            "inference_ms": t_inf * 1000,
            "num_detections": n_det,
            "detections": detections,
            "original_b64": img_to_b64(img_pil),
            "annotated_b64": img_to_b64(annotated),
        })

    avg_ms = np.mean(times) * 1000
    avg_dets = np.mean([r["num_detections"] for r in results_data])
    print(f"\n{'='*50}")
    print(f"Avg inference: {avg_ms:.1f}ms ({1000/avg_ms:.0f} FPS)")
    print(f"Avg detections: {avg_dets:.1f}")

    generate_html(results_data, avg_ms, avg_dets)
    print(f"HTML output: {OUTPUT_HTML}")


def generate_html(results, avg_ms, avg_dets):
    legend_html = ""
    for name in CLASS_NAMES:
        c = COLORS[name]
        legend_html += (
            f'<span style="display:inline-block;margin:2px 4px;padding:2px 8px;'
            f'border-radius:4px;font-size:0.75em;background:rgb({c[0]},{c[1]},{c[2]});'
            f'color:#fff;font-weight:600;">{name}</span>'
        )

    sections = ""
    for r in results:
        det_items = ""
        for d in r["detections"]:
            c = COLORS.get(d["class_name"], (255, 255, 255))
            det_items += (
                f'<div class="det-item">'
                f'<span><span style="display:inline-block;width:8px;height:8px;'
                f'border-radius:50%;background:rgb({c[0]},{c[1]},{c[2]});margin-right:6px;">'
                f'</span>{d["class_name"]}</span>'
                f'<span>{d["confidence"]:.0%}</span></div>'
            )

        sections += f"""
    <div class="image-section">
      <h2 class="image-title">{r['filename']} — {r['inference_ms']:.1f}ms — {r['num_detections']} detections</h2>
      <div class="dual-grid">
        <div class="card">
          <div class="card-header"><h3>Original</h3></div>
          <img src="data:image/jpeg;base64,{r['original_b64']}">
        </div>
        <div class="card">
          <div class="card-header">
            <h3>YOLOv8l-seg Segmentation</h3>
            <span class="badge badge-green">{r['num_detections']} det / {r['inference_ms']:.0f}ms</span>
          </div>
          <img src="data:image/jpeg;base64,{r['annotated_b64']}">
          <div class="det-list">{det_items}</div>
        </div>
      </div>
    </div>
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>YOLOv8l-seg — Inference Results</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0a0a0a; color: #e0e0e0; padding: 20px; }}
  h1 {{ text-align: center; margin-bottom: 6px; font-size: 1.8em; color: #fff; }}
  .subtitle {{ text-align: center; color: #888; margin-bottom: 20px; }}
  .legend {{ text-align: center; margin-bottom: 20px; padding: 10px; }}
  .summary {{ max-width: 900px; margin: 0 auto 30px; background: #1a1a1a; border-radius: 12px; border: 1px solid #333; overflow: hidden; }}
  .summary table {{ width: 100%; border-collapse: collapse; }}
  .summary th {{ background: #222; padding: 10px 16px; text-align: left; font-size: 0.85em; color: #888; text-transform: uppercase; }}
  .summary td {{ padding: 10px 16px; border-top: 1px solid #222; font-size: 0.95em; }}
  .highlight {{ color: #4ade80; font-weight: 600; }}
  .image-section {{ margin-bottom: 40px; }}
  .image-title {{ font-size: 1em; color: #888; margin-bottom: 12px; padding-left: 4px; font-family: monospace; }}
  .dual-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
  .card {{ background: #1a1a1a; border-radius: 10px; overflow: hidden; border: 1px solid #333; }}
  .card-header {{ padding: 10px 14px; border-bottom: 1px solid #333; display: flex; justify-content: space-between; align-items: center; }}
  .card-header h3 {{ font-size: 0.9em; }}
  .badge {{ padding: 3px 10px; border-radius: 20px; font-size: 0.7em; font-weight: 600; }}
  .badge-green {{ background: #1a3f2a; color: #4ade80; }}
  .card img {{ width: 100%; display: block; }}
  .det-list {{ padding: 8px 12px; max-height: 150px; overflow-y: auto; }}
  .det-item {{ display: flex; justify-content: space-between; padding: 2px 6px; font-size: 0.78em; border-radius: 3px; margin-bottom: 1px; }}
  .det-item:nth-child(odd) {{ background: #222; }}
</style>
</head>
<body>

<h1>YOLOv8l-seg — Local Inference Results</h1>
<p class="subtitle">SAM3-distilled large model — {len(results)} room images</p>

<div class="summary">
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Model</td><td>YOLOv8l-seg (92.3 MB, 19 classes)</td></tr>
<tr><td>Training</td><td>67 epochs (early stopped at 92), 65 min on L4</td></tr>
<tr><td>Val mAP50 (box/mask)</td><td class="highlight">0.627 / 0.584</td></tr>
<tr><td>Avg Inference</td><td class="highlight">{avg_ms:.1f}ms ({1000/avg_ms:.0f} FPS)</td></tr>
<tr><td>Avg Detections</td><td>{avg_dets:.1f}</td></tr>
<tr><td>Confidence Threshold</td><td>{CONF_THRESHOLD}</td></tr>
<tr><td>Images</td><td>{len(results)}</td></tr>
</table>
</div>

<div class="legend">{legend_html}</div>

{sections}

</body>
</html>"""

    Path(OUTPUT_HTML).write_text(html)


if __name__ == "__main__":
    main()
