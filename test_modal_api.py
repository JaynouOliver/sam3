"""
Test Modal-deployed YOLOv8n-seg API on room images and generate
an HTML visualization with segments overlaid on original images.
"""

import base64
import io
import json
import random
import time
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

API_URL = "https://mattoboard--mattoboard-segmentation-segmenter-predict.modal.run"
HEALTH_URL = "https://mattoboard--mattoboard-segmentation-segmenter-health.modal.run"
IMAGE_DIR = Path("room_images")
NUM_IMAGES = 10
CONF_THRESHOLD = 0.3
OUTPUT_HTML = "modal_api_results.html"

# 19-class color palette (same as compare_outputs.html)
CLASS_NAMES = [
    "ceilings", "curtains", "decor", "floors", "upholstery", "walls",
    "worktop_surface", "board_accessory", "faucet_tap", "fixtures",
    "handle", "knob", "other_hardware", "outdoor_fabric", "outdoor_paver",
    "stair_rod", "switch", "wallpaper_wallcovering", "na"
]

def get_color(class_id, num_classes=19):
    hue = class_id / num_classes
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return (int(r * 255), int(g * 255), int(b * 255))

COLORS = {name: get_color(i) for i, name in enumerate(CLASS_NAMES)}


def draw_detections(img_bytes, detections):
    """Draw segmentation masks and labels on the image."""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for det in detections:
        color = COLORS.get(det["class_name"], (255, 255, 255))
        rgba = color + (80,)  # semi-transparent fill

        # Draw polygon mask
        if "polygon" in det and len(det["polygon"]) >= 3:
            poly = [tuple(p) for p in det["polygon"]]
            draw.polygon(poly, fill=rgba, outline=color + (200,), width=2)

        # Draw bbox
        bbox = det["bbox"]
        draw.rectangle(bbox, outline=color + (220,), width=2)

    img = Image.alpha_composite(img, overlay).convert("RGB")

    # Draw labels on top (separate pass so they're not behind masks)
    label_draw = ImageDraw.Draw(img)
    for det in detections:
        color = COLORS.get(det["class_name"], (255, 255, 255))
        bbox = det["bbox"]
        label = f'{det["class_name"]} {det["confidence"]:.0%}'
        x, y = bbox[0], max(bbox[1] - 16, 0)
        # Background rectangle for text
        text_bbox = label_draw.textbbox((x, y), label)
        label_draw.rectangle(
            [text_bbox[0] - 2, text_bbox[1] - 1, text_bbox[2] + 2, text_bbox[3] + 1],
            fill=color
        )
        # White text
        label_draw.text((x, y), label, fill=(255, 255, 255))

    return img


def img_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    # Health check
    print("Checking health endpoint...")
    try:
        r = requests.get(HEALTH_URL, timeout=30)
        print(f"  Health: {r.json()}")
    except Exception as e:
        print(f"  Health check failed: {e}")

    # Select random images
    all_images = sorted(IMAGE_DIR.glob("*.*"))
    selected = random.sample(all_images, min(NUM_IMAGES, len(all_images)))
    print(f"\nTesting {len(selected)} images against Modal API...\n")

    results = []
    total_api_time = 0

    for i, img_path in enumerate(selected):
        img_bytes = img_path.read_bytes()
        img_b64 = base64.b64encode(img_bytes).decode()

        print(f"  [{i+1}/{len(selected)}] {img_path.name}...", end=" ", flush=True)

        t0 = time.time()
        try:
            resp = requests.post(
                API_URL,
                json={"image_base64": img_b64, "conf_threshold": CONF_THRESHOLD},
                timeout=60,
            )
            api_time = time.time() - t0
            total_api_time += api_time

            if resp.status_code == 200:
                data = resp.json()
                n_det = data["num_detections"]
                print(f"{n_det} detections, {api_time:.2f}s")

                # Draw annotated image
                annotated_img = draw_detections(img_bytes, data["detections"])

                results.append({
                    "filename": img_path.name,
                    "api_time": api_time,
                    "num_detections": n_det,
                    "detections": data["detections"],
                    "image_size": data["image_size"],
                    "original_b64": base64.b64encode(
                        img_to_b64_jpeg(img_bytes)
                    ).decode() if False else img_to_b64(Image.open(io.BytesIO(img_bytes))),
                    "annotated_b64": img_to_b64(annotated_img),
                })
            else:
                print(f"ERROR {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            api_time = time.time() - t0
            print(f"FAILED ({api_time:.1f}s): {e}")

    if not results:
        print("No successful results. Exiting.")
        return

    # Stats
    avg_time = total_api_time / len(results)
    avg_dets = sum(r["num_detections"] for r in results) / len(results)
    print(f"\n{'='*50}")
    print(f"Results: {len(results)}/{len(selected)} successful")
    print(f"Avg API response time: {avg_time:.2f}s (includes network)")
    print(f"Avg detections: {avg_dets:.1f}")

    # Generate HTML
    generate_html(results, avg_time, avg_dets)
    print(f"\nHTML output: {OUTPUT_HTML}")


def generate_html(results, avg_time, avg_dets):
    # Build legend
    legend_html = ""
    for name in CLASS_NAMES:
        c = COLORS[name]
        legend_html += (
            f'<span style="display:inline-block;margin:2px 4px;padding:2px 8px;'
            f'border-radius:4px;font-size:0.75em;background:rgb({c[0]},{c[1]},{c[2]});'
            f'color:#fff;font-weight:600;">{name}</span>'
        )

    # Build image sections
    sections_html = ""
    for r in results:
        # Detection list
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

        sections_html += f"""
    <div class="image-section">
      <h2 class="image-title">{r['filename']} — {r['api_time']:.2f}s — {r['num_detections']} detections</h2>
      <div class="dual-grid">
        <div class="card">
          <div class="card-header"><h3>Original</h3></div>
          <img src="data:image/jpeg;base64,{r['original_b64']}">
        </div>
        <div class="card">
          <div class="card-header">
            <h3>Modal API — Segmentation</h3>
            <span class="badge badge-green">{r['num_detections']} det</span>
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
<title>Modal API — YOLOv8n-seg Segmentation Results</title>
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

<h1>Modal API — YOLOv8n-seg Segmentation</h1>
<p class="subtitle">Deployed on Modal.com (T4 GPU) — testing {len(results)} room images</p>

<div class="summary">
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Endpoint</td><td style="font-family:monospace;font-size:0.85em;">mattoboard--mattoboard-segmentation-...</td></tr>
<tr><td>GPU</td><td>NVIDIA T4</td></tr>
<tr><td>Model</td><td>YOLOv8n-seg (6.5 MB, 19 classes)</td></tr>
<tr><td>Avg API Response</td><td class="highlight">{avg_time:.2f}s (includes network latency)</td></tr>
<tr><td>Avg Detections</td><td>{avg_dets:.1f}</td></tr>
<tr><td>Confidence Threshold</td><td>{CONF_THRESHOLD}</td></tr>
<tr><td>Images Tested</td><td>{len(results)}</td></tr>
</table>
</div>

<div class="legend">{legend_html}</div>

{sections_html}

</body>
</html>"""

    Path(OUTPUT_HTML).write_text(html)


if __name__ == "__main__":
    main()
