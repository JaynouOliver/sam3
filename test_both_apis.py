"""
Comprehensive comparison: YOLOv8n-seg vs YOLOv8l-seg on Modal.
Tests 10 images through both APIs, compares speed and detections,
generates side-by-side HTML visualization.
"""

import base64
import colorsys
import io
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw

NANO_URL = "https://mattoboard--mattoboard-segmentation-segmenter-predict.modal.run"
LARGE_URL = "https://mattoboard--mattoboard-segmentation-large-segmenter-predict.modal.run"
NANO_HEALTH = "https://mattoboard--mattoboard-segmentation-segmenter-health.modal.run"
LARGE_HEALTH = "https://mattoboard--mattoboard-segmentation-large-segmenter-health.modal.run"

IMAGE_DIR = Path("room_images")
NUM_IMAGES = 10
CONF_THRESHOLD = 0.3
OUTPUT_HTML = "compare_nano_vs_large.html"

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


def draw_detections(img_bytes, detections):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for det in detections:
        color = COLORS.get(det["class_name"], (255, 255, 255))
        rgba = color + (80,)
        if "polygon" in det and len(det["polygon"]) >= 3:
            poly = [tuple(p) for p in det["polygon"]]
            draw.polygon(poly, fill=rgba, outline=color + (200,), width=2)
        bbox = det["bbox"]
        draw.rectangle(bbox, outline=color + (220,), width=2)

    img = Image.alpha_composite(img, overlay).convert("RGB")
    label_draw = ImageDraw.Draw(img)
    for det in detections:
        color = COLORS.get(det["class_name"], (255, 255, 255))
        bbox = det["bbox"]
        label = f'{det["class_name"]} {det["confidence"]:.0%}'
        x, y = bbox[0], max(bbox[1] - 16, 0)
        tb = label_draw.textbbox((x, y), label)
        label_draw.rectangle([tb[0] - 2, tb[1] - 1, tb[2] + 2, tb[3] + 1], fill=color)
        label_draw.text((x, y), label, fill=(255, 255, 255))
    return img


def img_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def call_api(url, img_b64, conf=0.3):
    t0 = time.time()
    resp = requests.post(url, json={"image_base64": img_b64, "conf_threshold": conf}, timeout=120)
    elapsed = time.time() - t0
    if resp.status_code == 200:
        return resp.json(), elapsed
    else:
        print(f"  ERROR {resp.status_code}: {resp.text[:200]}")
        return None, elapsed


def main():
    # Warm up both endpoints
    print("Warming up endpoints...")
    for name, url in [("Nano", NANO_HEALTH), ("Large", LARGE_HEALTH)]:
        try:
            r = requests.get(url, timeout=120)
            print(f"  {name}: {r.json()}")
        except Exception as e:
            print(f"  {name} health failed: {e}")

    # Select images
    all_images = sorted(IMAGE_DIR.glob("*.*"))
    random.seed(42)  # reproducible
    selected = random.sample(all_images, min(NUM_IMAGES, len(all_images)))
    print(f"\nTesting {len(selected)} images through both APIs (conf={CONF_THRESHOLD})...\n")

    results = []
    nano_times, large_times = [], []
    nano_dets_total, large_dets_total = [], []
    nano_classes_all, large_classes_all = defaultdict(int), defaultdict(int)

    for i, img_path in enumerate(selected):
        img_bytes = img_path.read_bytes()
        img_b64 = base64.b64encode(img_bytes).decode()
        print(f"[{i+1}/{len(selected)}] {img_path.name}")

        # Call nano
        nano_data, nano_time = call_api(NANO_URL, img_b64, CONF_THRESHOLD)
        # Call large
        large_data, large_time = call_api(LARGE_URL, img_b64, CONF_THRESHOLD)

        if nano_data is None or large_data is None:
            print("  Skipping (API error)")
            continue

        nano_n = nano_data["num_detections"]
        large_n = large_data["num_detections"]
        nano_times.append(nano_time)
        large_times.append(large_time)
        nano_dets_total.append(nano_n)
        large_dets_total.append(large_n)

        for d in nano_data["detections"]:
            nano_classes_all[d["class_name"]] += 1
        for d in large_data["detections"]:
            large_classes_all[d["class_name"]] += 1

        print(f"  Nano:  {nano_n:2d} det, {nano_time:.2f}s | Large: {large_n:2d} det, {large_time:.2f}s")

        # Draw annotated images
        nano_img = draw_detections(img_bytes, nano_data["detections"])
        large_img = draw_detections(img_bytes, large_data["detections"])
        orig_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        results.append({
            "filename": img_path.name,
            "nano_time": nano_time,
            "large_time": large_time,
            "nano_dets": nano_n,
            "large_dets": large_n,
            "nano_detections": nano_data["detections"],
            "large_detections": large_data["detections"],
            "original_b64": img_to_b64(orig_img),
            "nano_b64": img_to_b64(nano_img),
            "large_b64": img_to_b64(large_img),
        })

    if not results:
        print("No results. Exiting.")
        return

    # Summary stats
    avg_nano_t = np.mean(nano_times)
    avg_large_t = np.mean(large_times)
    avg_nano_d = np.mean(nano_dets_total)
    avg_large_d = np.mean(large_dets_total)

    print(f"\n{'='*60}")
    print(f"{'SUMMARY':^60}")
    print(f"{'='*60}")
    print(f"{'':20} {'YOLOv8n (nano)':>18} {'YOLOv8l (large)':>18}")
    print(f"{'-'*60}")
    print(f"{'Avg API time':20} {avg_nano_t:>17.2f}s {avg_large_t:>17.2f}s")
    print(f"{'Avg detections':20} {avg_nano_d:>18.1f} {avg_large_d:>18.1f}")
    print(f"{'Total detections':20} {sum(nano_dets_total):>18d} {sum(large_dets_total):>18d}")
    print(f"{'Model size':20} {'6.5 MB':>18} {'92.3 MB':>18}")
    print(f"{'Val mAP50 (mask)':20} {'0.434':>18} {'0.584':>18}")
    print(f"\nPer-class detections:")
    all_cls = sorted(set(list(nano_classes_all.keys()) + list(large_classes_all.keys())))
    for cls in all_cls:
        n = nano_classes_all.get(cls, 0)
        l = large_classes_all.get(cls, 0)
        diff = l - n
        marker = f" (+{diff})" if diff > 0 else f" ({diff})" if diff < 0 else ""
        print(f"  {cls:25} {n:>5} {l:>5}{marker}")

    generate_html(results, avg_nano_t, avg_large_t, avg_nano_d, avg_large_d,
                  nano_classes_all, large_classes_all)
    print(f"\nHTML output: {OUTPUT_HTML}")


def generate_html(results, avg_nano_t, avg_large_t, avg_nano_d, avg_large_d,
                  nano_classes, large_classes):
    # Legend
    legend_html = ""
    for name in CLASS_NAMES:
        c = COLORS[name]
        legend_html += (
            f'<span style="display:inline-block;margin:2px 4px;padding:2px 8px;'
            f'border-radius:4px;font-size:0.75em;background:rgb({c[0]},{c[1]},{c[2]});'
            f'color:#fff;font-weight:600;">{name}</span>'
        )

    # Per-class table rows
    all_cls = sorted(set(list(nano_classes.keys()) + list(large_classes.keys())))
    class_rows = ""
    for cls in all_cls:
        n = nano_classes.get(cls, 0)
        l = large_classes.get(cls, 0)
        winner_n = ' class="winner"' if n > l else ''
        winner_l = ' class="winner"' if l > n else ''
        class_rows += f"<tr><td>{cls}</td><td{winner_n}>{n}</td><td{winner_l}>{l}</td></tr>\n"

    # Image sections
    sections = ""
    for r in results:
        # Detection lists
        def det_list_html(detections):
            html = ""
            for d in detections:
                c = COLORS.get(d["class_name"], (255, 255, 255))
                html += (
                    f'<div class="det-item">'
                    f'<span><span style="display:inline-block;width:8px;height:8px;'
                    f'border-radius:50%;background:rgb({c[0]},{c[1]},{c[2]});margin-right:6px;">'
                    f'</span>{d["class_name"]}</span>'
                    f'<span>{d["confidence"]:.0%}</span></div>'
                )
            return html

        sections += f"""
    <div class="image-section">
      <h2 class="image-title">{r['filename']}</h2>
      <div class="triple-grid">
        <div class="card">
          <div class="card-header"><h3>Original</h3></div>
          <img src="data:image/jpeg;base64,{r['original_b64']}">
        </div>
        <div class="card">
          <div class="card-header">
            <h3>YOLOv8n (nano)</h3>
            <span class="badge badge-blue">{r['nano_dets']} det / {r['nano_time']:.2f}s</span>
          </div>
          <img src="data:image/jpeg;base64,{r['nano_b64']}">
          <div class="det-list">{det_list_html(r['nano_detections'])}</div>
        </div>
        <div class="card">
          <div class="card-header">
            <h3>YOLOv8l (large)</h3>
            <span class="badge badge-green">{r['large_dets']} det / {r['large_time']:.2f}s</span>
          </div>
          <img src="data:image/jpeg;base64,{r['large_b64']}">
          <div class="det-list">{det_list_html(r['large_detections'])}</div>
        </div>
      </div>
    </div>
"""

    speedup = avg_nano_t / avg_large_t if avg_large_t > 0 else 0

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>YOLOv8n vs YOLOv8l — Modal API Comparison</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0a0a0a; color: #e0e0e0; padding: 20px; }}
  h1 {{ text-align: center; margin-bottom: 6px; font-size: 1.8em; color: #fff; }}
  .subtitle {{ text-align: center; color: #888; margin-bottom: 20px; }}
  .legend {{ text-align: center; margin-bottom: 20px; padding: 10px; }}
  .summary {{ max-width: 1000px; margin: 0 auto 30px; background: #1a1a1a; border-radius: 12px; border: 1px solid #333; overflow: hidden; }}
  .summary table {{ width: 100%; border-collapse: collapse; }}
  .summary th {{ background: #222; padding: 10px 16px; text-align: left; font-size: 0.85em; color: #888; text-transform: uppercase; }}
  .summary td {{ padding: 10px 16px; border-top: 1px solid #222; font-size: 0.95em; }}
  .winner {{ color: #4ade80; font-weight: 600; }}
  .image-section {{ margin-bottom: 40px; }}
  .image-title {{ font-size: 1em; color: #888; margin-bottom: 12px; padding-left: 4px; font-family: monospace; }}
  .triple-grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }}
  .card {{ background: #1a1a1a; border-radius: 10px; overflow: hidden; border: 1px solid #333; }}
  .card-header {{ padding: 10px 14px; border-bottom: 1px solid #333; display: flex; justify-content: space-between; align-items: center; }}
  .card-header h3 {{ font-size: 0.9em; }}
  .badge {{ padding: 3px 10px; border-radius: 20px; font-size: 0.7em; font-weight: 600; }}
  .badge-blue {{ background: #1e3a5f; color: #60a5fa; }}
  .badge-green {{ background: #1a3f2a; color: #4ade80; }}
  .card img {{ width: 100%; display: block; }}
  .det-list {{ padding: 8px 12px; max-height: 150px; overflow-y: auto; }}
  .det-item {{ display: flex; justify-content: space-between; padding: 2px 6px; font-size: 0.78em; border-radius: 3px; margin-bottom: 1px; }}
  .det-item:nth-child(odd) {{ background: #222; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; max-width: 1000px; margin: 0 auto 30px; }}
</style>
</head>
<body>

<h1>YOLOv8n (nano) vs YOLOv8l (large)</h1>
<p class="subtitle">Modal API head-to-head — {len(results)} room images, conf={CONF_THRESHOLD}</p>

<div class="summary">
<table>
<tr><th>Metric</th><th>YOLOv8n-seg (nano)</th><th>YOLOv8l-seg (large)</th></tr>
<tr><td>Model Size</td><td>6.5 MB</td><td>92.3 MB</td></tr>
<tr><td>Parameters</td><td>3.4M</td><td>45.9M</td></tr>
<tr><td>Val mAP50 (box)</td><td>0.469</td><td class="winner">0.627</td></tr>
<tr><td>Val mAP50 (mask)</td><td>0.434</td><td class="winner">0.584</td></tr>
<tr><td>Val mAP50-95 (mask)</td><td>0.294</td><td class="winner">0.404</td></tr>
<tr><td>Avg API Response</td><td>{"%.2f" % avg_nano_t}s</td><td>{"%.2f" % avg_large_t}s</td></tr>
<tr><td>Avg Detections</td><td>{avg_nano_d:.1f}</td><td class="winner">{avg_large_d:.1f}</td></tr>
<tr><td>Total Detections</td><td>{sum(r['nano_dets'] for r in results)}</td><td class="winner">{sum(r['large_dets'] for r in results)}</td></tr>
<tr><td>GPU</td><td>T4</td><td>T4</td></tr>
<tr><td>GPU Inference</td><td class="winner">~2ms</td><td>~20ms</td></tr>
</table>
</div>

<div class="two-col">
<div class="summary">
<table>
<tr><th colspan="3">Per-Class Detection Counts</th></tr>
<tr><th>Class</th><th>Nano</th><th>Large</th></tr>
{class_rows}
</table>
</div>
</div>

<div class="legend">{legend_html}</div>

{sections}

</body>
</html>"""

    Path(OUTPUT_HTML).write_text(html)


if __name__ == "__main__":
    main()
