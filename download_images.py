"""
Download ~1000 product images from Supabase rendered_image URLs.
Proportional mix across all productTypes.
"""
import json
import os
import random
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

INPUT = "/teamspace/studios/this_studio/products_for_labeling.json"
OUTPUT_DIR = "/teamspace/studios/this_studio/product_images"
TOTAL_TARGET = 1000
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load products
with open(INPUT) as f:
    products = json.load(f)

# Filter to those with rendered_image URLs
with_images = [p for p in products if p.get("rendered_image")]
print(f"Products with images: {len(with_images)}")

# Group by productType
by_type = {}
for p in with_images:
    t = p["productType"]
    by_type.setdefault(t, []).append(p)

# Proportional sampling
total_with_images = len(with_images)
random.seed(SEED)
selected = []
for ptype, items in by_type.items():
    proportion = len(items) / total_with_images
    n = max(1, int(TOTAL_TARGET * proportion))
    sampled = random.sample(items, min(n, len(items)))
    selected.extend(sampled)
    print(f"  {ptype:20s}: {len(items):>6} total, sampling {len(sampled)}")

# Trim to exactly TOTAL_TARGET
random.shuffle(selected)
selected = selected[:TOTAL_TARGET]
print(f"\nTotal selected: {len(selected)}")


def download_image(product):
    """Download a single image. Returns (product_id, success)."""
    url = product["rendered_image"]
    pid = product["id"]
    ext = os.path.splitext(url.split("?")[0])[-1] or ".jpg"
    fname = f"{pid}{ext}"
    fpath = os.path.join(OUTPUT_DIR, fname)

    if os.path.exists(fpath):
        return pid, True, "exists"

    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(fpath, "wb") as f:
            f.write(r.content)
        return pid, True, "downloaded"
    except Exception as e:
        return pid, False, str(e)


# Parallel download (20 threads for I/O-bound work)
print(f"\nDownloading {len(selected)} images with 20 threads...")
success = 0
failed = 0
with ThreadPoolExecutor(max_workers=20) as pool:
    futures = {pool.submit(download_image, p): p for p in selected}
    for i, future in enumerate(as_completed(futures)):
        pid, ok, msg = future.result()
        if ok:
            success += 1
        else:
            failed += 1
        if (i + 1) % 100 == 0 or i == len(selected) - 1:
            print(f"  Progress: {i+1}/{len(selected)} (success={success}, failed={failed})")

print(f"\nDone! {success} downloaded, {failed} failed")
print(f"Images in: {OUTPUT_DIR}/")
print(f"Total files: {len(os.listdir(OUTPUT_DIR))}")
