"""
Download 1000 room scene images from Firestore GCS URLs.
Max parallelism for speed.
"""
import json
import os
import random
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

INPUT = "/teamspace/studios/this_studio/firestore_gen_room_images.json"
OUTPUT_DIR = "/teamspace/studios/this_studio/room_images"
TARGET = 1000
SEED = 42
WORKERS = 40  # aggressive parallelism for I/O

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(INPUT) as f:
    data = json.load(f)

room_urls = data["room_images"]
print(f"Available room images: {len(room_urls)}")

random.seed(SEED)
random.shuffle(room_urls)
selected = room_urls[:TARGET]
print(f"Selected: {len(selected)}")


def download(item):
    url = item["url"]
    doc_id = item["doc_id"]
    # Determine extension
    ext = ".png"
    for e in [".jpg", ".jpeg", ".png", ".webp"]:
        if e in url.lower():
            ext = e
            break
    fname = f"{doc_id}_room{ext}"
    fpath = os.path.join(OUTPUT_DIR, fname)
    if os.path.exists(fpath):
        return fname, True, "exists"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(fpath, "wb") as f:
            f.write(r.content)
        return fname, True, "ok"
    except Exception as e:
        return fname, False, str(e)[:80]


t0 = time.time()
print(f"Downloading with {WORKERS} threads...")
ok = 0
fail = 0
with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    futures = {pool.submit(download, item): item for item in selected}
    for i, f in enumerate(as_completed(futures)):
        name, success, msg = f.result()
        if success:
            ok += 1
        else:
            fail += 1
        if (i + 1) % 200 == 0 or i == len(selected) - 1:
            print(f"  {i+1}/{len(selected)} done (ok={ok} fail={fail}) {time.time()-t0:.1f}s")

print(f"\nDone in {time.time()-t0:.1f}s | {ok} downloaded, {fail} failed")
print(f"Files: {len(os.listdir(OUTPUT_DIR))}")
