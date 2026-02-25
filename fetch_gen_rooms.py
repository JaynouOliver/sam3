"""
Extract room scene + closeup images from dsCollectionGenerator finalResults.
These are the GENERATED interior images — exactly what we need for SAM3 training.
"""
import json
import os
import subprocess
import requests

PROJECT_ID = "mattoboard-staging"
COLLECTION = "dsCollectionGenerator"

# Get access token
config_path = os.path.expanduser("~/.config/configstore/firebase-tools.json")
if not os.path.exists(config_path):
    config_path = "/teamspace/studios/this_studio/.config/configstore/firebase-tools.json"

with open(config_path) as f:
    config = json.load(f)

refresh_token = config.get("tokens", {}).get("refresh_token")
client_id = "563584335869-fgrhgmd47bqnekij5i8b5pr03ho849e6.apps.googleusercontent.com"
client_secret = "j9iVZfS8kkCEFUPaAeJV0sAi"

# Refresh token to get access token
resp = requests.post("https://oauth2.googleapis.com/token", data={
    "grant_type": "refresh_token",
    "refresh_token": refresh_token,
    "client_id": client_id,
    "client_secret": client_secret,
})
token = resp.json()["access_token"]
print("Token refreshed")

headers = {"Authorization": f"Bearer {token}"}
BASE_URL = f"https://firestore.googleapis.com/v1/projects/{PROJECT_ID}/databases/(default)/documents"


def parse_value(val):
    if "stringValue" in val:
        return val["stringValue"]
    elif "integerValue" in val:
        return int(val["integerValue"])
    elif "doubleValue" in val:
        return val["doubleValue"]
    elif "booleanValue" in val:
        return val["booleanValue"]
    elif "arrayValue" in val:
        return [parse_value(v) for v in val["arrayValue"].get("values", [])]
    elif "mapValue" in val:
        return {k: parse_value(v) for k, v in val["mapValue"].get("fields", {}).items()}
    elif "nullValue" in val:
        return None
    elif "timestampValue" in val:
        return val["timestampValue"]
    return str(val)


def extract_urls_deep(obj, prefix=""):
    """Recursively extract all URLs from nested dicts/lists."""
    urls = []
    if isinstance(obj, str):
        if 'http' in obj and any(ext in obj.lower() for ext in ['.jpg', '.png', '.jpeg', '.webp']):
            urls.append({"field": prefix, "url": obj})
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            urls.extend(extract_urls_deep(item, f"{prefix}[{i}]"))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            urls.extend(extract_urls_deep(v, f"{prefix}.{k}" if prefix else k))
    return urls


# Paginate through all docs, extract finalResults images
print(f"Fetching {COLLECTION}...")
all_room_images = []
all_closeup_images = []
all_other_gen_images = []
next_token = None
total_docs = 0
page = 0

while True:
    url = f"{BASE_URL}/{COLLECTION}?pageSize=100"
    if next_token:
        url += f"&pageToken={next_token}"

    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        print(f"Error {resp.status_code}: {resp.text[:200]}")
        break

    data = resp.json()
    docs = data.get("documents", [])
    if not docs:
        break

    total_docs += len(docs)
    page += 1

    for doc in docs:
        doc_id = doc["name"].split("/")[-1]
        fields = doc.get("fields", {})

        # Parse finalResults
        if "finalResults" in fields:
            final = parse_value(fields["finalResults"])
            if isinstance(final, dict):
                # Room scene images
                for key in ["roomScene", "roomImage", "room_scene", "generatedRoom"]:
                    if key in final and isinstance(final[key], str) and "http" in final[key]:
                        all_room_images.append({
                            "doc_id": doc_id,
                            "field": f"finalResults.{key}",
                            "url": final[key]
                        })

                # Closeup images
                for key in ["closeupImages", "closeup_images", "closeups"]:
                    if key in final and isinstance(final[key], list):
                        for i, url_val in enumerate(final[key]):
                            if isinstance(url_val, str) and "http" in url_val:
                                all_closeup_images.append({
                                    "doc_id": doc_id,
                                    "field": f"finalResults.{key}[{i}]",
                                    "url": url_val
                                })

                # Any other image URLs in finalResults
                other_urls = extract_urls_deep(final, "finalResults")
                for u in other_urls:
                    if u not in all_room_images and u not in all_closeup_images:
                        already = any(x["url"] == u["url"] for x in all_room_images + all_closeup_images)
                        if not already:
                            all_other_gen_images.append({
                                "doc_id": doc_id,
                                **u
                            })

    next_token = data.get("nextPageToken")
    if not next_token:
        break

    if page % 10 == 0:
        print(f"  Page {page}: {total_docs} docs | rooms={len(all_room_images)} closeups={len(all_closeup_images)} other={len(all_other_gen_images)}")

print(f"\nTotal docs: {total_docs}")
print(f"Room scene images: {len(all_room_images)}")
print(f"Closeup images: {len(all_closeup_images)}")
print(f"Other generated images: {len(all_other_gen_images)}")

# Combine all
all_images = {
    "room_images": all_room_images,
    "closeup_images": all_closeup_images,
    "other_images": all_other_gen_images,
    "total": len(all_room_images) + len(all_closeup_images) + len(all_other_gen_images),
}

output = "/teamspace/studios/this_studio/firestore_gen_room_images.json"
with open(output, "w") as f:
    json.dump(all_images, f, indent=2)

print(f"\nSaved to {output}")

# Show samples
if all_room_images:
    print("\nSample room images:")
    for img in all_room_images[:5]:
        print(f"  {img['url'][:120]}")

if all_closeup_images:
    print("\nSample closeup images:")
    for img in all_closeup_images[:5]:
        print(f"  {img['url'][:120]}")
