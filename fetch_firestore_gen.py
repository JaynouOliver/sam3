"""
Fetch generated images from Firestore dsCollectionGenerator collection.
Uses Firebase CLI access token + Firestore REST API.
Project: mattoboard-staging
READ-ONLY.
"""
import subprocess
import json
import requests

PROJECT_ID = "mattoboard-staging"
COLLECTION = "dsCollectionGenerator"

# Get access token from Firebase CLI
print("Getting access token from Firebase CLI...")
result = subprocess.run(
    ['firebase', 'login:ci', '--no-localhost'],
    capture_output=True, text=True, timeout=10
)

# Alternative: extract token from the firebase config
import os
import json as json_mod

# Try to get token from gcloud or firebase internals
# Firebase stores tokens in ~/.config/configstore/firebase-tools.json
config_paths = [
    os.path.expanduser("~/.config/configstore/firebase-tools.json"),
    os.path.expanduser("~/.config/firebase/firebase-tools.json"),
]

token = None
for path in config_paths:
    if os.path.exists(path):
        with open(path) as f:
            config = json_mod.load(f)
        # Navigate to tokens
        tokens = config.get("tokens", {})
        token = tokens.get("access_token")
        refresh_token = tokens.get("refresh_token")
        if token:
            print(f"Found access token in {path}")
            break

if not token:
    # Try using gcloud
    result = subprocess.run(
        ['gcloud', 'auth', 'print-access-token'],
        capture_output=True, text=True, timeout=10
    )
    if result.returncode == 0:
        token = result.stdout.strip()
        print("Got token from gcloud")

if not token:
    print("ERROR: Could not get access token. Trying firebase use with ADC...")
    # Last resort: use Application Default Credentials
    result = subprocess.run(
        ['gcloud', 'auth', 'application-default', 'print-access-token'],
        capture_output=True, text=True, timeout=10
    )
    if result.returncode == 0:
        token = result.stdout.strip()

if not token:
    print("Could not obtain access token. Please run:")
    print("  gcloud auth application-default login --no-launch-browser")
    exit(1)

# Query Firestore REST API
BASE_URL = f"https://firestore.googleapis.com/v1/projects/{PROJECT_ID}/databases/(default)/documents"

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
}

# First: get a few documents to understand schema
print(f"\nQuerying {COLLECTION} (first 5 docs)...")
url = f"{BASE_URL}/{COLLECTION}?pageSize=5"
resp = requests.get(url, headers=headers)

if resp.status_code != 200:
    print(f"Error {resp.status_code}: {resp.text[:500]}")

    # If token expired, try refreshing
    if resp.status_code == 401 or resp.status_code == 403:
        print("\nToken may be expired. Trying to refresh...")
        # Use refresh token
        for path in config_paths:
            if os.path.exists(path):
                with open(path) as f:
                    config = json_mod.load(f)
                refresh_token = config.get("tokens", {}).get("refresh_token")
                client_id = config.get("tokens", {}).get("client_id", "563584335869-fgrhgmd47bqnekij5i8b5pr03ho849e6.apps.googleusercontent.com")
                client_secret = config.get("tokens", {}).get("client_secret", "j9iVZfS8kkCEFUPaAeJV0sAi")

                if refresh_token:
                    refresh_resp = requests.post("https://oauth2.googleapis.com/token", data={
                        "grant_type": "refresh_token",
                        "refresh_token": refresh_token,
                        "client_id": client_id,
                        "client_secret": client_secret,
                    })
                    if refresh_resp.status_code == 200:
                        token = refresh_resp.json()["access_token"]
                        headers["Authorization"] = f"Bearer {token}"
                        print("Token refreshed!")
                        resp = requests.get(url, headers=headers)
                    else:
                        print(f"Refresh failed: {refresh_resp.text[:200]}")
                break

    if resp.status_code != 200:
        print(f"\nFinal error {resp.status_code}: {resp.text[:500]}")
        exit(1)

data = resp.json()
documents = data.get("documents", [])
print(f"Got {len(documents)} documents\n")

# Parse Firestore document format
def parse_value(val):
    """Parse Firestore REST API value format."""
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

def parse_doc(doc):
    """Parse a Firestore document."""
    fields = doc.get("fields", {})
    parsed = {}
    for k, v in fields.items():
        parsed[k] = parse_value(v)
    doc_id = doc["name"].split("/")[-1]
    return doc_id, parsed

# Show schema
for doc in documents[:3]:
    doc_id, fields = parse_doc(doc)
    print(f"Doc ID: {doc_id}")
    print(f"  Fields: {list(fields.keys())}")
    for k, v in fields.items():
        val_str = str(v)
        if len(val_str) > 200:
            val_str = val_str[:200] + "..."
        print(f"  {k}: {val_str}")
    print()

# Now paginate through all documents to find image URLs
print("Fetching all documents...")
all_image_urls = []
next_page_token = data.get("nextPageToken")
page_count = 1
total_docs = len(documents)

# Process first page
for doc in documents:
    doc_id, fields = parse_doc(doc)
    for key, val in fields.items():
        if isinstance(val, str) and 'http' in val and any(ext in val.lower() for ext in ['.jpg', '.png', '.jpeg', '.webp']):
            all_image_urls.append({"doc_id": doc_id, "field": key, "url": val})
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, str) and 'http' in item:
                    all_image_urls.append({"doc_id": doc_id, "field": key, "url": item})
                elif isinstance(item, dict):
                    for sk, sv in item.items():
                        if isinstance(sv, str) and 'http' in sv:
                            all_image_urls.append({"doc_id": doc_id, "field": f"{key}.{sk}", "url": sv})

# Paginate
while next_page_token:
    page_count += 1
    url = f"{BASE_URL}/{COLLECTION}?pageSize=100&pageToken={next_page_token}"
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        print(f"Page {page_count} error: {resp.status_code}")
        break
    data = resp.json()
    documents = data.get("documents", [])
    total_docs += len(documents)

    for doc in documents:
        doc_id, fields = parse_doc(doc)
        for key, val in fields.items():
            if isinstance(val, str) and 'http' in val and any(ext in val.lower() for ext in ['.jpg', '.png', '.jpeg', '.webp']):
                all_image_urls.append({"doc_id": doc_id, "field": key, "url": val})
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, str) and 'http' in item:
                        all_image_urls.append({"doc_id": doc_id, "field": key, "url": item})
                    elif isinstance(item, dict):
                        for sk, sv in item.items():
                            if isinstance(sv, str) and 'http' in sv:
                                all_image_urls.append({"doc_id": doc_id, "field": f"{key}.{sk}", "url": sv})

    next_page_token = data.get("nextPageToken")
    if page_count % 10 == 0:
        print(f"  Page {page_count}: {total_docs} docs, {len(all_image_urls)} images so far")

print(f"\nTotal documents: {total_docs}")
print(f"Total image URLs found: {len(all_image_urls)}")

if all_image_urls:
    print("\nSample URLs:")
    for img in all_image_urls[:10]:
        print(f"  [{img['field']}] {img['url'][:120]}")

output_path = "/teamspace/studios/this_studio/firestore_gen_images.json"
with open(output_path, "w") as f:
    json.dump(all_image_urls, f, indent=2)
print(f"\nSaved to {output_path}")
