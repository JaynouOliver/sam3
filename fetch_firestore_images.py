"""
Fetch generated images from Firestore dsCollectionGenerator collection.
Project: mattoboard-staging
READ-ONLY — no writes.
"""
import firebase_admin
from firebase_admin import credentials, firestore
import json

# Initialize with default credentials (from firebase login)
# For service account, use: credentials.Certificate('path/to/key.json')
app = firebase_admin.initialize_app(options={
    'projectId': 'mattoboard-staging'
})

db = firestore.client()

# Query dsCollectionGenerator collection
print("Fetching dsCollectionGenerator collection...")
collection_ref = db.collection('dsCollectionGenerator')

# First, get a small sample to understand the schema
docs = collection_ref.limit(5).stream()

print("\n--- Schema exploration (first 5 docs) ---\n")
sample_docs = []
for doc in docs:
    data = doc.to_dict()
    sample_docs.append(data)
    print(f"Doc ID: {doc.id}")
    print(f"  Fields: {list(data.keys())}")
    # Print values, truncating long ones
    for k, v in data.items():
        val_str = str(v)
        if len(val_str) > 200:
            val_str = val_str[:200] + "..."
        print(f"  {k}: {val_str}")
    print()

# Now get total count
print("Counting total documents...")
all_docs = collection_ref.stream()
count = 0
image_urls = []
for doc in all_docs:
    count += 1
    data = doc.to_dict()
    # Look for image-related fields
    for key in data:
        val = data[key]
        if isinstance(val, str) and ('http' in val and any(ext in val.lower() for ext in ['.jpg', '.png', '.jpeg', '.webp'])):
            image_urls.append({'doc_id': doc.id, 'field': key, 'url': val})
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, str) and ('http' in item and any(ext in item.lower() for ext in ['.jpg', '.png', '.jpeg', '.webp'])):
                    image_urls.append({'doc_id': doc.id, 'field': key, 'url': item})
                elif isinstance(item, dict):
                    for subk, subv in item.items():
                        if isinstance(subv, str) and ('http' in subv and any(ext in subv.lower() for ext in ['.jpg', '.png', '.jpeg', '.webp'])):
                            image_urls.append({'doc_id': doc.id, 'field': f"{key}.{subk}", 'url': subv})

print(f"\nTotal documents: {count}")
print(f"Image URLs found: {len(image_urls)}")

if image_urls:
    print("\nSample image URLs:")
    for img in image_urls[:10]:
        print(f"  [{img['field']}] {img['url'][:120]}")

# Save all image URLs
output_path = "/teamspace/studios/this_studio/firestore_gen_images.json"
with open(output_path, "w") as f:
    json.dump(image_urls, f, indent=2)
print(f"\nSaved to {output_path}")
