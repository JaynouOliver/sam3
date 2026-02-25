"""
Prepare YOLO dataset from SAM3 auto-labels.
Splits into train/val (80/20) and creates dataset.yaml.
"""
import os, shutil, random, yaml

LABELS_DIR = "/teamspace/studios/this_studio/sam3_labels/labels"
IMAGES_DIR = "/teamspace/studios/this_studio/firestore_gen"
DATASET_DIR = "/teamspace/studios/this_studio/yolo_dataset"
SPLIT_RATIO = 0.8
SEED = 42

CLASSES = [
    "ceilings", "curtains", "decor", "floors", "upholstery", "walls",
    "worktop_surface", "board_accessory", "faucet_tap", "fixtures",
    "handle", "knob", "other_hardware", "outdoor_fabric", "outdoor_paver",
    "stair_rod", "switch", "wallpaper_wallcovering", "na",
]

# Create directory structure
for split in ["train", "val"]:
    os.makedirs(os.path.join(DATASET_DIR, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, split, "labels"), exist_ok=True)

# Get all label files and match to images
label_files = sorted([f for f in os.listdir(LABELS_DIR) if f.endswith(".txt")])
pairs = []
for lf in label_files:
    stem = os.path.splitext(lf)[0]
    # Find matching image
    for ext in [".png", ".jpg", ".jpeg"]:
        img_path = os.path.join(IMAGES_DIR, stem + ext)
        if os.path.exists(img_path):
            pairs.append((img_path, os.path.join(LABELS_DIR, lf)))
            break

print(f"Found {len(pairs)} image-label pairs")

# Skip pairs with empty labels
valid_pairs = []
for img, lbl in pairs:
    if os.path.getsize(lbl) > 0:
        valid_pairs.append((img, lbl))
print(f"Valid (non-empty labels): {len(valid_pairs)}")

# Shuffle and split
random.seed(SEED)
random.shuffle(valid_pairs)
split_idx = int(len(valid_pairs) * SPLIT_RATIO)
train_pairs = valid_pairs[:split_idx]
val_pairs = valid_pairs[split_idx:]
print(f"Train: {len(train_pairs)} | Val: {len(val_pairs)}")

# Copy files
for split_name, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
    for img_path, lbl_path in split_pairs:
        img_fname = os.path.basename(img_path)
        lbl_fname = os.path.basename(lbl_path)
        shutil.copy2(img_path, os.path.join(DATASET_DIR, split_name, "images", img_fname))
        shutil.copy2(lbl_path, os.path.join(DATASET_DIR, split_name, "labels", lbl_fname))

# Create dataset.yaml
dataset_config = {
    "path": DATASET_DIR,
    "train": "train/images",
    "val": "val/images",
    "nc": len(CLASSES),
    "names": CLASSES,
}

yaml_path = os.path.join(DATASET_DIR, "dataset.yaml")
with open(yaml_path, "w") as f:
    yaml.dump(dataset_config, f, default_flow_style=False)

print(f"\nDataset ready at: {DATASET_DIR}")
print(f"Config: {yaml_path}")
print(f"Train images: {len(os.listdir(os.path.join(DATASET_DIR, 'train', 'images')))}")
print(f"Val images: {len(os.listdir(os.path.join(DATASET_DIR, 'val', 'images')))}")
