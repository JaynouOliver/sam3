# Mattoboard — SAM3 Autodistill Pipeline

## What This Is
A **teacher-student knowledge distillation pipeline** for [Mattoboard](https://mattoboard.com) (interior design product). SAM3 (Segment Anything 3) acts as the zero-shot "teacher" to auto-label interior room images with 19 material/surface categories, then we train a lightweight YOLOv8n "student" model for fast inference in production.

**Ticket**: Investigate Autodistill SAM3 for ultra-fast segmentation (20 material concepts)

## Key References
- [autodistill-sam3 repo](https://github.com/autodistill/autodistill-sam3)
- [Autodistill docs](https://docs.autodistill.com/)
- Firebase project: `mattoboard-staging`

## Pipeline Flow
```
Firebase/Firestore (mattoboard-staging, dsCollectionGenerator collection)
  → fetch_gen_rooms.py → firestore_gen_room_images.json (1,610 rooms + 8,456 closeups)
  → download_room_images.py → room_images/ (1,000 sampled PNGs)
  → SAM3 auto-labeling (train_pipeline.py)
      - SegmentAnything3 with CaptionOntology (19 classes)
      - NO FILTERS — all raw SAM3 detections used as-is
      - Outputs YOLO-format .txt segmentation polygon labels
  → YOLO dataset split (85/15 train/val)
  → YOLOv8n-seg training → pipeline_training/sam3_distilled/weights/best.pt
```

## CRITICAL: No Filters on SAM3 Output
Previously we applied conf >= 0.65, area >= 400px, and class-agnostic NMS.
This was **removed** on 2026-03-04 because:
1. **Confidence filter killed walls**: SAM3 returns walls at 0.53-0.59 confidence. The 0.65 threshold dropped 10 of 13 wall segments on test images.
2. **Class-agnostic NMS suppressed walls**: Walls overlap with furniture/decor. Class-agnostic NMS treated them as duplicates and deleted the lower-confidence one (usually walls).
3. **No same-class overlaps exist**: Analysis of raw SAM3 output showed wall segments tile adjacent regions — they don't stack on each other. NMS was solving a problem that didn't exist.
4. **1,000 images is a small dataset**: Filtering reduces training signal. With limited data, we want every detection SAM3 provides.

The raw SAM3 output (~19 detections/image for this ontology) is already reasonable and doesn't need filtering.

## The 19-Class Ontology (SAM3 prompt → YOLO label)
ceiling→ceilings, curtain→curtains, decorative object→decor, floor→floors,
upholstered furniture→upholstery, wall→walls, countertop surface→worktop_surface,
board accessory→board_accessory, faucet→faucet_tap, light fixture→fixtures,
door handle→handle, cabinet knob→knob, hardware fitting→other_hardware,
outdoor fabric→outdoor_fabric, outdoor paving stone→outdoor_paver,
stair rod→stair_rod, light switch→switch, wallpaper→wallpaper_wallcovering,
background→na

## Key Files (actively used)
| File | Purpose |
|------|---------|
| `train_pipeline.py` | **THE main script** — end-to-end: SAM3 label → dataset → YOLOv8n-seg train |
| `compare_trained.py` | SAM3 vs trained YOLO side-by-side HTML comparison |
| `visualize_sam3.py` | SAM3 visualization + HTML comparison output |
| `sam3_single_image.py` | Run SAM3 on a single image for debugging |
| `fetch_gen_rooms.py` | Fetch room image URLs from Firestore |
| `download_room_images.py` | Download sampled room images |

## Superseded / Legacy Files (do not use)
`full_pipeline.py`, `main.py`, `autolabel_sam3.py`, `parallel_label.py`,
`prepare_yolo_dataset.py`, `train_yolo.py`, `run_inference.py`,
`fetch_firestore_images.py`, `fetch_firestore_gen.py`,
`fetch_products.py`, `download_images.py`,
`compare_outputs.py`, `visualize_yolo_view.py`

## Running the Pipeline
```bash
# Test run (10 images, 20 epochs, ~5 min)
# Set NUM_IMAGES = 10 in train_pipeline.py, then:
python train_pipeline.py

# Full run (1000 images, 150 epochs, ~145 min)
# Set NUM_IMAGES = 1000 in train_pipeline.py, then:
python train_pipeline.py
```

Output directories (all gitignored):
- `pipeline_labels/` — SAM3 auto-generated YOLO segmentation labels
- `pipeline_dataset/` — Train/val split dataset with symlinked images
- `pipeline_training/` — YOLO training outputs (best.pt, metrics)

## Data Directories (all gitignored)
- `room_images/` — 1,000 room scene PNGs (source images)
- `firestore_gen/` — ~100 original downloaded images
- `product_images/` — 999 product images from Supabase
- `autodistill-sam3/` — Local editable clone of the package

## Training History
| Run | Images | Filters | Model | Wall mAP50 | Notes |
|-----|--------|---------|-------|------------|-------|
| sam3_distilled4 (2026-03-03) | 1,000 | conf≥0.65, area≥400, NMS agnostic | yolov8l-seg | 0.51 | Walls underperformed due to filtering |
| sam3_distilled5 (pending) | 1,000 | **none** | yolov8n-seg | TBD | Raw SAM3 output, expect better walls |

## Dependencies (no requirements.txt yet)
autodistill, autodistill-sam3, supervision, inference, roboflow,
torch, torchvision, ultralytics, firebase-admin, requests, psycopg2, scikit-learn

## Notes
- Roboflow API key and Supabase credentials are hardcoded in source files
- SAM3 is GPU-required; the local autodistill-sam3 clone is installed in editable mode
- Existing trained weights: `yolov8n.pt`, `yolov8n-seg.pt` (pretrained), `yolov8l-seg.pt`, `yolo26n.pt`
