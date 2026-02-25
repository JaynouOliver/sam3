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
  → SAM3 auto-labeling (full_pipeline.py or autolabel_sam3.py / parallel_label.py)
      - SegmentAnything3 with CaptionOntology (19 classes)
      - Filters: conf >= 0.65, area >= 400px, NMS 0.5 IoU
      - Outputs YOLO-format .txt labels
  → prepare_yolo_dataset.py → yolo_dataset/ (train/val split)
  → train_yolo.py → yolo_training/sam3_distilled/weights/best.pt
```

## The 19-Class Ontology (SAM3 prompt → YOLO label)
ceiling→ceilings, curtain→curtains, decorative object→decor, floor→floors,
upholstered furniture→upholstery, wall→walls, countertop surface→worktop_surface,
board accessory→board_accessory, faucet→faucet_tap, light fixture→fixtures,
door handle→handle, cabinet knob→knob, hardware fitting→other_hardware,
outdoor fabric→outdoor_fabric, outdoor paving stone→outdoor_paver,
stair rod→stair_rod, light switch→switch, wallpaper→wallpaper_wallcovering,
background→na

## Key Files
| File | Purpose |
|------|---------|
| `fetch_firestore_images.py` | Fetch image URLs from Firestore (SDK) |
| `fetch_firestore_gen.py` | Fetch image URLs from Firestore (REST API) |
| `fetch_gen_rooms.py` | Extract room scene + closeup URLs specifically |
| `fetch_products.py` | Fetch product images from Supabase |
| `download_room_images.py` | Download 1,000 sampled room images |
| `download_images.py` | Download 1,000 sampled product images |
| `autolabel_sam3.py` | Single-threaded SAM3 labeling (firestore_gen/) |
| `parallel_label.py` | Optimized producer-consumer labeling (product_images/) |
| `run_inference.py` | Single-image SAM3 test/demo |
| `prepare_yolo_dataset.py` | Build YOLO train/val dataset from labels |
| `train_yolo.py` | Train YOLOv8n student model |
| `full_pipeline.py` | End-to-end: label → dataset → train (room_images/) |
| `main.py` | Initial test script (fruit demo) |

## Data Directories (all gitignored)
- `firestore_gen/` — ~100 original downloaded images
- `room_images/` — 1,000 room scene PNGs
- `product_images/` — 999 product images from Supabase
- `sam3_labels/` — Labels for firestore_gen images
- `sam3_labels_1k/` — Labels for product images
- `yolo_dataset/` — Train/val split dataset
- `yolo_training/` — Training outputs (best.pt, metrics)
- `autodistill-sam3/` — Local editable clone of the package

## Dependencies (no requirements.txt yet)
autodistill, autodistill-sam3, supervision, inference, roboflow,
torch, torchvision, ultralytics, firebase-admin, requests, psycopg2, scikit-learn

## Notes
- Roboflow API key and Supabase credentials are hardcoded in source files
- SAM3 is GPU-required; the local autodistill-sam3 clone is installed in editable mode
- Existing trained weights: `yolov8n.pt` (pretrained), `yolo26n.pt`
