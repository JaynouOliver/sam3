# Mattoboard Segmentation API

Deployed on [Modal.com](https://modal.com) — YOLOv8n-seg model trained via SAM3 knowledge distillation.

## Endpoints

### Health Check

```
GET https://mattoboard--mattoboard-segmentation-segmenter-health.modal.run
```

**Response:**
```json
{"status": "ok", "model": "yolov8n-seg", "classes": 19}
```

### Predict (Segmentation)

```
POST https://mattoboard--mattoboard-segmentation-segmenter-predict.modal.run
Content-Type: application/json
```

**Request body:**
```json
{
  "image_base64": "<base64-encoded PNG/JPG>",
  "conf_threshold": 0.25
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `image_base64` | string | yes | — | Base64-encoded image (PNG or JPG) |
| `conf_threshold` | float | no | 0.25 | Minimum confidence threshold (0.0–1.0) |

**Response:**
```json
{
  "image_size": {"width": 1152, "height": 896},
  "num_detections": 64,
  "detections": [
    {
      "class_id": 2,
      "class_name": "decor",
      "confidence": 0.957,
      "bbox": [x1, y1, x2, y2],
      "polygon": [[x1, y1], [x2, y2], ...]
    }
  ]
}
```

**Detection fields:**

| Field | Type | Description |
|-------|------|-------------|
| `class_id` | int | Class index (0–18) |
| `class_name` | string | Human-readable class name |
| `confidence` | float | Detection confidence (0.0–1.0) |
| `bbox` | array | Bounding box `[x1, y1, x2, y2]` in pixels |
| `polygon` | array | Segmentation mask as `[[x, y], ...]` polygon points in pixels |

## Classes (19)

| ID | Name | SAM3 Prompt |
|----|------|-------------|
| 0 | ceilings | ceiling |
| 1 | curtains | curtain |
| 2 | decor | decorative object |
| 3 | floors | floor |
| 4 | upholstery | upholstered furniture |
| 5 | walls | wall |
| 6 | worktop_surface | countertop surface |
| 7 | board_accessory | board accessory |
| 8 | faucet_tap | faucet |
| 9 | fixtures | light fixture |
| 10 | handle | door handle |
| 11 | knob | cabinet knob |
| 12 | other_hardware | hardware fitting |
| 13 | outdoor_fabric | outdoor fabric |
| 14 | outdoor_paver | outdoor paving stone |
| 15 | stair_rod | stair rod |
| 16 | switch | light switch |
| 17 | wallpaper_wallcovering | wallpaper |
| 18 | na | background |

## Usage Examples

### Python

```python
import base64
import requests

# Encode image
with open("room.png", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

# Call API
resp = requests.post(
    "https://mattoboard--mattoboard-segmentation-segmenter-predict.modal.run",
    json={"image_base64": img_b64, "conf_threshold": 0.25},
    timeout=120,
)

data = resp.json()
print(f"Found {data['num_detections']} segments")
for det in data["detections"]:
    print(f"  {det['class_name']}: {det['confidence']:.2f}")
```

### cURL

```bash
IMG_B64=$(base64 -w0 room.png)
curl -X POST \
  https://mattoboard--mattoboard-segmentation-segmenter-predict.modal.run \
  -H "Content-Type: application/json" \
  -d "{\"image_base64\": \"$IMG_B64\"}"
```

### JavaScript

```javascript
const fs = require("fs");

const imgB64 = fs.readFileSync("room.png").toString("base64");
const resp = await fetch(
  "https://mattoboard--mattoboard-segmentation-segmenter-predict.modal.run",
  {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image_base64: imgB64, conf_threshold: 0.25 }),
  }
);
const data = await resp.json();
console.log(`Found ${data.num_detections} segments`);
```

## Interactive Docs (Swagger UI)

Both endpoints have auto-generated docs at:
- **Predict:** `https://mattoboard--mattoboard-segmentation-segmenter-predict.modal.run/docs`
- **Health:** `https://mattoboard--mattoboard-segmentation-segmenter-health.modal.run/docs`

## Deployment

```bash
# Deploy (production)
MODAL_PROFILE=mattoboard modal deploy deploy_modal.py

# Dev server (hot-reload)
MODAL_PROFILE=mattoboard modal serve deploy_modal.py
```

## Model Details

- **Architecture:** YOLOv8n-seg (nano, ~7MB)
- **Training data:** 992 room images auto-labeled by SAM3 (no filters)
- **Training run:** `sam3_distilled6` — 150 epochs, batch 16, imgsz 640
- **GPU:** NVIDIA T4 on Modal
- **Cold start:** ~10-15s (model load + warmup); warm requests are sub-second
- **Scale-down window:** 120s (container stays warm for 2 min after last request)

## Dashboard

View deployment status and logs:
https://modal.com/apps/mattoboard/main/deployed/mattoboard-segmentation
