"""
Modal.com deployment for Mattoboard YOLOv8l-seg model.

Deploy:   modal deploy deploy_modal_large.py
Test:     modal serve deploy_modal_large.py
"""

import modal
import io
import base64

app = modal.App("mattoboard-segmentation-large")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "ultralytics",
        "opencv-python-headless",
        "pillow",
        "numpy",
        "fastapi[standard]",
    )
    .add_local_file(
        "runs/segment/pipeline_training/sam3_distilled_l/weights/best.pt",
        "/model/best.pt",
    )
)


@app.cls(image=image, gpu="T4", scaledown_window=120)
class Segmenter:
    @modal.enter()
    def load_model(self):
        from ultralytics import YOLO
        import numpy as np

        self.model = YOLO("/model/best.pt")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(dummy, verbose=False)
        print("YOLOv8l-seg model loaded and warmed up")

    @modal.fastapi_endpoint(method="POST", docs=True)
    def predict(self, request: dict):
        """
        Segment an image and return detections with masks.

        POST JSON body:
        {
            "image_base64": "<base64-encoded image>",
            "conf_threshold": 0.25  (optional, default 0.25)
        }
        """
        import numpy as np
        from PIL import Image

        img_bytes = base64.b64decode(request["image_base64"])
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img)

        conf = request.get("conf_threshold", 0.25)
        results = self.model.predict(img_np, conf=conf, verbose=False)[0]

        detections = []
        names = results.names

        if results.boxes is not None:
            for i, box in enumerate(results.boxes):
                det = {
                    "class_id": int(box.cls[0]),
                    "class_name": names[int(box.cls[0])],
                    "confidence": round(float(box.conf[0]), 4),how do 
                    "bbox": [round(v, 1) for v in box.xyxy[0].tolist()],
                }
                if results.masks is not None and i < len(results.masks):
                    mask_xy = results.masks[i].xy
                    if len(mask_xy) > 0:
                        points = mask_xy[0].tolist()
                        det["polygon"] = [[round(x, 1), round(y, 1)] for x, y in points]
                detections.append(det)

        return {
            "image_size": {"width": img_np.shape[1], "height": img_np.shape[0]},
            "num_detections": len(detections),
            "detections": detections,
        }

    @modal.fastapi_endpoint(method="GET", docs=True)
    def health(self):
        """Health check endpoint."""
        return {"status": "ok", "model": "yolov8l-seg", "classes": 19, "size_mb": 92.3}
