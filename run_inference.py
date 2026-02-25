import os, time
os.environ['ROBOFLOW_API_KEY'] = '6mPyaZWFhvbmBKftxcq7'

from autodistill_sam3 import SegmentAnything3
from autodistill.detection import CaptionOntology
import supervision as sv

ontology = CaptionOntology({
    "floor": "floor",
    "wall": "wall",
    "upholstery": "upholstery",
    "ceiling": "ceiling",
    "sofa": "sofa",
    "carpet": "carpet",
    "wood": "wood",
    "fabric": "fabric",
    "glass": "glass",
    "metal": "metal",
})

print("Loading model...")
t0 = time.time()
model = SegmentAnything3(ontology=ontology)
print(f"Model loaded in {time.time()-t0:.2f}s")

# Change this to your image path
IMAGE = "CalmTeaHouseupscaled.jpg"

print(f"\nRunning inference on {IMAGE} ...")
t1 = time.time()
detections = model.predict(IMAGE)
elapsed = time.time() - t1

print(f"Inference done in {elapsed:.2f}s")
print(f"Detections (raw): {len(detections)} total")

# Filter: min confidence 0.7, min bbox area 500px
import numpy as np
areas = (detections.xyxy[:, 2] - detections.xyxy[:, 0]) * (detections.xyxy[:, 3] - detections.xyxy[:, 1])
mask = (detections.confidence >= 0.7) & (areas >= 500)
detections = detections[mask]
print(f"Detections (filtered conf>=0.7, area>=500px): {len(detections)} total\n")

classes = ontology.classes()
for i, (xyxy, conf, class_id) in enumerate(zip(detections.xyxy, detections.confidence, detections.class_id)):
    label = classes[class_id]
    area = int((xyxy[2]-xyxy[0]) * (xyxy[3]-xyxy[1]))
    print(f"  [{i}] {label:15s}  conf={conf:.2f}  area={area:>8}px  bbox={xyxy.astype(int).tolist()}")
