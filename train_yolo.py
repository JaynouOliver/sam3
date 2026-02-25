"""
Train YOLOv8 student model on SAM3 auto-labeled dataset.
Using YOLOv8n (nano) for maximum speed — the whole point of distillation.
"""
from ultralytics import YOLO

DATASET_YAML = "/teamspace/studios/this_studio/yolo_dataset/dataset.yaml"
PROJECT_DIR = "/teamspace/studios/this_studio/yolo_training"

# YOLOv8n = nano (fastest, smallest)
# YOLOv8s = small (slightly better accuracy, still fast)
model = YOLO("yolov8n.pt")

results = model.train(
    data=DATASET_YAML,
    epochs=100,
    imgsz=640,
    batch=16,
    patience=20,          # early stopping if no improvement for 20 epochs
    project=PROJECT_DIR,
    name="sam3_distilled",
    device=0,             # GPU
    workers=4,
    verbose=True,
    # Augmentation for small dataset
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    flipud=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
    scale=0.5,
)

print("\n" + "="*50)
print("Training complete!")
print(f"Best model: {PROJECT_DIR}/sam3_distilled/weights/best.pt")
print(f"Results: {PROJECT_DIR}/sam3_distilled/")
