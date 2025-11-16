# train.py
from ultralytics import YOLO

# Load pre-trained YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Train the model (example code)
model.train(
    data="data.yaml",  # dataset configuration
    epochs=50,
    imgsz=416,
    batch=16
)

# Save the trained model
model.save("yolov8_drone_final.pt")
