from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()

# Load YOLO model (YOLOv8n is lightweight)
model = YOLO("yolov8n.pt")

@app.post("/detect-drones/")
async def detect_drones(image: UploadFile = File(...)):
    # Read image bytes
    image_bytes = await image.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Run model
    results = model(img, conf=0.25)

    detections = []
    count = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label.lower() == "drone":  # If the model class name is "drone"
                count += 1

            detections.append({
                "class": label,
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist()
            })

    return {
        "total_drones": count,
        "detections": detections
    }
