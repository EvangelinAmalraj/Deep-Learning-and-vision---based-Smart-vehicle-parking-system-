from ultralytics import YOLO

# Load a pretrained model
model = YOLO("yolov8s.pt")  # small model (good balance)

# Train the model
model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640
)
