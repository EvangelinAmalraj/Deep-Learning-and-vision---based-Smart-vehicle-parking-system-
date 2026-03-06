import os
from ultralytics import YOLO
import torch

# build an absolute path to the dataset config so the script
# works no matter what the current working directory is
base_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(base_dir, "data.yaml")

# determine compute device
if torch.cuda.is_available():
    sel_device = 0
    print(f"CUDA available, training on GPU {sel_device}")
else:
    sel_device = 'cpu'
    print("No CUDA devices detected; training on CPU")

if __name__ == '__main__':
    # Load a pretrained model on the first available GPU
    model = YOLO("yolo11n.pt")  # small model (good balance)

    # Train the model
    model.train(
        data=data_file,
        epochs=70,
        patience=20,
        imgsz=640,
        batch=16,
        device=sel_device,
        name="svp_final"
    )