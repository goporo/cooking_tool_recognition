import torch
from ultralytics import YOLO

if __name__ == "__main__":
    torch.cuda.empty_cache()  # Clear cache to avoid memory issues
    model = YOLO("models/yolo11m.pt")

    model.train(
        data="datasets/data.yaml",  # Dataset config
        epochs=250,                # Reduce epochs if needed
        batch=2,                   # Reduce batch size to fit in VRAM
        imgsz=416,                 # Reduce image size to 416
        device="cuda",             # Use GPU
        lr0=0.005,                 # Learning rate
        workers=0,                 # Reduce worker threads to save memory
        pretrained=True,           # Use pretrained weights
        augment=True,              # Enable data augmentation
        cache=False,               # Avoid caching in memory
        half=True,                 # Use FP16 (lower precision)
    )
