import torch
from ultralytics import YOLO

if __name__ == "__main__":
    torch.cuda.empty_cache()  # Clear CUDA cache to free memory
    model = YOLO("models/yolo11m.pt")  # Ensure correct model selection

    model.train(
        data="datasets/data.yaml",  # Dataset configuration file
        epochs=150,                 # Reduce epochs if convergence is fast
        batch=6,                    # Increase batch size if VRAM allows
        imgsz=416,                  # Reduce image size for faster processing
        device="cuda",               # Use GPU
        lr0=0.003,                   # Slightly higher learning rate for quicker convergence
        workers=2,                    # Increase workers for data loading speed
        pretrained=True,             # Use pretrained weights
        augment=False,               # Disable heavy augmentations
        cache=False,                 # Disable caching to save memory
        half=True,                    # Use FP16 for faster computation
        dropout=0.05,                 # Reduce dropout slightly
        patience=20                   # Reduce patience to speed up early stopping
    )
