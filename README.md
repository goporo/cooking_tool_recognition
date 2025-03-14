### 🥘Cooking Tools Recognition
Project: Cooking Tools Recognition

GPU Used: NVIDIA GTX 1650

📖 Table of Contents
Project Overview
Installation
Dataset Preparation
Training the Model
Evaluating Model Performance
Running Inference
Optimizations for GTX 1650
Troubleshooting
Running the Scripts
Summary Report

📝 1. Project Overview
This project trains a YOLO model to detect cooking tools in images or videos. It uses the Ultralytics YOLO framework and is optimized for a GTX 1650 GPU.

Key Features:
✅ Custom YOLO training on a cooking tool dataset
✅ Optimized training for low-VRAM GPUs
✅ Object detection on images, videos, or live webcam feeds

🔧 2. Installation
Ensure you have Python 3.8+ and PyTorch installed.

Step 1: Install Required Libraries
```bash
pip install ultralytics
pip install opencv-python pillow tqdm
```
Step 2: Verify Installation
Run the following to check if YOLO is installed:

```python
from ultralytics import YOLO
print(YOLO('yolo11m.pt'))
```
If no errors occur, installation is successful.

📂 3. Dataset Preparation
Step 1: Organize the Dataset
Ensure your dataset follows the YOLO format:

```bash
datasets/
│── train/
│   ├── images/  # Training images
│   ├── labels/  # YOLO format labels
│── valid/
│   ├── images/  # Validation images
│   ├── labels/  # YOLO format labels
│── test/
│   ├── images/  # Test images
│   ├── labels/  # Test labels
```
Step 2: Create data.yaml File
Create a data.yaml file inside datasets/:

```yaml
train: datasets/train/images
val: datasets/valid/images
test: datasets/test/images

nc: 2  # Number of classes the model can regconize
names: ['Fork', 'Spoon']
```
🎯 4. Training the Model
Start Training (Optimized for GTX 1650)
```python
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11m.pt")  # Use a lightweight model for better performance

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
```
Estimated training time: ~2-4 minutes per epoch.

🚀 Running the Scripts
To train the model, run:
```bash
python train.py
```
To detect objects in test images, run:
```bash
python detect.py
```

🎉 Summary Report
This guide walks you through installing YOLO, preparing the dataset, training the model, evaluating performance, and running inference. It is optimized for GTX 1650 to prevent memory issues. The model is trained to detect 16 different cooking tools with optimized settings for low-VRAM GPUs. The training process is efficient, taking approximately 2-4 minutes per epoch.
