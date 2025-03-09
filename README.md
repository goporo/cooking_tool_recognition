📌 Cooking Tools Recognition
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

nc: 16  # Number of classes
names: 
['bottle', 'bowl', 'cup', 'cutting board', 'fork', 'fullbottle', 'fullbowl', 
'fullcup', 'fullpan', 'fullpot', 'knife', 'pan', 'plate', 'pot', 'spoon', 'whisk']
```
🎯 4. Training the Model
Start Training (Optimized for GTX 1650)
```python
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11m.pt")  # Use a lightweight model for better performance

    model.train(
        data="datasets/data.yaml",  # Path to dataset
        epochs=250,                 # Number of training epochs
        batch=2,                    # Reduce batch size to avoid OOM issues
        imgsz=416,                  # Smaller image size for faster training
        device="cuda",              # Use GPU
        lr0=0.005,                  # Learning rate
        workers=0,                  # Prevents multi-threading issues on Windows
        pretrained=True,            # Use pre-trained weights
        half=True                   # Enable mixed precision for better memory usage
    )
```
Estimated training time: ~2-4 minutes per epoch.

📊 5. Evaluating Model Performance
After training, evaluate your model using:

```python
model.val()
```
This provides mAP, precision, recall, and other metrics.

🔍 6. Running Inference
Detect Objects in an Image
```python
results = model("test_image.jpg", save=True, imgsz=416)
```
⚡ 7. Optimizations for GTX 1650
✅ Best Settings for Training on Low-VRAM GPU
Parameter	Recommended Value
Model	yolo11m.pt (Nano)
Batch Size	batch=2 (Reduce if OOM occurs)
Image Size	imgsz=416
Workers	workers=0 (Avoids CPU thread issues)
Mixed Precision	half=True (Reduces VRAM usage)
🔥 Additional Speed Optimizations
Reduce batch size → If OOM occurs, use batch=1.
Lower image size → imgsz=320 instead of 416.
Disable augmented training → Remove augment=True if memory issues occur.
Use CPU if necessary → device="cpu" if GPU crashes.
❌ 8. Troubleshooting
🔹 Out of Memory (OOM) Error
Solution: Reduce batch=1, lower imgsz=320, and enable half=True.
🔹 Training is Slow
Solution: Use yolo11m.pt instead of YOLOm.pt. Reduce dataset size for faster training.
🔹 Model Doesn't Detect Objects
Solution: Ensure labels are correctly formatted in YOLO format (.txt files with class x y w h).

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