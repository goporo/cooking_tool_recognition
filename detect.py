import cv2
import numpy as np
from ultralytics import YOLO
from collections import Counter
import os

CONFIDENCE_THRESHOLD = 0.5
MODEL_PATH = "runs/detect/train5/weights/best.pt"
# MODEL_PATH = "models/kitchen_utensils_v1.pt"
def detect_and_count(image_path, model_path=MODEL_PATH, conf_threshold=CONFIDENCE_THRESHOLD):
    # If you want to use the best weights from training, use the following line
    model = YOLO(model_path)

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image {image_path} not found!")
        return
    
    results = model(image)
    object_counts = Counter()

    class_names = model.names

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue  
            
            label = class_names.get(cls, f"Unknown_{cls}")
            object_counts[label] += 1
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image, object_counts

def process_images(test_path="test/", model_path=MODEL_PATH, conf_threshold=CONFIDENCE_THRESHOLD):
    os.makedirs("output", exist_ok=True)
    
    print("Processing test images...")
    test_images = [img for img in os.listdir(test_path) if img.endswith((".jpg", ".png"))]
    if not test_images:
        print("No test images found!")
        return
    
    for test_image in test_images:
        test_image_path = os.path.join(test_path, test_image)
        output_image, object_counts = detect_and_count(test_image_path, model_path, conf_threshold)
        
        output_file_path = os.path.join("output", f"{os.path.splitext(test_image)[0]}_output.jpg")
        cv2.imwrite(output_file_path, output_image)
        print(f"✅ Output saved to {output_file_path}")
        
        total_count = sum(object_counts.values())
        print(f"\n🔹 {test_image}\n Total {total_count} objects detected")
        for obj, count in object_counts.items():
            print(f"   {obj}: {count}")

    cv2.imshow("Detected Objects", output_image)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_images()
