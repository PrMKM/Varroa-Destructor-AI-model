from ultralytics import YOLO
import os

model = YOLO("runs/detect/train/weights/best.pt")

test_path = "images/test"

total_count = 0

for img_name in os.listdir(test_path):
    if img_name.endswith(".jpg"):
        img_path = os.path.join(test_path, img_name)

        results = model(img_path, save=True)

        boxes = results[0].boxes

        count = 0

        for box in boxes:
            conf = float(box.conf)
            if conf > 0.4:
                count += 1

        print(f"{img_name} → {count} Varroa")

        total_count += count

print("\nŁącznie wykryto Varroa:", total_count)