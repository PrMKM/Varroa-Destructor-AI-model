from ultralytics import YOLO

model = YOLO("runs/detect/train11/weights/best.pt")

results = model("images/train/img1.jpg", save=True)

boxes = results[0].boxes

count = 0

print("Liczba wykrytych Varroa:", len(boxes))

# for i, box in enumerate(boxes):
#     print(i, float(box.conf))

# for box in boxes:
#     conf = float(box.conf)
#     if conf > 0.4:
#         count += 1

# print("Liczba wykrytych Varroa:", count)