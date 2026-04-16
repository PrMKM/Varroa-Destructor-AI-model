# from ultralytics import YOLO
# import os

# model = YOLO("runs/detect/train/weights/best.pt")

# test_path = "images/test"

# total_count = 0

# for img_name in os.listdir(test_path):
#     if img_name.endswith(".jpg"):
#         img_path = os.path.join(test_path, img_name)

#         results = model(img_path, save=True)

#         boxes = results[0].boxes

#         count = 0

#         for box in boxes:
#             conf = float(box.conf)
#             if conf > 0.3:
#                 count += 1

#         print(f"{img_name} → {count} Varroa")

#         total_count += count

# print("\nŁącznie wykryto Varroa:", total_count)

# from ultralytics import YOLO
# import cv2
# import os

# model = YOLO("runs/detect/train19/weights/best.pt")

# SLICE_SIZE = 640
# OVERLAP = 0.2

# def slice_and_predict(img_path):
#     img = cv2.imread(img_path)
#     H, W = img.shape[:2]

#     step = int(SLICE_SIZE * (1 - OVERLAP))
#     total = 0

#     for y in range(0, H, step):
#         for x in range(0, W, step):

#             tile = img[y:y+SLICE_SIZE, x:x+SLICE_SIZE]

#             results = model(tile)

#             boxes = results[0].boxes

#             for box in boxes:
#                 if float(box.conf) > 0.3:
#                     total += 1

#     return total


# test_path = "images/test"

# global_count = 0

# for img_name in os.listdir(test_path):
#     if img_name.endswith(".jpg"):
#         path = os.path.join(test_path, img_name)

#         count = slice_and_predict(path)

#         print(f"{img_name} → {count}")
#         global_count += count

# print("\nTOTAL:", global_count)

# from ultralytics import YOLO
# import cv2
# import os

# model = YOLO("runs/detect/train19/weights/best.pt")

# SLICE_SIZE = 640
# OVERLAP = 0.2
# CONF = 0.35

# INPUT = "images/test"
# OUT_TILES = "results/tiles"
# OUT_FULL = "results/full"

# os.makedirs(OUT_TILES, exist_ok=True)
# os.makedirs(OUT_FULL, exist_ok=True)


# def slice_and_predict(img_path, img_name):
#     img = cv2.imread(img_path)
#     H, W = img.shape[:2]

#     step = int(SLICE_SIZE * (1 - OVERLAP))

#     total = 0
#     all_boxes = []

#     tile_id = 0

#     for y in range(0, H, step):
#         for x in range(0, W, step):

#             tile = img[y:y+SLICE_SIZE, x:x+SLICE_SIZE]

#             results = model(tile, verbose=False)
#             boxes = results[0].boxes

#             # kopia do rysowania kafelka
#             tile_vis = tile.copy()

#             for box in boxes:
#                 conf = float(box.conf)

#                 if conf > CONF:
#                     total += 1

#                     x1, y1, x2, y2 = map(int, box.xyxy[0])

#                     # 🔹 rysowanie na kafelku
#                     cv2.rectangle(tile_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

#                     # 🔹 przelicz na globalne współrzędne
#                     gx1 = x1 + x
#                     gy1 = y1 + y
#                     gx2 = x2 + x
#                     gy2 = y2 + y

#                     all_boxes.append((gx1, gy1, gx2, gy2))

#             # zapis kafelka (debug)
#             tile_name = f"{img_name}_tile_{tile_id}.jpg"
#             cv2.imwrite(os.path.join(OUT_TILES, tile_name), tile_vis)

#             tile_id += 1

#     # rysowanie na pełnym obrazie
#     full_vis = img.copy()

#     for (x1, y1, x2, y2) in all_boxes:
#         cv2.rectangle(full_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

#     cv2.imwrite(os.path.join(OUT_FULL, img_name), full_vis)

#     return total


# global_count = 0

# for img_name in os.listdir(INPUT):
#     if img_name.endswith(".jpg"):
#         path = os.path.join(INPUT, img_name)

#         count = slice_and_predict(path, img_name)

#         print(f"{img_name} → {count}")
#         global_count += count

# print("\nTOTAL:", global_count)

from ultralytics import YOLO
import cv2
import os
import numpy as np

model = YOLO("runs/detect/train19/weights/best.pt")

SLICE_SIZE = 640
OVERLAP = 0.2
CONF = 0.35

INPUT = "images/test"
OUT_FULL = "results/full"

os.makedirs(OUT_FULL, exist_ok=True)


# ================= NMS =================
def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def nms(boxes, scores, iou_threshold=0.5):
    idxs = np.argsort(scores)[::-1]
    keep = []

    while len(idxs) > 0:
        current = idxs[0]
        keep.append(current)

        rest = []
        for i in idxs[1:]:
            if iou(boxes[current], boxes[i]) < iou_threshold:
                rest.append(i)

        idxs = np.array(rest)

    return keep
# =======================================


def slice_and_predict(img_path, img_name):
    img = cv2.imread(img_path)
    H, W = img.shape[:2]

    step = int(SLICE_SIZE * (1 - OVERLAP))

    all_boxes = []
    all_scores = []

    for y in range(0, H, step):
        for x in range(0, W, step):

            tile = img[y:y+SLICE_SIZE, x:x+SLICE_SIZE]

            results = model(tile, verbose=False)
            boxes = results[0].boxes

            for box in boxes:
                conf = float(box.conf)

                if conf > CONF:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    gx1 = x1 + x
                    gy1 = y1 + y
                    gx2 = x2 + x
                    gy2 = y2 + y

                    all_boxes.append((gx1, gy1, gx2, gy2))
                    all_scores.append(conf)

    # 🔥 NMS (usuwa duplikaty)
    if len(all_boxes) > 0:
        keep = nms(all_boxes, all_scores, iou_threshold=0.5)

        boxes = [all_boxes[i] for i in keep]
        scores = [all_scores[i] for i in keep]
    else:
        boxes = []
        scores = []

    # 🔥 rysowanie
    vis = img.copy()

    for (box, score) in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(vis, f"{score:.2f}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    cv2.imwrite(os.path.join(OUT_FULL, img_name), vis)

    return len(boxes)


# ================= MAIN =================
global_count = 0

for img_name in os.listdir(INPUT):
    if img_name.endswith(".jpg"):
        path = os.path.join(INPUT, img_name)

        count = slice_and_predict(path, img_name)

        print(f"{img_name} → {count}")
        global_count += count

print("\nTOTAL:", global_count)