from ultralytics import YOLO
import cv2
import os
import numpy as np

model = YOLO("runs/detect/train7/weights/best.pt")

SLICE_SIZE = 640
OVERLAP = 0.2
CONF = 0.25

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

    # NMS (usuwa duplikaty)
    if len(all_boxes) > 0:
        keep = nms(all_boxes, all_scores, iou_threshold=0.4)

        boxes = [all_boxes[i] for i in keep]
        scores = [all_scores[i] for i in keep]
    else:
        boxes = []
        scores = []

    #rysowanie
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