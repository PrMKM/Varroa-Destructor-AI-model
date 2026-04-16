import os
import cv2

INPUT_IMAGES = "images/train"
INPUT_LABELS = "labels/train"

OUTPUT_IMAGES = "images_sliced/train"
OUTPUT_LABELS = "labels_sliced/train"

SLICE_SIZE = 640
OVERLAP = 0.2

os.makedirs(OUTPUT_IMAGES, exist_ok=True)
os.makedirs(OUTPUT_LABELS, exist_ok=True)


def load_labels(path):
    boxes = []
    if not os.path.exists(path):
        return boxes

    with open(path, "r") as f:
        for line in f:
            cls, x, y, w, h = map(float, line.strip().split())
            boxes.append((cls, x, y, w, h))
    return boxes


def save_labels(path, boxes):
    with open(path, "w") as f:
        for b in boxes:
            f.write(" ".join(map(str, b)) + "\n")


def yolo_to_xyxy(box, W, H):
    cls, x, y, w, h = box

    x *= W
    y *= H
    w *= W
    h *= H

    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2

    return cls, x1, y1, x2, y2


def xyxy_to_yolo(cls, x1, y1, x2, y2, tile_w, tile_h):
    w = x2 - x1
    h = y2 - y1

    x = x1 + w / 2
    y = y1 + h / 2

    return (
        cls,
        x / tile_w,
        y / tile_h,
        w / tile_w,
        h / tile_h
    )


def slice_image(img, boxes, name):
    H, W = img.shape[:2]
    step = int(SLICE_SIZE * (1 - OVERLAP))

    idx = 0

    for y in range(0, H, step):
        for x in range(0, W, step):

            x2 = min(x + SLICE_SIZE, W)
            y2 = min(y + SLICE_SIZE, H)

            tile = img[y:y2, x:x2]
            tile_h, tile_w = tile.shape[:2]

            tile_boxes = []

            for b in boxes:
                cls, bx1, by1, bx2, by2 = yolo_to_xyxy(b, W, H)

                # przecięcie bboxa z kafelkiem
                ix1 = max(bx1, x)
                iy1 = max(by1, y)
                ix2 = min(bx2, x2)
                iy2 = min(by2, y2)

                # jeśli brak przecięcia → skip
                if ix1 >= ix2 or iy1 >= iy2:
                    continue

                # przeniesienie do układu tile
                ix1 -= x
                iy1 -= y
                ix2 -= x
                iy2 -= y

                new_box = xyxy_to_yolo(cls, ix1, iy1, ix2, iy2, tile_w, tile_h)

                cls, nx, ny, nw, nh = new_box

                # KLUCZOWE: clamp + filtr śmieci
                if (
                    0 <= nx <= 1 and
                    0 <= ny <= 1 and
                    0 < nw <= 1 and
                    0 < nh <= 1
                ):
                    tile_boxes.append(new_box)

            img_name = f"{name}_{idx}.jpg"
            lbl_name = f"{name}_{idx}.txt"

            cv2.imwrite(os.path.join(OUTPUT_IMAGES, img_name), tile)
            save_labels(os.path.join(OUTPUT_LABELS, lbl_name), tile_boxes)

            idx += 1


for img_name in os.listdir(INPUT_IMAGES):
    if not img_name.endswith(".jpg"):
        continue

    img_path = os.path.join(INPUT_IMAGES, img_name)
    label_path = os.path.join(INPUT_LABELS, img_name.replace(".jpg", ".txt"))

    img = cv2.imread(img_path)
    boxes = load_labels(label_path)

    slice_image(img, boxes, img_name.replace(".jpg", ""))

print("SLICING GOTOWY (POPRAWNY)")