from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="dataset.yaml",

        imgsz=640,
        batch=4,
        epochs=250,

        # AUGMENTACJA
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        degrees=5.0,
        translate=0.1,
        scale=0.4,
        shear=2.0,

        fliplr=0.5,
        flipud=0.1,

        mosaic=1.0,
        close_mosaic=15,

        mixup=0.1,
        copy_paste=0.05,

        # stabilność
        lr0=0.0025,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,

        patience=60,

        optimizer="AdamW",

        device=0,
        workers=2,

        multi_scale=False,
        cache=False,

        #ważne dla małych obiektów
        box=8.0,
        cls=0.5,
        dfl=1.5
    )

if __name__ == "__main__":
    main()