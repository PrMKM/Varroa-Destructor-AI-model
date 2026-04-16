# from ultralytics import YOLO

# def main():
#     model = YOLO("yolov8n.pt")

#     model.train(
#         data="dataset.yaml",
#         imgsz=640,
#         batch=4,

#         epochs=200,

#         mosaic=1.0,
#         close_mosaic=10,
#         mixup=0.1,

#         lr0=0.003,
#         patience=50,

#         device=0,
#         workers=2,

#         optimizer="AdamW",
#         multi_scale=False
#     )

# if __name__ == "__main__":
#     main()

from ultralytics import YOLO

def main():
    #model = YOLO("yolov8n.pt")
    model = YOLO("yolov8s.pt")

    model.train(
        data="dataset.yaml",

        imgsz=640,
        #batch=4,
        batch=2,
        epochs=250,  # lekko zwiększamy

        # AUGMENTACJA (klucz)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,

        fliplr=0.5,
        flipud=0.1,

        mosaic=1.0,
        close_mosaic=15,
        mixup=0.15,

        copy_paste=0.1,

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

        # ważne przy małych obiektach
        box=8.0,
        cls=0.5,
        dfl=1.5
    )

if __name__ == "__main__":
    main()