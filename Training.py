from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="dataset.yaml",
        epochs=300,
        imgsz=960,
        batch=6,
        device=0,
        workers=2,
        mosaic=1.0,
        close_mosaic=10,
        lr0=0.005,
        patience=80,
        optimizer="AdamW"
    )

if __name__ == "__main__":
    main()