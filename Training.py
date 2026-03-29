from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="dataset.yaml",
        epochs=200,
        imgsz=1280,
        batch=4,
        device=0,
        workers=2
    )

if __name__ == "__main__":
    main()