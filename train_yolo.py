from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")

    results = model.train(
        data="fs_powerbank.yaml",
        epochs=100,
        batch=8,
        imgsz=640,
        device=0,
        patience=20,
        save=True,
        project="runs/train",
        name="fs_powerbank_model",
        workers=0
    )

    metrics = model.val()
    print("验证集评估结果：")
    print(f"mAP50：{metrics.box.map50:.4f}")

