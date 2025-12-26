from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("runs/train/fs_powerbank_model8/weights/best.pt")

    results = model.predict(
        source="test.jpg",
        save=True,
        conf=0.5,
        device=0,
    )

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = box.cls  # 类别索引
            conf = box.conf  # 置信度
            xyxy = box.xyxy  # 目标框坐标

            cls_name = model.names[int(cls.item())]
            conf_value = conf.item()
            xyxy_value = xyxy.cpu().numpy()

            print(f"检测到：{cls_name}，置信度：{conf_value:.2f}，坐标：{xyxy_value}")