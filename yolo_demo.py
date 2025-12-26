from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model("https://ultralytics.com/images/bus.jpg")

for result in results:
    boxes = result.boxes
    print("检测到的物体：", boxes.cls)  # 物体类别（数字对应标签，比如0是person，5是bus）
    print("检测框坐标：", boxes.xyxy)  # 检测框的左上角/右下角坐标
    print("置信度：", boxes.conf)  # 检测置信度（越接近1越准确）
    result.save(filename="detected_bus.jpg")