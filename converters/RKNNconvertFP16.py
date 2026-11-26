from ultralytics import YOLO

model = YOLO("yolo11l.pt")
model.export(format="rknn", name="rk3588", imgsz=640)
