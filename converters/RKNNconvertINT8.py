from ultralytics import YOLO

# 1. Switch to a smaller model if maximum FPS is your goal (Highly Recommended)

model = YOLO("yolo11l.pt") 

# 2. Export with INT8 quantization enabled
model.export(
    format="rknn", 
    name="rk3588", 
    imgsz=640,
    quantize=8,          # Forces INT8 quantization instead of FP16 (Crucial for FPS)
    data="coco8.yaml"    # Required: Passes a small dataset for calibration
)
