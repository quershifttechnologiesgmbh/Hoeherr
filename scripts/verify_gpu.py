"""Verify GPU setup and model loading."""
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"VRAM: {props.total_mem / 1e9:.1f} GB")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("No GPU detected - will use CPU (slower)")

try:
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    print("YOLOv8 loaded successfully")
except Exception as e:
    print(f"YOLOv8 load failed: {e}")
