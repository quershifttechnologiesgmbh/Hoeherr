"""
Player detection module using YOLOv8x fine-tuned on drone football footage.
Supports standard inference and SAHI (Sliced Aided Hyper Inference) for 4K frames.
"""
import cv2
import numpy as np
from pathlib import Path


def load_model(model_path: str, device: str = "cpu"):
    """Load a YOLOv8 model for inference."""
    from ultralytics import YOLO
    model = YOLO(model_path)
    return model


def detect_frame(model, frame: np.ndarray, conf: float = 0.25, iou: float = 0.45, imgsz: int = 640):
    """Run detection on a single frame."""
    results = model(frame, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
    detections = []
    if results.boxes is not None and len(results.boxes) > 0:
        for box, cls, score in zip(
            results.boxes.xyxy.cpu().numpy(),
            results.boxes.cls.cpu().numpy(),
            results.boxes.conf.cpu().numpy()
        ):
            detections.append({
                'bbox': [float(x) for x in box],
                'class': int(cls),
                'confidence': float(score),
                'center': [float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)]
            })
    return detections


def detect_with_sahi(model_path: str, image_path: str, slice_size: int = 640, overlap: float = 0.2, conf: float = 0.25, device: str = "cpu"):
    """
    SAHI sliced inference for small objects in high-res frames.
    Players at 50m drone height are only ~15-40px — SAHI slices the image
    into overlapping patches for much better detection.
    """
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=model_path,
        confidence_threshold=conf,
        device=device,
    )

    result = get_sliced_prediction(
        image=image_path,
        detection_model=detection_model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        postprocess_type="NMS",
        postprocess_match_threshold=0.5,
    )
    return result


def evaluate_baseline(data_yaml: str, model_name: str = "yolov8x.pt", device: str = "cpu"):
    """Evaluate pretrained YOLOv8x baseline on our test set."""
    from ultralytics import YOLO
    model = YOLO(model_name)
    results = model.val(data=data_yaml, split="test", imgsz=640, batch=16, device=device, verbose=True)
    return {
        "map50": float(results.box.map50),
        "map50_95": float(results.box.map),
    }
