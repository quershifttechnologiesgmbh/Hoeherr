#!/usr/bin/env python3
"""Train YOLOv8x on SoccerTrack drone football data."""
import os
import shutil
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

from ultralytics import YOLO

def main():
    model = YOLO("yolov8x.pt")

    results = model.train(
        data="data/processed/data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        patience=10,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        freeze=10,
        cos_lr=True,
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        degrees=180,
        scale=0.5,
        fliplr=0.5,
        flipud=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        project="runs/detect",
        name="hoeherr_v1",
        save=True,
        save_period=10,
        plots=True,
        val=True,
        verbose=True,
        workers=4,
        exist_ok=True,
    )

    print(f"\nTraining complete!")

    # Copy best model
    best_src = "runs/detect/hoeherr_v1/weights/best.pt"
    best_dst = "models/detection/best.pt"
    if os.path.exists(best_src):
        shutil.copy2(best_src, best_dst)
        print(f"Best model copied to {best_dst}")

if __name__ == "__main__":
    main()
