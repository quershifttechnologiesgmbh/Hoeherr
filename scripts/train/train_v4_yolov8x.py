#!/usr/bin/env python3
"""
v4_yolov8x: YOLOv8x (extra-large) with fully unfrozen backbone on combined data.

GPU 3 | YOLOv8x | freeze=0 | Combined data
Tests whether the larger model capacity improves detection.
"""
import os
import shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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
        freeze=0,
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
        name="hoeherr_v4_yolov8x",
        save=True,
        save_period=10,
        plots=True,
        val=True,
        verbose=True,
        workers=0,
        exist_ok=True,
    )

    print("\n[v4_yolov8x] Training complete!")

    # Evaluate on test set
    best_path = "runs/detect/hoeherr_v4_yolov8x/weights/best.pt"
    if os.path.exists(best_path):
        best_model = YOLO(best_path)
        val_results = best_model.val(
            data="data/processed/data.yaml",
            split="test",
            imgsz=640,
            batch=8,
            device=0,
            verbose=True,
        )
        print(f"[v4_yolov8x] Test mAP50: {val_results.box.map50:.4f}")
        print(f"[v4_yolov8x] Test mAP50-95: {val_results.box.map:.4f}")

        # Copy best model
        dst = "models/detection/best_v4_yolov8x.pt"
        shutil.copy2(best_path, dst)
        print(f"[v4_yolov8x] Best model copied to {dst}")


if __name__ == "__main__":
    main()
