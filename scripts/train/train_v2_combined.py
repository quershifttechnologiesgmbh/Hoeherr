#!/usr/bin/env python3
"""
v2_combined: YOLOv8m with frozen backbone on combined SoccerTrack+TeamTrack data.

GPU 1 | YOLOv8m | freeze=10 | Combined data
Baseline comparison: same architecture as v1 but with more data.
"""
import os
import shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from ultralytics import YOLO


def main():
    model = YOLO("yolov8m.pt")

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
        name="hoeherr_v2_combined",
        save=True,
        save_period=10,
        plots=True,
        val=True,
        verbose=True,
        workers=0,
        exist_ok=True,
    )

    print("\n[v2_combined] Training complete!")

    # Evaluate on test set
    best_path = "runs/detect/hoeherr_v2_combined/weights/best.pt"
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
        print(f"[v2_combined] Test mAP50: {val_results.box.map50:.4f}")
        print(f"[v2_combined] Test mAP50-95: {val_results.box.map:.4f}")

        # Copy best model
        dst = "models/detection/best_v2_combined.pt"
        shutil.copy2(best_path, dst)
        print(f"[v2_combined] Best model copied to {dst}")


if __name__ == "__main__":
    main()
