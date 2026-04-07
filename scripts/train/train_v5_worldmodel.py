#!/usr/bin/env python3
"""
v5_worldmodel: YOLOv8x with 3 classes (player, goalkeeper, ball) trained at
inference resolution (1280) on SoccerTrack top-view only.

Key differences from v4:
- 3 classes instead of 2 (goalkeeper distinction)
- imgsz=1280 to match inference resolution (was 640)
- epochs=80 for harder task (was 50)
- batch=4 for higher resolution (was 8)
- degrees=15 (was 180) — drone view has consistent orientation
- SoccerTrack only dataset (no TeamTrack) for domain consistency

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts/train/train_v5_worldmodel.py
"""
import os
import shutil

import torch
from ultralytics import YOLO


def main():
    # Resume from last checkpoint if available, otherwise start fresh
    last_ckpt = "runs/detect/runs/detect/hoeherr_v5_worldmodel/weights/last.pt"
    if os.path.exists(last_ckpt):
        print(f"[v5_worldmodel] Resuming from {last_ckpt}")
        model = YOLO(last_ckpt)
        results = model.train(
            resume=True,
        )
    else:
        model = YOLO("yolov8x.pt")
        results = model.train(
            data="data/processed_v2/data.yaml",
            epochs=80,
            imgsz=1280,
            batch=4,
            device=torch.device("cuda:0"),
            patience=15,
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.01,
            warmup_epochs=5,
            freeze=0,
            cos_lr=True,
            augment=True,
            mosaic=1.0,
            mixup=0.1,
            degrees=15,
            scale=0.5,
            fliplr=0.5,
            flipud=0.1,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            project="runs/detect",
            name="hoeherr_v5_worldmodel",
            save=True,
            save_period=-1,
            plots=True,
            val=True,
            verbose=True,
            workers=0,
            exist_ok=True,
        )

    print("\n[v5_worldmodel] Training complete!")

    # Evaluate on test set — check both possible paths
    best_path = "runs/detect/runs/detect/hoeherr_v5_worldmodel/weights/best.pt"
    if not os.path.exists(best_path):
        best_path = "runs/detect/hoeherr_v5_worldmodel/weights/best.pt"
    if os.path.exists(best_path):
        best_model = YOLO(best_path)
        val_results = best_model.val(
            data="data/processed_v2/data.yaml",
            split="test",
            imgsz=1280,
            batch=4,
            device=torch.device("cuda:0"),
            verbose=True,
        )
        print(f"[v5_worldmodel] Test mAP50:    {val_results.box.map50:.4f}")
        print(f"[v5_worldmodel] Test mAP50-95: {val_results.box.map:.4f}")

        # Per-class AP
        for i, name in enumerate(val_results.names.values()):
            if i < len(val_results.box.ap50):
                print(f"[v5_worldmodel]   AP50 {name}: {val_results.box.ap50[i]:.4f}")

        # Copy best model
        dst = "models/detection/best_v5_worldmodel.pt"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(best_path, dst)
        print(f"[v5_worldmodel] Best model copied to {dst}")


if __name__ == "__main__":
    main()
