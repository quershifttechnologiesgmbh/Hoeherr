#!/usr/bin/env python3
"""Train YOLOv8m on SoccerTrack drone football data.

Handles limited /dev/shm by using workers=0 and careful GPU memory management.
"""
import gc
import os
import shutil

# Force single GPU and memory settings BEFORE importing torch
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch

def main():
    # Aggressively clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        free, total = torch.cuda.mem_get_info(0)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Free: {free/1024**3:.1f} GB / Total: {total/1024**3:.1f} GB")

    from ultralytics import YOLO

    model = YOLO("yolov8m.pt")

    results = model.train(
        data="data/processed/data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,           # Reduced from 16 to be safe with orphaned GPU memory
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
        workers=0,          # Avoid /dev/shm shared memory issues
        exist_ok=True,
    )

    print("\nTraining complete!")

    # Copy best model to deployment location
    best_src = "runs/detect/hoeherr_v1/weights/best.pt"
    best_dst = "models/detection/best.pt"
    os.makedirs(os.path.dirname(best_dst), exist_ok=True)
    if os.path.exists(best_src):
        shutil.copy2(best_src, best_dst)
        print(f"Best model copied to {best_dst}")
    else:
        print(f"Warning: {best_src} not found")

if __name__ == "__main__":
    main()
