"""
Fine-tune YOLOv8x on football drone dataset.
Usage: python scripts/train/train_detection.py
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 detection model")
    parser.add_argument("--data", default="configs/data.yaml", help="Dataset YAML path")
    parser.add_argument("--model", default="yolov8x.pt", help="Base model")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="hoeherr_v1")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
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
        name=args.name,
        save=True,
        save_period=10,
        plots=True,
        val=True,
    )

    # Evaluate
    best_model = YOLO(f"runs/detect/{args.name}/weights/best.pt")
    val_results = best_model.val(data=args.data, split="test", imgsz=args.imgsz, batch=args.batch, device=args.device)
    print(f"Fine-tuned mAP50: {val_results.box.map50:.4f}")
    print(f"Fine-tuned mAP50-95: {val_results.box.map:.4f}")


if __name__ == "__main__":
    main()
