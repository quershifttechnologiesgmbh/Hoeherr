#!/usr/bin/env python3
"""
Evaluate SAHI (Sliced Aided Hyper Inference) for ball detection improvement.

Compares standard YOLOv8 inference vs SAHI sliced inference on the validation
set, specifically measuring ball class mAP50.

Usage:
    python scripts/evaluate_sahi.py [--model MODEL_PATH] [--device GPU_ID]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent


def load_ground_truth(labels_dir):
    """Load YOLO-format ground truth labels.
    Returns dict: image_stem -> list of (class_id, cx, cy, w, h)
    """
    gt = {}
    for lbl_path in sorted(labels_dir.glob("*.txt")):
        boxes = []
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    boxes.append((cls, cx, cy, w, h))
        gt[lbl_path.stem] = boxes
    return gt


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    """Convert YOLO normalized coords to pixel xyxy."""
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return x1, y1, x2, y2


def compute_iou(box1, box2):
    """Compute IoU between two xyxy boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


def compute_ap50(predictions, ground_truths, target_class, iou_threshold=0.5):
    """Compute AP@50 for a specific class.

    predictions: list of (image_stem, class_id, confidence, x1, y1, x2, y2)
    ground_truths: dict image_stem -> list of (class_id, x1, y1, x2, y2)
    """
    # Filter to target class
    preds = [(s, c, conf, x1, y1, x2, y2)
             for s, c, conf, x1, y1, x2, y2 in predictions if c == target_class]
    preds.sort(key=lambda x: x[2], reverse=True)  # sort by confidence

    # Count total GT boxes for target class
    n_gt = 0
    gt_matched = defaultdict(lambda: defaultdict(bool))  # image -> gt_idx -> matched
    gt_boxes = {}
    for stem, boxes in ground_truths.items():
        cls_boxes = [(x1, y1, x2, y2) for cls, x1, y1, x2, y2 in boxes if cls == target_class]
        gt_boxes[stem] = cls_boxes
        n_gt += len(cls_boxes)

    if n_gt == 0:
        return 0.0, 0, 0

    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))

    for i, (stem, cls, conf, px1, py1, px2, py2) in enumerate(preds):
        if stem not in gt_boxes or len(gt_boxes[stem]) == 0:
            fp[i] = 1
            continue

        best_iou = 0
        best_j = -1
        for j, (gx1, gy1, gx2, gy2) in enumerate(gt_boxes[stem]):
            iou = compute_iou((px1, py1, px2, py2), (gx1, gy1, gx2, gy2))
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_iou >= iou_threshold and not gt_matched[stem][best_j]:
            tp[i] = 1
            gt_matched[stem][best_j] = True
        else:
            fp[i] = 1

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / n_gt
    precisions = cum_tp / (cum_tp + cum_fp)

    # AP using all-point interpolation
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])

    return float(ap), int(cum_tp[-1]) if len(cum_tp) > 0 else 0, n_gt


def run_standard_inference(model, images_dir, conf=0.25, imgsz=640):
    """Run standard YOLOv8 inference on all images."""
    predictions = []
    image_files = sorted(images_dir.glob("*.jpg"))

    for img_path in image_files:
        results = model(str(img_path), conf=conf, iou=0.45, imgsz=imgsz, verbose=False)[0]
        if results.boxes is not None and len(results.boxes) > 0:
            img_h, img_w = results.orig_shape
            for box, cls, score in zip(
                results.boxes.xyxy.cpu().numpy(),
                results.boxes.cls.cpu().numpy(),
                results.boxes.conf.cpu().numpy()
            ):
                predictions.append((
                    img_path.stem, int(cls), float(score),
                    float(box[0]), float(box[1]), float(box[2]), float(box[3])
                ))

    return predictions


def run_sahi_inference(model_path, images_dir, device, conf=0.25,
                       slice_size=640, overlap=0.2):
    """Run SAHI sliced inference on all images."""
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=str(model_path),
        confidence_threshold=conf,
        device=device,
    )

    predictions = []
    image_files = sorted(images_dir.glob("*.jpg"))

    for img_path in image_files:
        result = get_sliced_prediction(
            image=str(img_path),
            detection_model=detection_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
            postprocess_type="NMS",
            postprocess_match_threshold=0.5,
            verbose=0,
        )

        for pred in result.object_prediction_list:
            bbox = pred.bbox
            predictions.append((
                img_path.stem, pred.category.id, pred.score.value,
                float(bbox.minx), float(bbox.miny),
                float(bbox.maxx), float(bbox.maxy)
            ))

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAHI ball detection")
    parser.add_argument("--model", default=None,
                        help="Model path (default: best available)")
    parser.add_argument("--device", default="0", help="GPU device")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--slice-size", type=int, default=640, help="SAHI slice size")
    parser.add_argument("--overlap", type=float, default=0.2, help="SAHI overlap ratio")
    args = parser.parse_args()

    # Find best model
    if args.model:
        model_path = Path(args.model)
    else:
        candidates = [
            BASE_DIR / "models" / "detection" / "best.pt",
            BASE_DIR / "runs" / "detect" / "hoeherr_v1" / "weights" / "best.pt",
        ]
        model_path = None
        for c in candidates:
            if c.exists():
                model_path = c
                break
        if model_path is None:
            print("ERROR: No model found. Specify --model path.")
            sys.exit(1)

    print(f"Model: {model_path}")
    print(f"Device: {args.device}")

    # Load data config
    data_yaml = BASE_DIR / "data" / "processed" / "data.yaml"
    with open(data_yaml) as f:
        data_cfg = yaml.safe_load(f)

    data_root = Path(data_cfg["path"])
    val_images = data_root / data_cfg["val"]
    val_labels = data_root / "labels" / "val"

    print(f"Val images: {val_images}")
    print(f"Val labels: {val_labels}")

    # Load ground truth
    gt_raw = load_ground_truth(val_labels)

    # Convert GT to pixel coordinates for matching
    gt_pixel = {}
    for stem, boxes in gt_raw.items():
        img_path = val_images / f"{stem}.jpg"
        if img_path.exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                img_h, img_w = img.shape[:2]
                pixel_boxes = []
                for cls, cx, cy, w, h in boxes:
                    x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, w, h, img_w, img_h)
                    pixel_boxes.append((cls, x1, y1, x2, y2))
                gt_pixel[stem] = pixel_boxes

    n_val = len(gt_pixel)
    n_ball_gt = sum(1 for boxes in gt_pixel.values() for cls, *_ in boxes if cls == 1)
    n_player_gt = sum(1 for boxes in gt_pixel.values() for cls, *_ in boxes if cls == 0)
    print(f"\nGround truth: {n_val} images, {n_player_gt} players, {n_ball_gt} balls")

    # --- Standard inference ---
    print("\n" + "=" * 50)
    print("Running STANDARD inference ...")
    print("=" * 50)

    from ultralytics import YOLO
    model = YOLO(str(model_path))

    std_preds = run_standard_inference(model, val_images, conf=args.conf)

    std_ball_ap, std_ball_tp, std_ball_gt = compute_ap50(std_preds, gt_pixel, target_class=1)
    std_player_ap, std_player_tp, std_player_gt = compute_ap50(std_preds, gt_pixel, target_class=0)
    std_map50 = (std_ball_ap + std_player_ap) / 2

    print(f"Standard - Player AP@50: {std_player_ap:.4f} ({std_player_tp}/{std_player_gt} matched)")
    print(f"Standard - Ball   AP@50: {std_ball_ap:.4f} ({std_ball_tp}/{std_ball_gt} matched)")
    print(f"Standard - mAP@50:       {std_map50:.4f}")

    # --- SAHI inference ---
    print("\n" + "=" * 50)
    print(f"Running SAHI inference (slice={args.slice_size}, overlap={args.overlap}) ...")
    print("=" * 50)

    sahi_preds = run_sahi_inference(
        model_path, val_images, args.device,
        conf=args.conf, slice_size=args.slice_size, overlap=args.overlap
    )

    sahi_ball_ap, sahi_ball_tp, sahi_ball_gt = compute_ap50(sahi_preds, gt_pixel, target_class=1)
    sahi_player_ap, sahi_player_tp, sahi_player_gt = compute_ap50(sahi_preds, gt_pixel, target_class=0)
    sahi_map50 = (sahi_ball_ap + sahi_player_ap) / 2

    print(f"SAHI     - Player AP@50: {sahi_player_ap:.4f} ({sahi_player_tp}/{sahi_player_gt} matched)")
    print(f"SAHI     - Ball   AP@50: {sahi_ball_ap:.4f} ({sahi_ball_tp}/{sahi_ball_gt} matched)")
    print(f"SAHI     - mAP@50:       {sahi_map50:.4f}")

    # --- Comparison ---
    print("\n" + "=" * 50)
    print("COMPARISON: Standard vs SAHI")
    print("=" * 50)
    print(f"{'Metric':<20} {'Standard':>10} {'SAHI':>10} {'Delta':>10}")
    print("-" * 50)
    print(f"{'Player AP@50':<20} {std_player_ap:>10.4f} {sahi_player_ap:>10.4f} {sahi_player_ap - std_player_ap:>+10.4f}")
    print(f"{'Ball AP@50':<20} {std_ball_ap:>10.4f} {sahi_ball_ap:>10.4f} {sahi_ball_ap - std_ball_ap:>+10.4f}")
    print(f"{'mAP@50':<20} {std_map50:>10.4f} {sahi_map50:>10.4f} {sahi_map50 - std_map50:>+10.4f}")

    # Save results
    results = {
        "model": str(model_path),
        "val_images": n_val,
        "standard": {
            "player_ap50": std_player_ap,
            "ball_ap50": std_ball_ap,
            "map50": std_map50,
        },
        "sahi": {
            "slice_size": args.slice_size,
            "overlap": args.overlap,
            "player_ap50": sahi_player_ap,
            "ball_ap50": sahi_ball_ap,
            "map50": sahi_map50,
        },
        "improvement": {
            "player_ap50": sahi_player_ap - std_player_ap,
            "ball_ap50": sahi_ball_ap - std_ball_ap,
            "map50": sahi_map50 - std_map50,
        }
    }

    results_path = BASE_DIR / "reports" / "sahi_comparison.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
