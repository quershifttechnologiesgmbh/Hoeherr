#!/usr/bin/env python3
"""
Prepare TeamTrack soccer_side dataset for YOLOv8 training.

Reads the wide-format CSV annotations (identical format to SoccerTrack),
extracts every 5th frame from corresponding videos, converts bounding boxes
to YOLO format, and MERGES into the existing data/processed/ directory
alongside SoccerTrack data.

TeamTrack soccer_side:
    Resolution: 6500x1000 (panoramic side-view)
    Frame rate: 25 FPS
    Classes:  0=player, 1=ball

Usage:
    python scripts/prepare_teamtrack.py
"""

import os
import csv
import math
import random
import shutil
from pathlib import Path

import cv2

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "teamtrack" / "teamtrack" / "teamtrack" / "soccer_side"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

IMG_WIDTH = 6500
IMG_HEIGHT = 1000

# Output image size -- resize for disk efficiency; YOLO labels use normalised
# coordinates so they remain valid regardless of output resolution.
OUT_WIDTH = 1280
OUT_HEIGHT = 197  # preserve 6500:1000 aspect ratio → 1280 * (1000/6500) ≈ 197
JPEG_QUALITY = 85

FRAME_STEP = 5  # extract every 5th frame
RANDOM_SEED = 42

CLASS_MAP = {
    "0": 0,      # team 0 -> player
    "1": 0,      # team 1 -> player
    "BALL": 1,   # ball
}

CLASS_NAMES = {0: "player", 1: "ball"}

PREFIX = "tt_"  # prefix to avoid filename collisions with SoccerTrack


# ---------------------------------------------------------------------------
# Helpers (identical logic to prepare_soccertrack.py)
# ---------------------------------------------------------------------------

def parse_csv_header(csv_path):
    """Parse the wide-format TeamTrack CSV header (rows 0-3).

    Returns a list of dicts, one per entity (player/ball), each containing:
        - team_id: str  ("0", "1", or "BALL")
        - player_id: str
        - col_indices: dict mapping attribute name -> column index
    """
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        row_team = next(reader)       # Row 0: TeamID header
        row_player = next(reader)     # Row 1: PlayerID header
        row_attrs = next(reader)      # Row 2: Attributes header
        _row_frame = next(reader)     # Row 3: "frame" (empty cols)

    entities = {}
    for col_idx in range(1, len(row_team)):
        team_id = row_team[col_idx].strip()
        player_id = row_player[col_idx].strip()
        attr = row_attrs[col_idx].strip()

        if not team_id and not player_id:
            continue

        key = (team_id, player_id)
        if key not in entities:
            entities[key] = {
                "team_id": team_id,
                "player_id": player_id,
                "col_indices": {},
            }
        entities[key]["col_indices"][attr] = col_idx

    return list(entities.values())


def parse_csv_data(csv_path, entities):
    """Parse data rows (row 4+). Returns frame_num -> list of detections."""
    frame_annotations = {}

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for _ in range(4):
            next(reader)

        for row in reader:
            if not row or not row[0].strip():
                continue
            try:
                frame_num = int(row[0])
            except ValueError:
                continue

            if (frame_num - 1) % FRAME_STEP != 0:
                continue

            detections = []
            for entity in entities:
                team_id = entity["team_id"]
                if team_id not in CLASS_MAP:
                    continue
                class_id = CLASS_MAP[team_id]
                cols = entity["col_indices"]

                required = ["bb_left", "bb_top", "bb_width", "bb_height"]
                if not all(attr in cols for attr in required):
                    continue

                try:
                    bb_left = float(row[cols["bb_left"]])
                    bb_top = float(row[cols["bb_top"]])
                    bb_width = float(row[cols["bb_width"]])
                    bb_height = float(row[cols["bb_height"]])
                except (ValueError, IndexError):
                    continue

                if any(math.isnan(v) for v in [bb_left, bb_top, bb_width, bb_height]):
                    continue
                if bb_width <= 0 or bb_height <= 0:
                    continue

                detections.append((class_id, bb_left, bb_top, bb_width, bb_height))

            frame_annotations[frame_num] = detections

    return frame_annotations


def bbox_to_yolo(bb_left, bb_top, bb_width, bb_height, img_w, img_h):
    """Convert (left, top, width, height) to YOLO (cx, cy, w, h) normalized."""
    cx = (bb_left + bb_width / 2.0) / img_w
    cy = (bb_top + bb_height / 2.0) / img_h
    w = bb_width / img_w
    h = bb_height / img_h

    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    return cx, cy, w, h


def extract_frames(video_path, frame_numbers, output_dir, clip_name):
    """Extract specific frames from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  WARNING: Cannot open video {video_path}")
        return {}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_set = set(frame_numbers)
    extracted = {}

    for target_frame in sorted(frame_set):
        video_frame_idx = target_frame - 1
        if video_frame_idx < 0 or video_frame_idx >= total_frames:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (OUT_WIDTH, OUT_HEIGHT),
                           interpolation=cv2.INTER_AREA)

        img_name = f"{PREFIX}{clip_name}_frame_{target_frame:06d}.jpg"
        img_path = output_dir / img_name
        cv2.imwrite(str(img_path), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        extracted[target_frame] = img_path

    cap.release()
    return extracted


def write_yolo_labels(frame_annotations, extracted_frames, labels_dir):
    """Write YOLO-format label files for extracted frames."""
    written = 0
    for frame_num, detections in frame_annotations.items():
        if frame_num not in extracted_frames:
            continue

        img_path = extracted_frames[frame_num]
        label_name = img_path.stem + ".txt"
        label_path = labels_dir / label_name

        with open(label_path, "w") as f:
            for class_id, bb_left, bb_top, bb_width, bb_height in detections:
                cx, cy, w, h = bbox_to_yolo(bb_left, bb_top, bb_width, bb_height,
                                             IMG_WIDTH, IMG_HEIGHT)
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        written += 1

    return written


def update_data_yaml(output_dir):
    """Update data.yaml to reflect current dataset."""
    yaml_content = f"""path: {output_dir}
train: images/train
val: images/val
test: images/test

nc: {len(CLASS_NAMES)}
names:
"""
    for cls_id in sorted(CLASS_NAMES.keys()):
        yaml_content += f"  {cls_id}: {CLASS_NAMES[cls_id]}\n"

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    return yaml_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("TeamTrack Soccer Side Data Preparation for YOLOv8")
    print("=" * 70)

    # Count existing SoccerTrack samples
    existing_train = len(list((OUTPUT_DIR / "images" / "train").glob("*.jpg")))
    existing_val = len(list((OUTPUT_DIR / "images" / "val").glob("*.jpg")))
    existing_test = len(list((OUTPUT_DIR / "images" / "test").glob("*.jpg")))
    print(f"Existing processed data: train={existing_train}, val={existing_val}, test={existing_test}")
    print()

    splits = {"train": [], "val": [], "test": []}
    total_detections = 0
    player_count = 0
    ball_count = 0

    for split_name in ["train", "val", "test"]:
        split_dir = RAW_DIR / split_name
        ann_dir = split_dir / "annotations"
        vid_dir = split_dir / "videos"

        if not ann_dir.exists():
            print(f"  WARNING: {ann_dir} does not exist, skipping.")
            continue

        csv_files = sorted(ann_dir.glob("*.csv"))
        print(f"[{split_name}] Found {len(csv_files)} annotation files")

        # Output directly into the correct split directory
        img_out = OUTPUT_DIR / "images" / split_name
        lbl_out = OUTPUT_DIR / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        split_images = 0
        split_dets = 0

        for i, csv_path in enumerate(csv_files):
            clip_name = csv_path.stem
            video_path = vid_dir / f"{clip_name}.mp4"

            if not video_path.exists():
                print(f"  [{i+1}/{len(csv_files)}] Video not found for {clip_name}, skipping.")
                continue

            # 1. Parse CSV header
            entities = parse_csv_header(csv_path)

            # 2. Parse data rows
            frame_annotations = parse_csv_data(csv_path, entities)

            if not frame_annotations:
                continue

            # 3. Extract frames
            extracted = extract_frames(video_path, frame_annotations.keys(),
                                       img_out, clip_name)

            # 4. Write labels
            labels_written = write_yolo_labels(frame_annotations, extracted, lbl_out)

            split_images += len(extracted)
            for frame_num in extracted:
                for det in frame_annotations.get(frame_num, []):
                    split_dets += 1
                    total_detections += 1
                    if det[0] == 0:
                        player_count += 1
                    elif det[0] == 1:
                        ball_count += 1

            if (i + 1) % 10 == 0 or (i + 1) == len(csv_files):
                print(f"  [{i+1}/{len(csv_files)}] Processed {clip_name} "
                      f"({len(extracted)} frames, {labels_written} labels)")

        print(f"  {split_name}: {split_images} new images, {split_dets} detections")
        print()

    # Update data.yaml
    yaml_path = update_data_yaml(OUTPUT_DIR)
    print(f"data.yaml updated: {yaml_path}")

    # Final counts
    final_train = len(list((OUTPUT_DIR / "images" / "train").glob("*.jpg")))
    final_val = len(list((OUTPUT_DIR / "images" / "val").glob("*.jpg")))
    final_test = len(list((OUTPUT_DIR / "images" / "test").glob("*.jpg")))

    print()
    print("=" * 70)
    print("SUMMARY (Combined SoccerTrack + TeamTrack)")
    print("=" * 70)
    print(f"Total images      : {final_train + final_val + final_test}")
    print(f"  Train           : {final_train}")
    print(f"  Val             : {final_val}")
    print(f"  Test            : {final_test}")
    print(f"TeamTrack added   : {total_detections} detections "
          f"(players: {player_count}, balls: {ball_count})")
    print(f"data.yaml         : {yaml_path}")
    print("=" * 70)
    print("Done!")


if __name__ == "__main__":
    main()
