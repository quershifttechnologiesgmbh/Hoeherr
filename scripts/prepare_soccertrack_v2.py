#!/usr/bin/env python3
"""
Prepare SoccerTrack top-view dataset for YOLOv8 training (v2 with goalkeepers).

Differences from v1 (prepare_soccertrack.py):
- 3 classes: player (0), goalkeeper (1), ball (2)
- Goalkeepers auto-labeled using homography-based pitch position
- SoccerTrack top-view only (no TeamTrack — different perspective hurts)
- Higher output resolution: 1920x1080

Usage:
    python scripts/prepare_soccertrack_v2.py
"""

import ast
import csv
import json
import math
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "soccertrack" / "top_view"
ANNOTATIONS_DIR = RAW_DIR / "annotations"
VIDEOS_DIR = RAW_DIR / "videos"
KEYPOINTS_PATH = BASE_DIR / "data" / "raw" / "soccertrack" / "drone_keypoints.json"
OUTPUT_DIR = BASE_DIR / "data" / "processed_v2"

IMG_WIDTH = 3840
IMG_HEIGHT = 2160

# Higher resolution output for better small-object detection
OUT_WIDTH = 1920
OUT_HEIGHT = 1080
JPEG_QUALITY = 85

FRAME_STEP = 5  # extract every 5th frame

SPLIT_RATIOS = (0.8, 0.1, 0.1)  # train / val / test
RANDOM_SEED = 42

# 3-class mapping: player=0, goalkeeper=1, ball=2
CLASS_NAMES = {0: "player", 1: "goalkeeper", 2: "ball"}


# ---------------------------------------------------------------------------
# Homography helpers
# ---------------------------------------------------------------------------

def load_homography(keypoints_path):
    """Load pre-annotated keypoints and compute pixel->pitch homography."""
    with open(keypoints_path) as f:
        kp_data = json.load(f)

    pixel_pts = []
    pitch_pts = []
    for pitch_str, pixel_coord in kp_data.items():
        pitch_xy = ast.literal_eval(pitch_str)
        pitch_pts.append(list(pitch_xy))
        pixel_pts.append(pixel_coord)

    pixel_pts = np.array(pixel_pts, dtype=np.float32)
    pitch_pts = np.array(pitch_pts, dtype=np.float32)

    H, mask = cv2.findHomography(pixel_pts, pitch_pts, cv2.RANSAC, 5.0)
    n_inliers = int(mask.sum()) if mask is not None else 0
    print(f"  Homography: {n_inliers}/{len(pixel_pts)} inliers")
    return H


def pixel_to_pitch(H, px, py):
    """Convert pixel coords to pitch coords using homography H."""
    pt = np.array([[[px, py]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(pt, H)
    return float(transformed[0, 0, 0]), float(transformed[0, 0, 1])


# ---------------------------------------------------------------------------
# CSV parsing helpers
# ---------------------------------------------------------------------------

def parse_csv_header(csv_path):
    """Parse the wide-format SoccerTrack CSV header (rows 0-3).

    Returns a list of dicts, one per entity (player/ball), each containing:
        - team_id: str  ("0", "1", or "BALL")
        - player_id: str
        - col_indices: dict mapping attribute name -> column index
    """
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        row_team = next(reader)
        row_player = next(reader)
        row_attrs = next(reader)
        _row_frame = next(reader)

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


def identify_goalkeepers(csv_path, entities, H):
    """Identify goalkeepers by computing avg pitch-x position per player.

    Per team, the player whose average position is closest to the goal line
    they are defending is labeled as the goalkeeper.

    Returns a set of (team_id, player_id) tuples for goalkeepers.
    """
    # Accumulate bbox center positions per (team_id, player_id)
    player_positions = defaultdict(list)  # (team_id, player_id) -> [pitch_x, ...]

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for _ in range(4):
            next(reader)

        for row in reader:
            if not row or not row[0].strip():
                continue
            try:
                int(row[0])
            except ValueError:
                continue

            for entity in entities:
                team_id = entity["team_id"]
                player_id = entity["player_id"]
                if team_id == "BALL":
                    continue

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

                # Compute center in pixel coords, then to pitch
                cx_px = bb_left + bb_width / 2.0
                cy_px = bb_top + bb_height / 2.0
                pitch_x, _ = pixel_to_pitch(H, cx_px, cy_px)
                player_positions[(team_id, player_id)].append(pitch_x)

    # Per team, find the goalkeeper
    goalkeepers = set()
    for team_id_str in ["0", "1"]:
        team_players = []
        for (tid, pid), x_vals in player_positions.items():
            if tid == team_id_str and len(x_vals) > 10:
                avg_x = np.mean(x_vals)
                team_players.append((pid, avg_x))

        if not team_players:
            continue

        # Determine which goal this team defends
        team_avg_x = np.mean([x for _, x in team_players])
        if team_avg_x < 52.5:
            # Team centroid closer to x=0: GK is the one with min x
            gk_pid = min(team_players, key=lambda p: p[1])[0]
        else:
            # Team centroid closer to x=105: GK is the one with max x
            gk_pid = max(team_players, key=lambda p: p[1])[0]

        goalkeepers.add((team_id_str, gk_pid))
        gk_avg_x = dict(team_players)[gk_pid]
        print(f"    Team {team_id_str}: goalkeeper = Player {gk_pid} "
              f"(avg x = {gk_avg_x:.1f})")

    return goalkeepers


def parse_csv_data(csv_path, entities, goalkeepers):
    """Parse the data rows (row 4+) of the CSV.

    Returns a dict: frame_number -> list of (class_id, bb_left, bb_top, bb_width, bb_height)
    Only frames where (frame_number - 1) % FRAME_STEP == 0 are kept.

    class_id mapping:
        - 0: player (team 0 or 1, non-goalkeeper)
        - 1: goalkeeper
        - 2: ball
    """
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
                player_id = entity["player_id"]
                cols = entity["col_indices"]

                # Determine class
                if team_id == "BALL":
                    class_id = 2  # ball
                elif (team_id, player_id) in goalkeepers:
                    class_id = 1  # goalkeeper
                elif team_id in ("0", "1"):
                    class_id = 0  # player
                else:
                    continue

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
    """Extract specific frames from a video file. Returns dict frame_num -> image_path."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  WARNING: Cannot open video {video_path}")
        return {}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_set = set(frame_numbers)
    extracted = {}

    for target_frame in sorted(frame_set):
        video_frame_idx = target_frame - 1  # CSV 1-indexed -> video 0-indexed
        if video_frame_idx < 0 or video_frame_idx >= total_frames:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (OUT_WIDTH, OUT_HEIGHT),
                           interpolation=cv2.INTER_AREA)

        img_name = f"{clip_name}_frame_{target_frame:06d}.jpg"
        img_path = output_dir / img_name
        cv2.imwrite(str(img_path), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        extracted[target_frame] = img_path

    cap.release()
    return extracted


def write_yolo_labels(frame_annotations, extracted_frames, labels_dir, clip_name):
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


def create_splits(all_samples, split_ratios, seed):
    """Split samples into train/val/test."""
    random.seed(seed)
    shuffled = list(all_samples)
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * split_ratios[0])
    val_end = train_end + int(n * split_ratios[1])

    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def generate_data_yaml(output_dir):
    """Generate data.yaml for YOLOv8 training."""
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
    print("SoccerTrack Data Preparation v2 (with Goalkeeper Labels)")
    print("=" * 70)
    print(f"Annotations dir : {ANNOTATIONS_DIR}")
    print(f"Videos dir      : {VIDEOS_DIR}")
    print(f"Keypoints       : {KEYPOINTS_PATH}")
    print(f"Output dir      : {OUTPUT_DIR}")
    print(f"Frame step      : every {FRAME_STEP}th frame")
    print(f"Source resolution: {IMG_WIDTH}x{IMG_HEIGHT}")
    print(f"Output resolution: {OUT_WIDTH}x{OUT_HEIGHT}  (JPEG quality={JPEG_QUALITY})")
    print(f"Split ratios    : train={SPLIT_RATIOS[0]}, val={SPLIT_RATIOS[1]}, "
          f"test={SPLIT_RATIOS[2]}")
    print(f"Classes         : {CLASS_NAMES}")
    print()

    # Load homography for goalkeeper identification
    print("Loading homography for goalkeeper detection...")
    H = load_homography(KEYPOINTS_PATH)
    print()

    # Create temporary staging dirs
    staging_images = OUTPUT_DIR / "_staging" / "images"
    staging_labels = OUTPUT_DIR / "_staging" / "labels"
    staging_images.mkdir(parents=True, exist_ok=True)
    staging_labels.mkdir(parents=True, exist_ok=True)

    # Find all CSV annotation files (SoccerTrack top-view only)
    csv_files = sorted(ANNOTATIONS_DIR.glob("*.csv"))
    print(f"Found {len(csv_files)} annotation CSV files.")
    print()

    all_samples = []
    class_counts = defaultdict(int)  # class_id -> count

    for i, csv_path in enumerate(csv_files):
        clip_name = csv_path.stem
        video_path = VIDEOS_DIR / f"{clip_name}.mp4"

        print(f"[{i+1}/{len(csv_files)}] Processing {clip_name} ...")

        if not video_path.exists():
            print(f"  WARNING: Video not found at {video_path}, skipping.")
            continue

        # 1. Parse CSV header
        entities = parse_csv_header(csv_path)
        print(f"  Entities: {len(entities)} (players + ball)")

        # 2. Identify goalkeepers using homography
        goalkeepers = identify_goalkeepers(csv_path, entities, H)

        # 3. Parse CSV data with goalkeeper labels
        frame_annotations = parse_csv_data(csv_path, entities, goalkeepers)
        print(f"  Frames with annotations: {len(frame_annotations)}")

        if not frame_annotations:
            print("  No frames to process, skipping.")
            continue

        # 4. Extract frames from video
        extracted = extract_frames(video_path, frame_annotations.keys(),
                                   staging_images, clip_name)
        print(f"  Frames extracted: {len(extracted)}")

        # 5. Write YOLO labels
        labels_written = write_yolo_labels(frame_annotations, extracted,
                                           staging_labels, clip_name)
        print(f"  Label files written: {labels_written}")

        # Collect samples and stats
        for frame_num in extracted:
            img_path = extracted[frame_num]
            lbl_path = staging_labels / (img_path.stem + ".txt")
            if lbl_path.exists():
                all_samples.append((img_path, lbl_path))
                for det in frame_annotations.get(frame_num, []):
                    class_counts[det[0]] += 1

    print()
    print(f"Total samples collected: {len(all_samples)}")
    total_detections = sum(class_counts.values())
    print(f"Total detections: {total_detections}")
    for cls_id in sorted(class_counts.keys()):
        print(f"  {CLASS_NAMES[cls_id]:12s}: {class_counts[cls_id]}")
    print()

    if not all_samples:
        print("ERROR: No samples were generated. Exiting.")
        return

    # 6. Create train/val/test splits
    print("Creating train/val/test splits ...")
    splits = create_splits(all_samples, SPLIT_RATIOS, RANDOM_SEED)

    for split_name, samples in splits.items():
        img_dir = OUTPUT_DIR / "images" / split_name
        lbl_dir = OUTPUT_DIR / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_src, lbl_src in samples:
            shutil.move(str(img_src), str(img_dir / img_src.name))
            shutil.move(str(lbl_src), str(lbl_dir / lbl_src.name))

        print(f"  {split_name:5s}: {len(samples)} images")

    # 7. Clean up staging directory
    shutil.rmtree(OUTPUT_DIR / "_staging", ignore_errors=True)

    # 8. Generate data.yaml
    yaml_path = generate_data_yaml(OUTPUT_DIR)
    print(f"\ndata.yaml written to: {yaml_path}")

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Dataset directory : {OUTPUT_DIR}")
    print(f"Total images      : {len(all_samples)}")
    print(f"  Train           : {len(splits['train'])}")
    print(f"  Val             : {len(splits['val'])}")
    print(f"  Test            : {len(splits['test'])}")
    print(f"Total detections  : {total_detections}")
    for cls_id in sorted(class_counts.keys()):
        print(f"  {CLASS_NAMES[cls_id]:12s}: {class_counts[cls_id]}")
    print(f"Avg detections/img: {total_detections / len(all_samples):.1f}")
    print(f"Classes           : {CLASS_NAMES}")
    print(f"data.yaml         : {yaml_path}")
    print("=" * 70)
    print("Done!")


if __name__ == "__main__":
    main()
