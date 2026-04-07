#!/usr/bin/env python3
"""
Full analysis pipeline: Detection + BoT-SORT Tracking + Team Classification +
Homography + Annotated Video + Physical/Tactical/Individual Metrics + HTML Report.

World-model constraints (v2):
- Top-K detection filter (max 25 per frame)
- Pitch-boundary filter (drop out-of-field detections)
- Kalman filter smoothing + outlier rejection (replaces simple speed clamping)
- Track consolidation (merge fragmented tracks)
- Role detection (goalkeeper, referee, player, ball)
"""
import argparse
import ast
import base64
import io
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tracking.bot_sort_tracker import run_tracking
from src.tracking.kalman_smoother import smooth_tracks
from src.team_classification.jersey_classifier import TeamClassifier
from src.homography.pitch_detector import PitchDetector
from src.metrics.physical_metrics import PhysicalMetrics
from src.metrics.tactical_metrics import TacticalMetrics
from src.metrics.individual_metrics import IndividualMetrics

# ---------------------------------------------------------------------------
# World Model Constants
# ---------------------------------------------------------------------------
MAX_DETECTIONS_PER_FRAME = 25       # 11+11+1+3 = 26 max real objects
MAX_SPEED_KMH = 40.0               # absolute cap on reported speed
TRACK_MERGE_GAP_FRAMES = 30        # max gap to consider merging two tracks
TRACK_MERGE_DISTANCE_M = 5.0       # max avg-position distance for merge
MIN_TRACK_POSITIONS = 200           # min positions to keep a track

# Pitch boundaries (meters) with margin for measurement noise
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
PITCH_MARGIN = 3.0                  # allow small margin outside lines


def parse_args():
    parser = argparse.ArgumentParser(description="Run full analysis pipeline")
    parser.add_argument("--model", required=True, help="Path to YOLO model")
    parser.add_argument("--videos", nargs="+", required=True, help="Input video paths")
    parser.add_argument("--keypoints", required=True, help="Path to drone keypoints JSON")
    parser.add_argument("--tracker", default="configs/botsort.yaml", help="Tracker config")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--output", default="reports/analysis_output/")
    parser.add_argument("--fps", type=float, default=30.0, help="Video FPS")
    parser.add_argument("--output-scale", type=float, default=0.5)

    # Hybrid pipeline flag
    parser.add_argument("--hybrid", action="store_true",
                        help="Use hybrid YOLO+Gemma4 pipeline (delegates to run_hybrid_analysis)")
    parser.add_argument("--no-gemma4", action="store_true",
                        help="With --hybrid: skip Gemma 4, use YOLO only")
    parser.add_argument("--gemma4-model", default="/home/jovyan/.local/models/gemma4-e4b",
                        help="Gemma 4 model path (for --hybrid)")
    parser.add_argument("--gemma4-device", default="cuda:1",
                        help="Gemma 4 GPU device (for --hybrid)")

    return parser.parse_args()


# ==============================================================================
# World Model Constraints
# ==============================================================================

def apply_topk_filter(all_tracks, max_k=MAX_DETECTIONS_PER_FRAME):
    """Keep only top-K detections per frame by confidence."""
    total_removed = 0
    for i, frame_tracks in enumerate(all_tracks):
        if len(frame_tracks) > max_k:
            frame_tracks.sort(key=lambda t: t['confidence'], reverse=True)
            total_removed += len(frame_tracks) - max_k
            all_tracks[i] = frame_tracks[:max_k]
    if total_removed > 0:
        print(f"  Top-K filter: removed {total_removed} low-confidence detections "
              f"(limit={max_k}/frame)")
    return all_tracks


def apply_pitch_boundary_filter(all_tracks, margin=PITCH_MARGIN):
    """Remove detections whose pitch position is outside the field.

    Drops coaches, cameramen, spectators, and other people detected
    outside the playing area.
    """
    x_min, x_max = -margin, PITCH_LENGTH + margin
    y_min, y_max = -margin, PITCH_WIDTH + margin

    removed = 0
    for i, frame_tracks in enumerate(all_tracks):
        filtered = []
        for t in frame_tracks:
            pos = t.get('pitch_pos')
            if pos is None:
                filtered.append(t)  # keep detections not yet transformed
                continue
            if x_min <= pos[0] <= x_max and y_min <= pos[1] <= y_max:
                filtered.append(t)
            else:
                removed += 1
        all_tracks[i] = filtered

    print(f"  Pitch boundary filter: removed {removed} out-of-field detections "
          f"(margin={margin}m)")
    return all_tracks


def apply_kalman_smoothing(all_tracks, fps):
    """Apply per-track Kalman filter for position smoothing + outlier rejection.

    Replaces simple speed clamping with a proper constant-velocity motion model.
    Outlier measurements (from ID switches, homography noise) are detected via
    Mahalanobis gating and replaced with the filter's prediction.
    """
    all_tracks, stats = smooth_tracks(
        all_tracks,
        fps=fps,
        process_noise_std=0.5,       # moderate acceleration noise
        measurement_noise_std=0.8,   # homography reprojection noise ~0.8m
        gate_threshold=4.0,          # ~99.99% chi2(2) — reject extreme outliers
    )
    print(f"  Kalman smoothing: {stats['total_measurements']} measurements, "
          f"{stats['outliers_rejected']} outliers rejected "
          f"({stats['rejection_rate_pct']:.1f}%), "
          f"{stats['active_filters']} track filters")


def consolidate_tracks(all_tracks, fps,
                       gap_frames=TRACK_MERGE_GAP_FRAMES,
                       merge_dist=TRACK_MERGE_DISTANCE_M,
                       min_positions=MIN_TRACK_POSITIONS):
    """Merge fragmented tracks and filter short ones.

    1. Build per-track stats (first/last frame, avg position, team)
    2. Merge pairs where track A ends within gap_frames of track B starting,
       same team, and avg positions within merge_dist
    3. Filter tracks with fewer than min_positions detections
    """
    # Build per-track stats
    track_stats = {}
    for fi, frame_tracks in enumerate(all_tracks):
        for di, t in enumerate(frame_tracks):
            tid = t['track_id']
            if tid not in track_stats:
                track_stats[tid] = {
                    'first_frame': fi, 'last_frame': fi,
                    'positions': [], 'team': t.get('team'),
                    'count': 0,
                }
            stats = track_stats[tid]
            stats['last_frame'] = fi
            stats['count'] += 1
            if t.get('pitch_pos') is not None:
                stats['positions'].append(t['pitch_pos'])
            if t.get('team') is not None:
                stats['team'] = t['team']

    # Compute avg positions
    for tid, stats in track_stats.items():
        if stats['positions']:
            stats['avg_pos'] = np.mean(stats['positions'], axis=0)
        else:
            stats['avg_pos'] = None

    # Find merge pairs: A ends before B starts, same team, close position
    merge_map = {}  # tid_to_merge -> tid_target
    sorted_tids = sorted(track_stats.keys(),
                         key=lambda t: track_stats[t]['first_frame'])

    for i, tid_a in enumerate(sorted_tids):
        if tid_a in merge_map:
            continue
        sa = track_stats[tid_a]
        if sa['avg_pos'] is None:
            continue
        for j in range(i + 1, len(sorted_tids)):
            tid_b = sorted_tids[j]
            if tid_b in merge_map:
                continue
            sb = track_stats[tid_b]
            if sb['avg_pos'] is None:
                continue
            # B must start after A ends
            if sb['first_frame'] < sa['last_frame']:
                continue
            if sb['first_frame'] - sa['last_frame'] > gap_frames:
                # Sorted by first_frame, so further ones will be even farther
                break
            # Same team check
            if sa['team'] != sb['team']:
                continue
            # Position proximity check
            dist = np.linalg.norm(sa['avg_pos'] - sb['avg_pos'])
            if dist <= merge_dist:
                merge_map[tid_b] = tid_a
                # Update A's stats to cover merged range
                sa['last_frame'] = max(sa['last_frame'], sb['last_frame'])
                sa['count'] += sb['count']
                sa['positions'].extend(sb['positions'])
                if sa['positions']:
                    sa['avg_pos'] = np.mean(sa['positions'], axis=0)

    # Apply merges: rename track IDs
    if merge_map:
        for frame_tracks in all_tracks:
            for t in frame_tracks:
                if t['track_id'] in merge_map:
                    t['track_id'] = merge_map[t['track_id']]

    # Recount after merge
    track_counts = defaultdict(int)
    for frame_tracks in all_tracks:
        for t in frame_tracks:
            if t.get('pitch_pos') is not None:
                track_counts[t['track_id']] += 1

    # Filter: remove short tracks
    short_tids = {tid for tid, cnt in track_counts.items() if cnt < min_positions}
    removed_dets = 0
    for i, frame_tracks in enumerate(all_tracks):
        orig_len = len(frame_tracks)
        all_tracks[i] = [t for t in frame_tracks if t['track_id'] not in short_tids]
        removed_dets += orig_len - len(all_tracks[i])

    remaining_tids = set()
    for frame_tracks in all_tracks:
        for t in frame_tracks:
            remaining_tids.add(t['track_id'])

    print(f"  Track consolidation: merged {len(merge_map)} track pairs, "
          f"removed {len(short_tids)} short tracks ({removed_dets} detections), "
          f"{len(remaining_tids)} tracks remaining")
    return all_tracks


def detect_roles(all_tracks, fps):
    """Assign roles: player, goalkeeper, referee, ball.

    - Ball: class != 0 (from detector)
    - Referee: player detections far from both team centroids
    - Goalkeeper: per team, player with avg position closest to goal line
    - Player: all other player detections
    """
    # Build per-track aggregated data
    track_data = defaultdict(lambda: {
        'positions': [], 'team': None, 'cls': None, 'det_count': 0,
    })
    for frame_tracks in all_tracks:
        for t in frame_tracks:
            td = track_data[t['track_id']]
            td['det_count'] += 1
            if t.get('pitch_pos') is not None:
                td['positions'].append(t['pitch_pos'])
            if t.get('team') is not None:
                td['team'] = t['team']
            td['cls'] = t.get('class', 0)

    # Compute team centroids for referee detection
    team_centroids = {}
    team_dists = {}  # tid -> distance to own team centroid
    for team_id in [0, 1]:
        team_positions = []
        for tid, td in track_data.items():
            if td['cls'] == 0 and td['team'] == team_id and td['positions']:
                team_positions.append(np.mean(td['positions'], axis=0))
        if team_positions:
            team_centroids[team_id] = np.mean(team_positions, axis=0)

    # Identify referees: player tracks with high distance from both centroids
    referee_tids = set()
    if len(team_centroids) == 2:
        for tid, td in track_data.items():
            if td['cls'] != 0 or not td['positions']:
                continue
            avg_pos = np.mean(td['positions'], axis=0)
            d0 = np.linalg.norm(avg_pos - team_centroids[0])
            d1 = np.linalg.norm(avg_pos - team_centroids[1])
            min_dist = min(d0, d1)
            # If player doesn't fit either team well (far from both centroids)
            # and has no consistent team assignment
            if td['team'] is None and min_dist > 15.0:
                referee_tids.add(tid)

    # Identify goalkeepers: per team, player with avg x closest to goal line
    gk_tids = set()
    for team_id in [0, 1]:
        team_tids = []
        for tid, td in track_data.items():
            if (td['cls'] == 0 and td['team'] == team_id
                    and td['positions'] and tid not in referee_tids):
                avg_pos = np.mean(td['positions'], axis=0)
                team_tids.append((tid, avg_pos[0]))

        if not team_tids:
            continue

        # Team near goal at x=0: goalkeeper has min avg x
        # Team near goal at x=105: goalkeeper has max avg x
        avg_x_vals = [x for _, x in team_tids]
        team_center_x = np.mean(avg_x_vals)
        if team_center_x < 52.5:
            # Team plays towards x=0, GK is closest to x=0
            gk_tid = min(team_tids, key=lambda p: p[1])[0]
        else:
            # Team plays towards x=105, GK is closest to x=105
            gk_tid = max(team_tids, key=lambda p: p[1])[0]
        gk_tids.add(gk_tid)

    # Apply roles to all detections
    role_counts = defaultdict(int)
    for frame_tracks in all_tracks:
        for t in frame_tracks:
            tid = t['track_id']
            if t.get('class', 0) != 0:
                t['role'] = 'ball'
            elif tid in referee_tids:
                t['role'] = 'referee'
            elif tid in gk_tids:
                t['role'] = 'goalkeeper'
            else:
                t['role'] = 'player'
            role_counts[t['role']] += 1

    print(f"  Role detection: {dict(role_counts)}")
    return all_tracks


# ==============================================================================
# Step 1: Detection + Tracking
# ==============================================================================

def run_detection_tracking(args):
    """Run detection + BoT-SORT tracking on all video clips."""
    print("=" * 60)
    print("STEP 1: Detection + Tracking")
    print("=" * 60)

    all_tracks = []  # list of per-frame track lists across all clips
    clip_info = []   # (video_path, n_frames, frame_offset)
    frame_offset = 0

    for i, video_path in enumerate(args.videos):
        print(f"\n  Processing clip {i+1}/{len(args.videos)}: {Path(video_path).name}")
        t0 = time.time()

        # Ensure CUDA memory is available
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info(0)
            print(f"    GPU memory: {free/1e9:.1f}/{total/1e9:.1f} GB free")

        clip_tracks = run_tracking(
            model_path=args.model,
            video_path=video_path,
            tracker_config=args.tracker,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
        )

        n_frames = len(clip_tracks)
        elapsed = time.time() - t0

        # Offset frame numbers for concatenation
        for frame_tracks in clip_tracks:
            for t in frame_tracks:
                t['frame'] += frame_offset
            all_tracks.append(frame_tracks)

        clip_info.append({
            'path': video_path,
            'n_frames': n_frames,
            'frame_offset': frame_offset,
        })

        n_dets = sum(len(ft) for ft in clip_tracks)
        track_ids = set()
        for ft in clip_tracks:
            for t in ft:
                track_ids.add(t['track_id'])

        print(f"    Frames: {n_frames}, Detections: {n_dets}, "
              f"Unique tracks: {len(track_ids)}, Time: {elapsed:.1f}s")

        frame_offset += n_frames

    total_dets = sum(len(ft) for ft in all_tracks)
    all_ids = set()
    for ft in all_tracks:
        for t in ft:
            all_ids.add(t['track_id'])

    print(f"\n  Total: {len(all_tracks)} frames, {total_dets} detections, "
          f"{len(all_ids)} unique track IDs")

    return all_tracks, clip_info


# ==============================================================================
# Step 2: Homography from Keypoints
# ==============================================================================

def compute_homography_from_keypoints(keypoints_path):
    """Load pre-annotated keypoints and compute pixel->pitch homography."""
    print("\n" + "=" * 60)
    print("STEP 2: Homography from Keypoints")
    print("=" * 60)

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

    print(f"  Loaded {len(pixel_pts)} keypoint correspondences")

    H, mask = cv2.findHomography(pixel_pts, pitch_pts, cv2.RANSAC, 5.0)
    n_inliers = int(mask.sum()) if mask is not None else 0
    print(f"  Homography computed: {n_inliers}/{len(pixel_pts)} inliers")

    # Validation: check center point
    detector = PitchDetector()
    center_idx = np.argmin(np.linalg.norm(pitch_pts - [52.5, 34.0], axis=1))
    center_px = pixel_pts[center_idx]
    center_pitch = detector.pixel_to_pitch(H, center_px)
    print(f"  Validation: center pixel {center_px} -> pitch "
          f"({center_pitch[0]:.1f}, {center_pitch[1]:.1f}) (expected ~52.5, 34.0)")

    return H


# ==============================================================================
# Step 3: Team Classification + Position Transform
# ==============================================================================

def classify_and_transform(all_tracks, clip_info, H, args):
    """Team classification via jersey color + transform positions to pitch coords."""
    print("\n" + "=" * 60)
    print("STEP 3: Team Classification + Position Transform")
    print("=" * 60)

    detector = PitchDetector()
    classifier = TeamClassifier(n_teams=2)

    # --- Calibration: read first 100 frames from first clip ---
    print("  Calibrating team classifier on first 100 frames...")
    first_clip = clip_info[0]
    cap = cv2.VideoCapture(first_clip['path'])

    cal_frames = []
    cal_dets = []
    n_cal = min(100, first_clip['n_frames'])

    for i in range(n_cal):
        ret, frame = cap.read()
        if not ret:
            break
        cal_frames.append(frame)
        cal_dets.append(all_tracks[i])
    cap.release()

    classifier.calibrate(cal_frames, cal_dets, n_calibration_frames=n_cal)
    print(f"  Calibrated on {len(cal_frames)} frames")

    # --- Apply homography to all tracks ---
    print("  Transforming positions to pitch coordinates...")
    for frame_tracks in all_tracks:
        for t in frame_tracks:
            px, py = t['center_px']
            pitch_x, pitch_y = detector.pixel_to_pitch(H, [px, py])
            pitch_x = max(0, min(105.0, pitch_x))
            pitch_y = max(0, min(68.0, pitch_y))
            t['pitch_pos'] = [float(pitch_x), float(pitch_y)]

    # --- Classify teams frame by frame ---
    print("  Classifying teams...")
    global_frame = 0
    for ci, info in enumerate(clip_info):
        cap = cv2.VideoCapture(info['path'])
        print(f"    Clip {ci+1}/{len(clip_info)}: {Path(info['path']).name}")

        for local_f in range(info['n_frames']):
            ret, frame = cap.read()
            if not ret:
                break

            frame_tracks = all_tracks[global_frame]
            if len(frame_tracks) > 0:
                team_map = classifier.classify_teams(frame, frame_tracks)
                for idx, team_label in team_map.items():
                    frame_tracks[idx]['team'] = team_label

            global_frame += 1
        cap.release()

    # Count team assignments
    team_counts = defaultdict(int)
    no_team = 0
    for ft in all_tracks:
        for t in ft:
            if t['team'] is not None:
                team_counts[t['team']] += 1
            else:
                no_team += 1

    print(f"  Team assignments: {dict(team_counts)}, unassigned: {no_team}")

    return all_tracks


# ==============================================================================
# Step 4: Render Annotated Video
# ==============================================================================

def _draw_mini_pitch(w, h, frame_tracks):
    """Draw a mini pitch map with player positions."""
    pitch = np.zeros((h, w, 3), dtype=np.uint8)
    pitch[:] = (34, 139, 34)  # Dark green

    line_color = (255, 255, 255)
    # Border
    cv2.rectangle(pitch, (2, 2), (w - 3, h - 3), line_color, 1)
    # Center line
    cv2.line(pitch, (w // 2, 2), (w // 2, h - 3), line_color, 1)
    # Center circle
    r = int(9.15 / 105 * w)
    cv2.circle(pitch, (w // 2, h // 2), r, line_color, 1)
    # Penalty areas
    pa_w = int(16.5 / 105 * w)
    pa_h = int(40.3 / 68 * h)
    pa_y1 = (h - pa_h) // 2
    cv2.rectangle(pitch, (2, pa_y1), (pa_w, pa_y1 + pa_h), line_color, 1)
    cv2.rectangle(pitch, (w - pa_w - 1, pa_y1), (w - 3, pa_y1 + pa_h), line_color, 1)

    TEAM_COLORS = {0: (255, 100, 0), 1: (0, 0, 255)}
    GK_COLORS = {0: (255, 200, 0), 1: (100, 100, 255)}
    BALL_COLOR = (0, 255, 255)
    REFEREE_COLOR = (0, 255, 0)
    DEFAULT_COLOR = (200, 200, 200)

    for t in frame_tracks:
        if t['pitch_pos'] is None:
            continue
        px, py = t['pitch_pos']
        mx = int(px / 105 * (w - 4)) + 2
        my = int(py / 68 * (h - 4)) + 2
        mx = max(2, min(w - 3, mx))
        my = max(2, min(h - 3, my))

        role = t.get('role', 'player')
        if role == 'ball':
            color = BALL_COLOR
            cv2.circle(pitch, (mx, my), 4, color, -1)
        elif role == 'referee':
            cv2.circle(pitch, (mx, my), 3, REFEREE_COLOR, -1)
        elif role == 'goalkeeper' and t['team'] is not None:
            color = GK_COLORS.get(t['team'], DEFAULT_COLOR)
            cv2.circle(pitch, (mx, my), 4, color, -1)
            cv2.circle(pitch, (mx, my), 4, (255, 255, 255), 1)  # white ring
        elif t['team'] is not None:
            color = TEAM_COLORS.get(t['team'], DEFAULT_COLOR)
            cv2.circle(pitch, (mx, my), 3, color, -1)
        else:
            cv2.circle(pitch, (mx, my), 3, DEFAULT_COLOR, -1)

    return pitch


def render_annotated_video(all_tracks, clip_info, H, args):
    """Render annotated video with bounding boxes, track IDs, and mini pitch map."""
    print("\n" + "=" * 60)
    print("STEP 4: Rendering Annotated Video")
    print("=" * 60)

    output_path = os.path.join(args.output, "annotated_tracking.mp4")

    # Get frame dimensions from first clip
    cap = cv2.VideoCapture(clip_info[0]['path'])
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
    cap.release()

    out_w = int(orig_w * args.output_scale)
    out_h = int(orig_h * args.output_scale)

    print(f"  Input: {orig_w}x{orig_h} @ {fps:.1f}fps")
    print(f"  Output: {out_w}x{out_h} -> {output_path}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    # Team colors: team 0 = blue, team 1 = red, ball = yellow, unassigned = gray
    TEAM_COLORS = {0: (255, 100, 0), 1: (0, 0, 255)}  # BGR
    GK_COLORS = {0: (255, 200, 0), 1: (100, 100, 255)}
    BALL_COLOR = (0, 255, 255)
    REFEREE_COLOR = (0, 255, 0)
    DEFAULT_COLOR = (180, 180, 180)

    # Mini pitch map dimensions
    PITCH_MAP_W = 250
    PITCH_MAP_H = int(250 * 68 / 105)
    PITCH_MAP_MARGIN = 15

    global_frame = 0
    total_frames = len(all_tracks)

    for ci, info in enumerate(clip_info):
        cap = cv2.VideoCapture(info['path'])
        print(f"  Rendering clip {ci+1}/{len(clip_info)}: {Path(info['path']).name}")

        for local_f in range(info['n_frames']):
            ret, frame = cap.read()
            if not ret:
                break

            frame_tracks = all_tracks[global_frame]

            # Scale frame
            frame = cv2.resize(frame, (out_w, out_h))
            scale = args.output_scale

            # Draw bounding boxes
            for t in frame_tracks:
                x1, y1, x2, y2 = [int(c * scale) for c in t['bbox']]
                role = t.get('role', 'player')

                if role == 'ball':
                    color = BALL_COLOR
                elif role == 'referee':
                    color = REFEREE_COLOR
                elif role == 'goalkeeper' and t['team'] is not None:
                    color = GK_COLORS.get(t['team'], DEFAULT_COLOR)
                elif t['team'] is not None:
                    color = TEAM_COLORS.get(t['team'], DEFAULT_COLOR)
                else:
                    color = DEFAULT_COLOR

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Track ID + role + confidence label
                role_tag = {'goalkeeper': 'GK', 'referee': 'REF',
                            'ball': 'BALL'}.get(role, '')
                label = f"#{t['track_id']}"
                if role_tag:
                    label += f" {role_tag}"
                label += f" {t['confidence']:.2f}"
                label_y = max(y1 - 8, 15)
                cv2.putText(frame, label, (x1, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw mini pitch map overlay in bottom-right corner
            pitch_map = _draw_mini_pitch(PITCH_MAP_W, PITCH_MAP_H, frame_tracks)
            py1 = out_h - PITCH_MAP_H - PITCH_MAP_MARGIN
            py2 = out_h - PITCH_MAP_MARGIN
            px1 = out_w - PITCH_MAP_W - PITCH_MAP_MARGIN
            px2 = out_w - PITCH_MAP_MARGIN

            roi = frame[py1:py2, px1:px2]
            blended = cv2.addWeighted(roi, 0.3, pitch_map, 0.7, 0)
            frame[py1:py2, px1:px2] = blended

            # Frame counter
            cv2.putText(frame, f"Frame {global_frame}/{total_frames}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            writer.write(frame)
            global_frame += 1

            if global_frame % 100 == 0:
                print(f"    Frame {global_frame}/{total_frames}")

        cap.release()

    writer.release()
    print(f"  Annotated video saved: {output_path}")


# ==============================================================================
# Step 5: Compute Metrics
# ==============================================================================

def compute_metrics(all_tracks, args):
    """Compute physical, tactical, and individual metrics.

    Uses role information to:
    - Exclude referees from team metrics
    - Separate GK vs outfield stats
    - Use exactly top-10 outfield players per team for formation detection
    """
    print("\n" + "=" * 60)
    print("STEP 5: Computing Metrics")
    print("=" * 60)

    total_frames = len(all_tracks)
    fps = args.fps

    # Build per-track data
    player_data = defaultdict(lambda: {
        'positions': [], 'frames': [], 'team': None, 'class': None,
        'role': 'player',
    })

    for ft in all_tracks:
        for t in ft:
            tid = t['track_id']
            if t['pitch_pos'] is not None:
                player_data[tid]['positions'].append(t['pitch_pos'])
                player_data[tid]['frames'].append(t['frame'])
            if t['team'] is not None:
                player_data[tid]['team'] = t['team']
            player_data[tid]['class'] = t.get('class', 0)
            player_data[tid]['role'] = t.get('role', 'player')

    # --- Physical + Individual Metrics ---
    print("  Computing physical metrics...")
    phys = PhysicalMetrics(fps=fps)
    indiv = IndividualMetrics()

    physical_results = {}
    individual_results = {}
    min_positions = MIN_TRACK_POSITIONS

    for tid, data in player_data.items():
        # Skip ball and referee tracks
        if data['role'] in ('ball', 'referee'):
            continue
        positions = np.array(data['positions'])
        if len(positions) < min_positions:
            continue

        phys_result = phys.compute_all(positions)
        # Clamp max speed to world-model limit
        phys_result['max_speed_kmh'] = min(phys_result['max_speed_kmh'],
                                           MAX_SPEED_KMH)

        physical_results[tid] = phys_result
        individual_results[tid] = {
            'heatmap_data': indiv.heatmap(positions).tolist(),
            'playing_time': indiv.playing_time(data['frames'], total_frames, fps=fps),
            'average_position': indiv.average_position(positions),
            'team': data['team'],
            'role': data['role'],
            'n_positions': len(positions),
        }

    print(f"  Physical metrics computed for {len(physical_results)} players "
          f"(excl. referees and short tracks)")

    # --- Tactical Metrics ---
    print("  Computing tactical metrics...")
    tact = TacticalMetrics()

    # Get average positions per team (players + goalkeepers only, no referees)
    team_positions = defaultdict(list)      # team -> [(avg_pos, role)]
    for tid, data in player_data.items():
        if data['role'] not in ('player', 'goalkeeper'):
            continue
        if data['team'] is None:
            continue
        if len(data['positions']) < min_positions:
            continue
        avg_pos = np.mean(data['positions'], axis=0)
        team_positions[data['team']].append((avg_pos, data['role']))

    tactical_results = {}
    for team_id, pos_role_list in team_positions.items():
        all_positions = np.array([p for p, _ in pos_role_list])
        team_key = f"team_{team_id}"

        result = {
            'n_players': len(all_positions),
            'compactness': tact.compactness(all_positions),
            'width_depth': tact.team_width_depth(all_positions),
        }

        # Formation detection: use only outfield players (exclude GK)
        outfield_positions = np.array(
            [p for p, r in pos_role_list if r == 'player']
        )
        if len(outfield_positions) > 10:
            # Keep only top 10 by distance from own goal (most advanced)
            # Sort by x-distance from the team's goal
            team_avg_x = np.mean(outfield_positions[:, 0])
            if team_avg_x < 52.5:
                # Team plays towards x=0, sort ascending (defenders first)
                order = np.argsort(outfield_positions[:, 0])
            else:
                # Team plays towards x=105, sort descending
                order = np.argsort(-outfield_positions[:, 0])
            outfield_positions = outfield_positions[order[:10]]

        if len(outfield_positions) >= 3:
            formation, score = tact.formation_detection(outfield_positions)
            result['formation'] = formation
            result['formation_score'] = (float(score)
                                         if score != float('inf') else None)
            result['outfield_count'] = len(outfield_positions)

        tactical_results[team_key] = result

    # Zone control (all players, no referees)
    if len(team_positions) >= 2:
        team_keys = sorted(team_positions.keys())
        team_a_pos = np.array([p for p, _ in team_positions[team_keys[0]]])
        team_b_pos = np.array([p for p, _ in team_positions[team_keys[1]]])
        tactical_results['zone_control'] = tact.zone_control(team_a_pos,
                                                             team_b_pos)

    print(f"  Tactical metrics computed for {len(team_positions)} teams")

    # Count roles in results
    role_counts = defaultdict(int)
    for tid, data in individual_results.items():
        role_counts[data.get('role', 'player')] += 1
    print(f"  Role breakdown in metrics: {dict(role_counts)}")

    metrics = {
        'physical': {str(k): v for k, v in physical_results.items()},
        'individual': {str(k): v for k, v in individual_results.items()},
        'tactical': tactical_results,
        'summary': {
            'total_frames': total_frames,
            'fps': fps,
            'duration_seconds': total_frames / fps,
            'n_players_tracked': len(physical_results),
            'n_clips': len(args.videos),
        }
    }

    return metrics


# ==============================================================================
# Step 6: Generate HTML Report
# ==============================================================================

def generate_report(metrics, args):
    """Generate Plotly-based HTML dashboard."""
    print("\n" + "=" * 60)
    print("STEP 6: Generating HTML Report")
    print("=" * 60)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    phys = metrics['physical']
    indiv = metrics['individual']
    tact = metrics['tactical']
    summary = metrics['summary']

    # Sort players by distance
    sorted_players = sorted(phys.items(), key=lambda x: x[1]['total_distance_km'], reverse=True)
    player_ids = [p[0] for p in sorted_players]

    # Generate heatmap images as base64
    heatmap_imgs = {}
    for pid in player_ids[:6]:  # Top 6 players
        if pid in indiv and 'heatmap_data' in indiv[pid]:
            fig, ax = plt.subplots(1, 1, figsize=(5.25, 3.4))
            hm = np.array(indiv[pid]['heatmap_data'])
            ax.imshow(hm, cmap='hot', interpolation='bilinear', aspect='auto',
                      extent=[0, 105, 68, 0])
            team_label = f"Team {'A' if indiv[pid].get('team') == 0 else 'B'}"
            ax.set_title(f"Player #{pid} ({team_label})", fontsize=10)
            ax.set_xlabel("Length (m)")
            ax.set_ylabel("Width (m)")
            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=80, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            heatmap_imgs[pid] = base64.b64encode(buf.read()).decode('utf-8')

    # Chart data
    distances = [phys[p]['total_distance_km'] for p in player_ids]
    avg_speeds = [phys[p]['avg_speed_kmh'] for p in player_ids]
    max_speeds = [phys[p]['max_speed_kmh'] for p in player_ids]
    teams = [indiv.get(p, {}).get('team', -1) for p in player_ids]
    team_colors = ['#0064FF' if t == 0 else '#FF0000' if t == 1 else '#999' for t in teams]
    labels = [f"#{p}" for p in player_ids]

    # Speed zone data
    zone_names = ['walking', 'jogging', 'running', 'high_speed', 'sprinting']
    zone_colors = ['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#F44336']
    zone_data = {zn: [] for zn in zone_names}
    for p in player_ids:
        for zn in zone_names:
            zone_data[zn].append(phys[p]['speed_zones'][zn]['percentage'])

    # Zone control
    zc = tact.get('zone_control', {'team_a': 50, 'team_b': 50})

    # Formation info
    formation_info = ""
    for tk, tv in tact.items():
        if tk.startswith('team_'):
            team_letter = 'A' if tk == 'team_0' else 'B'
            formation_info += f"<li><strong>Team {team_letter}:</strong> "
            if tv.get('formation'):
                formation_info += f"Formation: {tv['formation']}"
                if tv.get('formation_score') is not None:
                    formation_info += f" (score: {tv['formation_score']:.1f})"
                formation_info += ", "
            if tv.get('compactness'):
                formation_info += f"Compactness: {tv['compactness']['area_m2']:.0f} m&sup2;, "
            if tv.get('width_depth'):
                formation_info += (f"Width: {tv['width_depth']['width_m']:.1f}m, "
                                   f"Depth: {tv['width_depth']['depth_m']:.1f}m")
            formation_info += f", Players: {tv.get('n_players', '?')}"
            if tv.get('outfield_count') is not None:
                formation_info += f" (outfield: {tv['outfield_count']})"
            formation_info += "</li>"

    # Summary table rows
    summary_rows = ""
    for pid in player_ids:
        p = phys[pid]
        i = indiv.get(pid, {})
        team_label = 'A' if i.get('team') == 0 else 'B' if i.get('team') == 1 else '?'
        role = i.get('role', 'player')
        role_label = {'goalkeeper': 'GK', 'player': ''}.get(role, role)
        avg_pos = i.get('average_position', {})
        pt = i.get('playing_time', {})
        summary_rows += (
            f"<tr>"
            f"<td>#{pid}</td><td>{team_label}</td><td>{role_label}</td>"
            f"<td>{p['total_distance_km']:.3f}</td>"
            f"<td>{p['avg_speed_kmh']:.1f}</td>"
            f"<td>{p['max_speed_kmh']:.1f}</td>"
            f"<td>{p['sprint_count']}</td>"
            f"<td>{pt.get('playing_time_minutes', 0):.1f}</td>"
            f"<td>({avg_pos.get('x', 0):.1f}, {avg_pos.get('y', 0):.1f})</td>"
            f"</tr>\n"
        )

    # Heatmap HTML
    heatmap_html = ""
    for pid, img_b64 in heatmap_imgs.items():
        heatmap_html += (
            f'<img src="data:image/png;base64,{img_b64}" '
            f'style="margin:5px; border:1px solid #ddd; border-radius:4px;">\n'
        )

    # Build speed zone traces for Plotly
    zone_traces = []
    for i, zn in enumerate(zone_names):
        zone_traces.append(
            f'{{"x": {json.dumps(labels)}, "y": {json.dumps(zone_data[zn])}, '
            f'"name": "{zn.replace("_", " ").title()}", "type": "bar", '
            f'"marker": {{"color": "{zone_colors[i]}"}}}}'
        )
    zone_traces_js = ",\n    ".join(zone_traces)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Football Analysis Report</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
    body {{ font-family: 'Segoe UI', sans-serif; margin: 20px; background: #f5f5f5; color: #333; }}
    .container {{ max-width: 1400px; margin: auto; }}
    h1 {{ color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 10px; }}
    h2 {{ color: #283593; margin-top: 30px; }}
    .card {{ background: white; border-radius: 8px; padding: 20px; margin: 15px 0;
             box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }}
    .chart {{ min-height: 400px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #ddd; }}
    th {{ background: #e8eaf6; font-weight: 600; }}
    tr:hover {{ background: #f5f5f5; }}
    .summary-stat {{ display: inline-block; background: #e8eaf6; padding: 10px 20px;
                     border-radius: 8px; margin: 5px; text-align: center; }}
    .summary-stat .value {{ font-size: 24px; font-weight: bold; color: #1a237e; }}
    .summary-stat .label {{ font-size: 12px; color: #666; }}
    .heatmaps {{ display: flex; flex-wrap: wrap; justify-content: center; }}
</style>
</head>
<body>
<div class="container">
<h1>Football Video Analysis Report</h1>

<div class="card">
    <h2>Overview</h2>
    <div style="text-align: center;">
        <div class="summary-stat">
            <div class="value">{summary['n_players_tracked']}</div>
            <div class="label">Players Tracked</div>
        </div>
        <div class="summary-stat">
            <div class="value">{summary['duration_seconds']:.0f}s</div>
            <div class="label">Duration</div>
        </div>
        <div class="summary-stat">
            <div class="value">{summary['total_frames']}</div>
            <div class="label">Total Frames</div>
        </div>
        <div class="summary-stat">
            <div class="value">{summary['n_clips']}</div>
            <div class="label">Video Clips</div>
        </div>
        <div class="summary-stat">
            <div class="value">{summary['fps']:.0f}</div>
            <div class="label">FPS</div>
        </div>
    </div>
</div>

<div class="grid">
    <div class="card">
        <h2>Distance Covered (km)</h2>
        <div id="distance-chart" class="chart"></div>
    </div>
    <div class="card">
        <h2>Speed (km/h)</h2>
        <div id="speed-chart" class="chart"></div>
    </div>
</div>

<div class="card">
    <h2>Speed Zone Distribution (%)</h2>
    <div id="zone-chart" class="chart"></div>
</div>

<div class="grid">
    <div class="card">
        <h2>Zone Control</h2>
        <div id="zone-control-chart" class="chart"></div>
    </div>
    <div class="card">
        <h2>Tactical Analysis</h2>
        <ul>{formation_info}</ul>
    </div>
</div>

<div class="card">
    <h2>Player Heatmaps (Top Players)</h2>
    <div class="heatmaps">{heatmap_html}</div>
</div>

<div class="card">
    <h2>Player Summary Table</h2>
    <table>
        <thead><tr>
            <th>Player</th><th>Team</th><th>Role</th><th>Distance (km)</th>
            <th>Avg Speed (km/h)</th><th>Max Speed (km/h)</th><th>Sprints</th>
            <th>Playing Time (min)</th><th>Avg Position</th>
        </tr></thead>
        <tbody>{summary_rows}</tbody>
    </table>
</div>

</div>

<script>
// Distance chart
Plotly.newPlot('distance-chart', [{{
    x: {json.dumps(labels)},
    y: {json.dumps(distances)},
    type: 'bar',
    marker: {{ color: {json.dumps(team_colors)} }}
}}], {{
    margin: {{t: 10, b: 60}},
    yaxis: {{title: 'km'}},
    xaxis: {{tickangle: -45}}
}});

// Speed chart
Plotly.newPlot('speed-chart', [
    {{ x: {json.dumps(labels)}, y: {json.dumps(avg_speeds)}, name: 'Avg', type: 'bar',
       marker: {{color: '#42A5F5'}} }},
    {{ x: {json.dumps(labels)}, y: {json.dumps(max_speeds)}, name: 'Max', type: 'bar',
       marker: {{color: '#EF5350'}} }}
], {{
    barmode: 'group',
    margin: {{t: 10, b: 60}},
    yaxis: {{title: 'km/h'}},
    xaxis: {{tickangle: -45}}
}});

// Speed zone stacked bar
Plotly.newPlot('zone-chart', [
    {zone_traces_js}
], {{
    barmode: 'stack',
    margin: {{t: 10, b: 60}},
    yaxis: {{title: '%'}},
    xaxis: {{tickangle: -45}},
    legend: {{orientation: 'h', y: 1.1}}
}});

// Zone control pie
Plotly.newPlot('zone-control-chart', [{{
    values: [{zc.get('team_a', 50):.1f}, {zc.get('team_b', 50):.1f}],
    labels: ['Team A', 'Team B'],
    type: 'pie',
    marker: {{ colors: ['#0064FF', '#FF0000'] }},
    hole: 0.4
}}], {{
    margin: {{t: 30, b: 30}}
}});
</script>
</body>
</html>"""

    report_path = os.path.join(args.output, "analysis_report.html")
    with open(report_path, 'w') as f:
        f.write(html)
    print(f"  Report saved: {report_path}")

    return report_path


# ==============================================================================
# Main
# ==============================================================================

def main():
    args = parse_args()

    # Resolve paths relative to project root
    os.chdir(PROJECT_ROOT)
    os.makedirs(args.output, exist_ok=True)

    # --hybrid flag: delegate to hybrid orchestrator
    if getattr(args, 'hybrid', False):
        from src.hybrid.orchestrator import HybridPipelineOrchestrator
        orchestrator = HybridPipelineOrchestrator(
            model_path=args.model,
            videos=args.videos,
            keypoints_path=args.keypoints,
            tracker_config=args.tracker,
            yolo_device=args.device,
            imgsz=args.imgsz,
            conf=args.conf,
            fps=args.fps,
            output_scale=args.output_scale,
            output_dir=args.output,
            gemma4_model=getattr(args, 'gemma4_model', '/home/jovyan/.local/models/gemma4-e4b'),
            gemma4_device=getattr(args, 'gemma4_device', 'cuda:1'),
            no_gemma4=getattr(args, 'no_gemma4', False),
        )
        orchestrator.run()
        return

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output dir:   {args.output}")
    print(f"Model:        {args.model}")
    print(f"Videos:       {args.videos}")
    print(f"Device: {args.device}, imgsz: {args.imgsz}, conf: {args.conf}")
    print()

    t_start = time.time()

    # Step 1: Detection + Tracking
    all_tracks, clip_info = run_detection_tracking(args)

    # Step 1a: Top-K detection filter
    print("\n  Applying world model constraints...")
    all_tracks = apply_topk_filter(all_tracks)

    # Step 2: Homography
    H = compute_homography_from_keypoints(args.keypoints)

    # Step 3: Team Classification + Position Transform
    all_tracks = classify_and_transform(all_tracks, clip_info, H, args)

    # Step 3a: Pitch boundary filter (drop out-of-field detections)
    all_tracks = apply_pitch_boundary_filter(all_tracks)

    # Step 3b: Kalman filter smoothing + outlier rejection
    apply_kalman_smoothing(all_tracks, args.fps)

    # Step 3c: Track consolidation
    all_tracks = consolidate_tracks(all_tracks, args.fps)

    # Step 3d: Role detection (needs team assignments)
    all_tracks = detect_roles(all_tracks, args.fps)

    # Step 4: Annotated Video
    render_annotated_video(all_tracks, clip_info, H, args)

    # Step 5: Metrics
    metrics = compute_metrics(all_tracks, args)

    # Save raw data
    tracks_path = os.path.join(args.output, "tracks.json")
    flat_tracks = []
    for ft in all_tracks:
        for t in ft:
            flat_tracks.append(t)
    with open(tracks_path, 'w') as f:
        json.dump(flat_tracks, f, indent=2)
    print(f"\n  Tracks saved: {tracks_path} ({len(flat_tracks)} detections)")

    metrics_path = os.path.join(args.output, "metrics.json")
    # Strip heatmap_data from metrics JSON (too large for JSON)
    metrics_save = json.loads(json.dumps(metrics))
    for pid in metrics_save.get('individual', {}):
        metrics_save['individual'][pid].pop('heatmap_data', None)
    with open(metrics_path, 'w') as f:
        json.dump(metrics_save, f, indent=2)
    print(f"  Metrics saved: {metrics_path}")

    # Step 6: HTML Report
    generate_report(metrics, args)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"PIPELINE COMPLETE — Total time: {elapsed:.1f}s")
    print(f"{'=' * 60}")
    print(f"Outputs in {args.output}:")
    print(f"  - annotated_tracking.mp4")
    print(f"  - tracks.json")
    print(f"  - metrics.json")
    print(f"  - analysis_report.html")


if __name__ == "__main__":
    main()
