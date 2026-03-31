"""
Multi-object tracking using BoT-SORT via Ultralytics.
Optimized for drone footage with Camera Motion Compensation (CMC).
"""
import numpy as np
from pathlib import Path


def run_tracking(model_path: str, video_path: str, tracker_config: str = "botsort.yaml",
                 conf: float = 0.25, iou: float = 0.45, imgsz: int = 640,
                 device: str = "cpu", vid_stride: int = 1):
    """
    Run detection + tracking on a video using BoT-SORT.
    Returns list of per-frame track data.
    """
    from ultralytics import YOLO

    model = YOLO(model_path)
    results = model.track(
        source=video_path,
        tracker=tracker_config,
        persist=True,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        stream=True,
        vid_stride=vid_stride,
    )

    all_tracks = []
    for frame_idx, result in enumerate(results):
        frame_tracks = []
        if result.boxes.id is not None:
            for box, track_id, cls, score in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.id.cpu().numpy(),
                result.boxes.cls.cpu().numpy(),
                result.boxes.conf.cpu().numpy()
            ):
                frame_tracks.append({
                    'frame': frame_idx,
                    'track_id': int(track_id),
                    'bbox': [float(x) for x in box],
                    'center_px': [float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)],
                    'class': int(cls),
                    'confidence': float(score),
                    'team': None,
                    'pitch_pos': None,
                })
        all_tracks.append(frame_tracks)

    return all_tracks


def tracks_to_mot_format(all_tracks):
    """Convert internal track format to MOT Challenge format for evaluation."""
    lines = []
    for frame_tracks in all_tracks:
        for t in frame_tracks:
            x1, y1, x2, y2 = t['bbox']
            w = x2 - x1
            h = y2 - y1
            lines.append(f"{t['frame'] + 1},{t['track_id']},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{t['confidence']:.4f},-1,-1,-1")
    return "\n".join(lines)


def load_mot_format(filepath: str):
    """Load MOT Challenge format annotations."""
    tracks = {}
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split(',')
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])

            if frame_id not in tracks:
                tracks[frame_id] = []
            tracks[frame_id].append({
                'id': track_id,
                'x': x, 'y': y, 'w': w, 'h': h
            })
    return tracks
