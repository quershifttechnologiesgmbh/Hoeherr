"""Video utility functions."""
import cv2
import os


def extract_frames(video_path: str, output_dir: str, every_n_frames: int = 5) -> int:
    """Extract every nth frame from a video."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % every_n_frames == 0:
            filename = f"frame_{frame_count:06d}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), frame)
            saved_count += 1
        frame_count += 1

    cap.release()
    return saved_count


def get_video_info(video_path: str) -> dict:
    """Get video metadata."""
    cap = cv2.VideoCapture(video_path)
    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration_seconds': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0,
    }
    cap.release()
    return info
