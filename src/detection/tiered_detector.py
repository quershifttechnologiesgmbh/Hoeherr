"""
Tiered Detection Module.

Two-pass detection strategy:
  Pass 1: YOLO detection on scaled frame (1920px) — fast, gets rough positions
  Pass 2: Extract 4K crops for each detection — jersey numbers, appearance features

This avoids the downsample-or-SAHI dilemma:
  - Downsampling to 1280px loses jersey details (players are ~15px)
  - Full SAHI on 4K is 6x slower
  - Tiered approach: fast detection + detailed crops only where needed
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict


class TieredDetector:
    """Two-pass detection: scaled detection + full-res crop extraction."""

    def __init__(
        self,
        detection_width: int = 1920,
        crop_size: int = 256,        # Size of 4K crops around each detection
        crop_padding: float = 1.5,   # Padding factor for crops
    ):
        self.detection_width = detection_width
        self.crop_size = crop_size
        self.crop_padding = crop_padding

    def prepare_detection_frame(self, frame_4k: np.ndarray) -> Tuple[np.ndarray, float]:
        """Scale 4K frame down to detection resolution.

        Args:
            frame_4k: Full resolution frame (e.g., 3840x2160)

        Returns:
            (scaled_frame, scale_factor)
        """
        h, w = frame_4k.shape[:2]
        scale = self.detection_width / w
        new_w = self.detection_width
        new_h = int(h * scale)
        scaled = cv2.resize(frame_4k, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return scaled, scale

    def scale_detections_to_4k(
        self, detections: list, scale_factor: float
    ) -> list:
        """Scale detection bboxes back to 4K coordinates.

        Args:
            detections: List of track dicts with 'bbox' and 'center_px'
            scale_factor: Factor used to scale 4K → detection resolution

        Returns:
            detections with bbox and center_px in 4K coordinates
        """
        inv_scale = 1.0 / scale_factor
        for det in detections:
            if 'bbox' in det:
                det['bbox_scaled'] = det['bbox'].copy()
                det['bbox'] = [c * inv_scale for c in det['bbox']]
            if 'center_px' in det:
                det['center_px_scaled'] = det['center_px'].copy()
                det['center_px'] = [c * inv_scale for c in det['center_px']]
        return detections

    def extract_crops(
        self, frame_4k: np.ndarray, detections: list
    ) -> Dict[int, np.ndarray]:
        """Extract full-resolution crops around each detection.

        Args:
            frame_4k: Full resolution frame
            detections: Track dicts with 4K-scale bboxes

        Returns:
            Dict mapping detection index to cropped image
        """
        h, w = frame_4k.shape[:2]
        crops = {}

        for idx, det in enumerate(detections):
            bbox = det.get('bbox', [0, 0, 0, 0])
            x1, y1, x2, y2 = bbox

            # Calculate padded crop region
            bw = x2 - x1
            bh = y2 - y1
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # Expand by padding factor
            crop_w = max(bw * self.crop_padding, self.crop_size)
            crop_h = max(bh * self.crop_padding, self.crop_size)

            cx1 = int(max(0, cx - crop_w / 2))
            cy1 = int(max(0, cy - crop_h / 2))
            cx2 = int(min(w, cx + crop_w / 2))
            cy2 = int(min(h, cy + crop_h / 2))

            if cx2 - cx1 < 10 or cy2 - cy1 < 10:
                continue

            crop = frame_4k[cy1:cy2, cx1:cx2]

            # Resize to standard crop size for consistent feature extraction
            if crop.shape[0] > 0 and crop.shape[1] > 0:
                crop = cv2.resize(crop, (self.crop_size, self.crop_size))
                crops[idx] = crop

        return crops

    def extract_jersey_crop(
        self, frame_4k: np.ndarray, bbox: list
    ) -> Optional[np.ndarray]:
        """Extract a high-resolution crop focused on the jersey area.

        For jersey number recognition, we need:
        - Upper 50% of player (jersey/back area)
        - Full 4K resolution for maximum detail
        - Sufficient padding for OCR

        Args:
            frame_4k: Full resolution frame
            bbox: [x1, y1, x2, y2] in 4K coordinates

        Returns:
            Cropped jersey region or None
        """
        h, w = frame_4k.shape[:2]
        x1, y1, x2, y2 = [int(c) for c in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 - x1 < 10 or y2 - y1 < 15:
            return None

        # Take upper 50% (jersey area)
        jersey_h = int((y2 - y1) * 0.5)
        jersey = frame_4k[y1:y1 + jersey_h, x1:x2]

        if jersey.shape[0] < 5 or jersey.shape[1] < 5:
            return None

        # Upscale if too small
        if jersey.shape[1] < 128:
            scale = 128 / jersey.shape[1]
            jersey = cv2.resize(
                jersey, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_CUBIC
            )

        return jersey


class TieredTrackingAdapter:
    """Adapter that integrates tiered detection with the existing BoT-SORT pipeline.

    Usage:
        adapter = TieredTrackingAdapter(detection_width=1920)
        all_tracks, all_crops = adapter.run_tracking_tiered(
            model_path, video_path, tracker_config, ...
        )
    """

    def __init__(self, detection_width: int = 1920, crop_size: int = 256):
        self.tiered = TieredDetector(
            detection_width=detection_width,
            crop_size=crop_size,
        )

    def run_tracking_tiered(
        self,
        model_path: str,
        video_path: str,
        tracker_config: str = "botsort.yaml",
        conf: float = 0.25,
        iou: float = 0.45,
        device: str = "cuda:0",
        extract_crops: bool = True,
        crop_interval: int = 5,  # Extract crops every N frames
    ) -> Tuple[list, Optional[Dict[int, Dict[int, np.ndarray]]]]:
        """Run YOLO tracking on scaled frames, with optional 4K crop extraction.

        Instead of modifying the YOLO tracker (which uses its own internal
        frame reading), we:
        1. Run tracking normally on the video
        2. Make a second pass to extract 4K crops at specified intervals

        Args:
            model_path: Path to YOLO model
            video_path: Path to video file
            tracker_config: Tracker configuration
            conf: Detection confidence threshold
            iou: IoU threshold
            device: GPU device
            extract_crops: Whether to extract 4K crops
            crop_interval: Extract crops every N frames

        Returns:
            (all_tracks, frame_crops)
            frame_crops: {frame_idx: {det_idx: crop_image}}
        """
        from src.tracking.bot_sort_tracker import run_tracking

        # Pass 1: Standard tracking (YOLO handles its own frame reading)
        # YOLO will use its internal imgsz parameter for scaling
        all_tracks = run_tracking(
            model_path=model_path,
            video_path=video_path,
            tracker_config=tracker_config,
            conf=conf,
            imgsz=self.tiered.detection_width,
            device=device,
        )

        # Pass 2: Extract 4K crops at intervals
        frame_crops = None
        if extract_crops:
            frame_crops = self._extract_crops_pass(
                video_path, all_tracks, crop_interval
            )

        return all_tracks, frame_crops

    def _extract_crops_pass(
        self,
        video_path: str,
        all_tracks: list,
        crop_interval: int,
    ) -> Dict[int, Dict[int, np.ndarray]]:
        """Second pass: read video at full resolution and extract crops.

        Args:
            video_path: Path to video
            all_tracks: Tracking results from Pass 1
            crop_interval: Extract crops every N frames

        Returns:
            {frame_idx: {det_idx: crop_image}}
        """
        cap = cv2.VideoCapture(video_path)
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        total_frames = len(all_tracks)

        # Calculate scale factor from detection to original resolution
        scale_factor = self.tiered.detection_width / orig_w

        frame_crops = {}
        frame_idx = 0

        while True:
            ret, frame_4k = cap.read()
            if not ret or frame_idx >= total_frames:
                break

            if frame_idx % crop_interval == 0 and frame_idx < len(all_tracks):
                frame_tracks = all_tracks[frame_idx]
                if frame_tracks:
                    # Scale detections to 4K coordinates
                    scaled_tracks = []
                    for t in frame_tracks:
                        if t.get('bbox'):
                            scaled_bbox = [c / scale_factor for c in t['bbox']]
                            scaled_tracks.append({**t, 'bbox': scaled_bbox})

                    crops = self.tiered.extract_crops(frame_4k, scaled_tracks)
                    if crops:
                        frame_crops[frame_idx] = crops

            frame_idx += 1

        cap.release()

        print(f"  [TieredDetection] Extracted crops for {len(frame_crops)} frames "
              f"from {total_frames} total")

        return frame_crops
