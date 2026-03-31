"""
Main pipeline: Video in -> Analysis report out.
Orchestrates detection, tracking, team classification, homography, and metrics.
"""
import json
import time
import numpy as np
from pathlib import Path

from src.tracking.bot_sort_tracker import run_tracking
from src.team_classification.jersey_classifier import TeamClassifier
from src.homography.pitch_detector import PitchDetector
from src.metrics.physical_metrics import PhysicalMetrics
from src.metrics.tactical_metrics import TacticalMetrics
from src.metrics.individual_metrics import IndividualMetrics


class HoeherrPipeline:
    def __init__(self, model_path: str, tracker_config: str = "configs/botsort.yaml",
                 fps: int = 25, device: str = "cpu"):
        self.model_path = model_path
        self.tracker_config = tracker_config
        self.fps = fps
        self.device = device

        self.team_classifier = TeamClassifier()
        self.pitch_detector = PitchDetector()
        self.physical = PhysicalMetrics(fps=fps)
        self.tactical = TacticalMetrics()
        self.individual = IndividualMetrics()

    def process_match(self, video_path: str, output_dir: str) -> dict:
        """Process a complete match video."""
        start_time = time.time()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[1/5] Detection + Tracking: {video_path}")
        tracks = self._detect_and_track(video_path)

        print(f"[2/5] Team Classification")
        tracks = self._classify_teams(video_path, tracks)

        print(f"[3/5] Homography Calibration")
        homography = self._calibrate_homography(video_path)

        print(f"[4/5] Pitch Position Transform")
        tracks = self._transform_to_pitch(tracks, homography)

        print(f"[5/5] Computing Metrics")
        report = self._compute_metrics(tracks)

        # Save results
        with open(output_dir / "tracks.json", 'w') as f:
            json.dump(tracks, f)
        with open(output_dir / "report.json", 'w') as f:
            json.dump(report, f, default=str)

        elapsed = time.time() - start_time
        report['match_info']['processing_time_minutes'] = round(elapsed / 60, 2)
        print(f"Done! Processing: {elapsed / 60:.1f} minutes")
        return report

    def _detect_and_track(self, video_path: str):
        return run_tracking(
            model_path=self.model_path,
            video_path=video_path,
            tracker_config=self.tracker_config,
            conf=0.25,
            iou=0.45,
            imgsz=640,
            device=self.device,
            vid_stride=1,
        )

    def _classify_teams(self, video_path: str, tracks):
        import cv2
        cap = cv2.VideoCapture(video_path)

        calibration_frames = []
        for _ in range(100):
            ret, frame = cap.read()
            if not ret:
                break
            calibration_frames.append(frame)

        if calibration_frames and tracks:
            self.team_classifier.calibrate(calibration_frames, tracks[:100])

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for frame_idx, frame_tracks in enumerate(tracks):
            ret, frame = cap.read()
            if not ret:
                break
            team_assignments = self.team_classifier.classify_teams(frame, frame_tracks)
            for idx, team in team_assignments.items():
                tracks[frame_idx][idx]['team'] = team

        cap.release()
        return tracks

    def _calibrate_homography(self, video_path: str):
        import cv2
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None

        keypoints = self.pitch_detector.find_keypoints(frame)
        if keypoints and len(keypoints.get('intersections', [])) >= 4:
            src_points = keypoints['intersections'][:4]
            H = self.pitch_detector.compute_homography(src_points)
        else:
            from src.homography.gps_fallback import gps_homography
            H, _, _ = gps_homography(
                drone_lat=0, drone_lon=0, drone_altitude=50,
                pitch_center_lat=0, pitch_center_lon=0,
                pitch_orientation_deg=0,
                image_width=frame.shape[1],
                image_height=frame.shape[0]
            )
        return H

    def _transform_to_pitch(self, tracks, H):
        if H is None:
            return tracks
        for frame_tracks in tracks:
            for det in frame_tracks:
                try:
                    pitch_pos = self.pitch_detector.pixel_to_pitch(H, det['center_px'])
                    det['pitch_pos'] = [float(pitch_pos[0]), float(pitch_pos[1])]
                except Exception:
                    det['pitch_pos'] = None
        return tracks

    def _compute_metrics(self, tracks):
        player_tracks = {}
        for frame_tracks in tracks:
            for det in frame_tracks:
                tid = det['track_id']
                if tid not in player_tracks:
                    player_tracks[tid] = {
                        'positions': [],
                        'frames': [],
                        'team': det.get('team'),
                        'class': det.get('class', 0)
                    }
                if det.get('pitch_pos'):
                    player_tracks[tid]['positions'].append(det['pitch_pos'])
                    player_tracks[tid]['frames'].append(det['frame'])

        total_frames = len(tracks)
        report = {
            'match_info': {
                'total_frames': total_frames,
                'duration_minutes': round(total_frames / self.fps / 60, 2),
                'fps': self.fps,
            },
            'players': {},
            'team_metrics': {},
        }

        team_positions = {0: [], 1: []}

        for track_id, pdata in player_tracks.items():
            if len(pdata['positions']) < 100:
                continue

            positions = np.array(pdata['positions'])
            physical = self.physical.compute_all(positions)
            heatmap_data = self.individual.heatmap(positions)
            play_time = self.individual.playing_time(pdata['frames'], total_frames, self.fps)
            avg_pos = self.individual.average_position(positions)

            report['players'][str(track_id)] = {
                'team': pdata['team'],
                'class': pdata['class'],
                'physical': physical,
                'playing_time': play_time,
                'average_position': avg_pos,
                'heatmap': heatmap_data.tolist(),
            }

            if pdata['team'] in (0, 1):
                team_positions[pdata['team']].append(np.mean(positions, axis=0))

        # Team-level metrics
        for team_id in [0, 1]:
            if len(team_positions[team_id]) >= 3:
                tp = np.array(team_positions[team_id])
                report['team_metrics'][str(team_id)] = {
                    'compactness': self.tactical.compactness(tp),
                    'width_depth': self.tactical.team_width_depth(tp),
                }

        # Zone control
        if len(team_positions[0]) >= 1 and len(team_positions[1]) >= 1:
            report['team_metrics']['zone_control'] = self.tactical.zone_control(
                np.array(team_positions[0]),
                np.array(team_positions[1])
            )

        return report
