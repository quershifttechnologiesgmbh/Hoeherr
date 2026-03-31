"""
Physical performance metrics from tracking data.
Input: pitch coordinates (meters) per player per frame
Output: distance, speed, sprints, acceleration
"""
import numpy as np
from scipy.ndimage import uniform_filter1d


class PhysicalMetrics:
    SPEED_ZONES = {
        'walking': (0, 7),
        'jogging': (7, 14),
        'running': (14, 19.8),
        'high_speed': (19.8, 25.1),
        'sprinting': (25.1, float('inf'))
    }

    def __init__(self, fps=25):
        self.fps = fps
        self.dt = 1.0 / fps

    def compute_all(self, positions):
        """
        Compute all physical metrics for a player.
        positions: np.array shape (N, 2) — (x, y) in meters per frame
        """
        if len(positions) < 3:
            return self._empty_metrics()

        smoothed = self._smooth_positions(positions, window=5)
        velocity = np.diff(smoothed, axis=0) / self.dt
        speed_ms = np.linalg.norm(velocity, axis=1)
        speed_kmh = speed_ms * 3.6

        acceleration = np.diff(velocity, axis=0) / self.dt
        accel_magnitude = np.linalg.norm(acceleration, axis=1)

        diffs = np.linalg.norm(np.diff(smoothed, axis=0), axis=1)

        return {
            'total_distance_km': float(np.sum(diffs) / 1000),
            'avg_speed_kmh': float(np.mean(speed_kmh)),
            'max_speed_kmh': float(np.max(speed_kmh)),
            'speed_zones': self._speed_zone_distribution(speed_kmh),
            'sprint_count': self._count_sprints(speed_kmh),
            'high_speed_distance_m': float(self._distance_in_zone(smoothed, speed_kmh, 'high_speed')),
            'sprint_distance_m': float(self._distance_in_zone(smoothed, speed_kmh, 'sprinting')),
            'high_intensity_accelerations': int(np.sum(accel_magnitude > 3.0)),
            'high_intensity_decelerations': int(np.sum(accel_magnitude > 3.0)),
        }

    def _empty_metrics(self):
        return {
            'total_distance_km': 0.0,
            'avg_speed_kmh': 0.0,
            'max_speed_kmh': 0.0,
            'speed_zones': {zone: {'time_seconds': 0.0, 'percentage': 0.0} for zone in self.SPEED_ZONES},
            'sprint_count': 0,
            'high_speed_distance_m': 0.0,
            'sprint_distance_m': 0.0,
            'high_intensity_accelerations': 0,
            'high_intensity_decelerations': 0,
        }

    def _smooth_positions(self, positions, window=5):
        smoothed = np.copy(positions).astype(float)
        smoothed[:, 0] = uniform_filter1d(positions[:, 0].astype(float), size=window)
        smoothed[:, 1] = uniform_filter1d(positions[:, 1].astype(float), size=window)
        return smoothed

    def _speed_zone_distribution(self, speed_kmh):
        total_frames = len(speed_kmh)
        distribution = {}
        for zone_name, (low, high) in self.SPEED_ZONES.items():
            frames_in_zone = np.sum((speed_kmh >= low) & (speed_kmh < high))
            distribution[zone_name] = {
                'time_seconds': float(frames_in_zone * self.dt),
                'percentage': float(frames_in_zone / total_frames * 100) if total_frames > 0 else 0.0
            }
        return distribution

    def _count_sprints(self, speed_kmh, min_duration_frames=5):
        is_sprinting = speed_kmh > 25.1
        sprint_count = 0
        current_duration = 0
        for s in is_sprinting:
            if s:
                current_duration += 1
            else:
                if current_duration >= min_duration_frames:
                    sprint_count += 1
                current_duration = 0
        if current_duration >= min_duration_frames:
            sprint_count += 1
        return sprint_count

    def _distance_in_zone(self, positions, speed_kmh, zone_name):
        low, high = self.SPEED_ZONES[zone_name]
        mask = (speed_kmh >= low) & (speed_kmh < high)
        diffs = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        return float(np.sum(diffs[mask]))
