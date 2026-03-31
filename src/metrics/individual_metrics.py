"""
Individual player metrics: heatmaps, playing time, movement patterns.
"""
import numpy as np
from scipy.ndimage import gaussian_filter


class IndividualMetrics:

    def heatmap(self, positions, pitch_length=105, pitch_width=68, resolution=0.5):
        """
        Create activity heatmap.
        resolution: meters per pixel (0.5 = 210x136 grid)
        """
        grid_w = int(pitch_length / resolution)
        grid_h = int(pitch_width / resolution)
        heatmap = np.zeros((grid_h, grid_w))

        for pos in positions:
            x_idx = int(pos[0] / resolution)
            y_idx = int(pos[1] / resolution)
            if 0 <= x_idx < grid_w and 0 <= y_idx < grid_h:
                heatmap[y_idx, x_idx] += 1

        heatmap = gaussian_filter(heatmap, sigma=3)

        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap

    def playing_time(self, track_frames, total_frames, fps=25):
        """Actual playing time (frames where player is tracked)."""
        return {
            'frames_tracked': len(track_frames),
            'total_frames': total_frames,
            'playing_time_minutes': float(len(track_frames) / fps / 60),
            'percentage': float(len(track_frames) / total_frames * 100) if total_frames > 0 else 0.0
        }

    def average_position(self, positions):
        """Average position on the pitch."""
        if len(positions) == 0:
            return {'x': 0.0, 'y': 0.0}
        avg = np.mean(positions, axis=0)
        return {'x': float(avg[0]), 'y': float(avg[1])}
