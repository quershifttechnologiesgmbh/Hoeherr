"""Tests for individual metrics."""
import numpy as np
from src.metrics.individual_metrics import IndividualMetrics


class TestIndividualMetrics:
    def setup_method(self):
        self.individual = IndividualMetrics()

    def test_heatmap_shape(self):
        """Heatmap should have correct dimensions."""
        positions = np.array([[52.5, 34.0]] * 100)
        heatmap = self.individual.heatmap(positions)
        assert heatmap.shape == (136, 210)  # 68/0.5, 105/0.5

    def test_heatmap_peak(self):
        """Heatmap peak should be near the player's position."""
        positions = np.array([[52.5, 34.0]] * 1000)
        heatmap = self.individual.heatmap(positions)
        peak_y, peak_x = np.unravel_index(heatmap.argmax(), heatmap.shape)
        assert abs(peak_x - 105) < 10  # 52.5m / 0.5 = 105
        assert abs(peak_y - 68) < 10   # 34m / 0.5 = 68

    def test_playing_time(self):
        """Playing time calculation."""
        frames = list(range(0, 1500))  # 60 seconds at 25fps
        result = self.individual.playing_time(frames, 2500, fps=25)
        assert result['frames_tracked'] == 1500
        assert abs(result['playing_time_minutes'] - 1.0) < 0.01
        assert result['percentage'] == 60.0

    def test_average_position(self):
        """Average position calculation."""
        positions = np.array([[10, 20], [30, 40], [50, 60]])
        result = self.individual.average_position(positions)
        assert result['x'] == 30.0
        assert result['y'] == 40.0
