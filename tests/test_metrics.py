"""Tests for physical metrics computation."""
import numpy as np
import pytest
from src.metrics.physical_metrics import PhysicalMetrics


class TestPhysicalMetrics:
    def setup_method(self):
        self.metrics = PhysicalMetrics(fps=25)

    def test_stationary_player(self):
        """Stationary player: 0 distance, 0 speed."""
        positions = np.ones((250, 2)) * 50
        result = self.metrics.compute_all(positions)
        assert result['total_distance_km'] < 0.001
        assert result['sprint_count'] == 0

    def test_known_distance(self):
        """Player runs 100m in 10 seconds (10 m/s = 36 km/h)."""
        frames = 250
        positions = np.column_stack([
            np.linspace(0, 100, frames),
            np.ones(frames) * 34
        ])
        result = self.metrics.compute_all(positions)
        assert abs(result['total_distance_km'] - 0.1) < 0.02
        assert result['max_speed_kmh'] > 25

    def test_sprint_detection(self):
        """Detects sprint correctly."""
        frames = 500
        positions = np.zeros((frames, 2))
        positions[:250, 0] = np.linspace(0, 20, 250)
        positions[250:375, 0] = np.linspace(20, 60, 125)
        positions[375:, 0] = np.linspace(60, 70, 125)
        result = self.metrics.compute_all(positions)
        assert result['sprint_count'] >= 1

    def test_speed_zones(self):
        """Speed zones sum to ~100%."""
        positions = np.column_stack([
            np.linspace(0, 50, 250),
            np.ones(250) * 34
        ])
        result = self.metrics.compute_all(positions)
        total_pct = sum(z['percentage'] for z in result['speed_zones'].values())
        assert abs(total_pct - 100.0) < 1.0

    def test_empty_positions(self):
        """Handles very short position arrays."""
        positions = np.array([[0, 0], [1, 1]])
        result = self.metrics.compute_all(positions)
        assert result['total_distance_km'] == 0.0


class TestPhysicalMetricsFPS:
    def test_different_fps(self):
        """Metrics should scale with FPS."""
        for fps in [25, 30, 50]:
            m = PhysicalMetrics(fps=fps)
            frames = fps * 10  # 10 seconds
            positions = np.column_stack([
                np.linspace(0, 100, frames),
                np.ones(frames) * 34
            ])
            result = m.compute_all(positions)
            assert abs(result['total_distance_km'] - 0.1) < 0.02
