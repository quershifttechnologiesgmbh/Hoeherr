"""Tests for tactical metrics."""
import numpy as np
import pytest
from src.metrics.tactical_metrics import TacticalMetrics


class TestTacticalMetrics:
    def setup_method(self):
        self.tactical = TacticalMetrics()

    def test_compactness(self):
        """Compact team should have smaller area."""
        compact = np.array([[50, 30], [52, 30], [50, 32], [52, 32]])
        spread = np.array([[10, 10], [90, 10], [10, 60], [90, 60]])

        c1 = self.tactical.compactness(compact)
        c2 = self.tactical.compactness(spread)

        assert c1 is not None
        assert c2 is not None
        assert c1['area_m2'] < c2['area_m2']

    def test_team_width_depth(self):
        """Width and depth calculation."""
        positions = np.array([[10, 20], [30, 40], [50, 60]])
        result = self.tactical.team_width_depth(positions)
        assert result['width_m'] == 40.0
        assert result['depth_m'] == 40.0

    def test_zone_control_equal(self):
        """Symmetric positions should give ~50/50 control."""
        team_a = np.array([[25, 34], [52.5, 17], [52.5, 51]])
        team_b = np.array([[80, 34], [52.5, 10], [52.5, 58]])
        result = self.tactical.zone_control(team_a, team_b)
        assert 'team_a' in result
        assert 'team_b' in result
        assert abs(result['team_a'] + result['team_b'] - 100.0) < 1.0

    def test_formation_detection(self):
        """Formation detection returns a valid formation name."""
        # 10 players in a rough 4-3-3 shape
        positions = np.array([
            [20, 10], [40, 10], [60, 10], [80, 10],  # defense
            [30, 30], [50, 30], [70, 30],              # midfield
            [25, 55], [50, 55], [75, 55],              # attack
        ])
        formation, score = self.tactical.formation_detection(positions)
        assert formation is not None
        assert formation in ['4-4-2', '4-3-3', '3-5-2', '4-2-3-1', '4-1-4-1', '3-4-3']
