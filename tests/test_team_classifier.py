"""Tests for team classification."""
import numpy as np
from src.team_classification.jersey_classifier import TeamClassifier


class TestTeamClassifier:
    def test_init(self):
        tc = TeamClassifier(n_teams=2)
        assert tc.n_teams == 2
        assert tc.kmeans is None

    def test_extract_jersey_color_empty_bbox(self):
        """Empty/invalid bbox should return None."""
        tc = TeamClassifier()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = tc.extract_jersey_color(frame, [50, 50, 50, 50])
        assert result is None

    def test_classify_teams_too_few_detections(self):
        """Too few detections should return empty dict."""
        tc = TeamClassifier()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        dets = [{'bbox': [10, 10, 20, 20], 'class': 0}]
        result = tc.classify_teams(frame, dets)
        assert result == {}
