"""Tests for homography computation."""
import numpy as np
from src.homography.pitch_detector import PitchDetector
from src.homography.gps_fallback import gps_homography


class TestPitchDetector:
    def setup_method(self):
        self.detector = PitchDetector()

    def test_pitch_dimensions(self):
        """Standard pitch dimensions."""
        assert self.detector.PITCH_LENGTH == 105.0
        assert self.detector.PITCH_WIDTH == 68.0

    def test_homography_with_known_points(self):
        """Homography with 4 known correspondences should work."""
        src = np.array([[100, 100], [3740, 100], [3740, 2060], [100, 2060]], dtype=np.float32)
        dst = np.array([[0, 0], [105, 0], [105, 68], [0, 68]], dtype=np.float32)

        H = self.detector.compute_homography(src.tolist(), dst)
        assert H is not None
        assert H.shape == (3, 3)

    def test_pixel_to_pitch(self):
        """Pixel to pitch transform should be approximately correct."""
        src = np.array([[0, 0], [1000, 0], [1000, 680], [0, 680]], dtype=np.float32)
        dst = np.array([[0, 0], [105, 0], [105, 68], [0, 68]], dtype=np.float32)

        H = self.detector.compute_homography(src.tolist(), dst)
        pitch_x, pitch_y = self.detector.pixel_to_pitch(H, [500, 340])
        assert abs(pitch_x - 52.5) < 1.0
        assert abs(pitch_y - 34.0) < 1.0

    def test_too_few_points(self):
        """Less than 4 points should return None."""
        src = [[100, 100], [200, 200], [300, 300]]
        H = self.detector.compute_homography(src)
        assert H is None


class TestGPSFallback:
    def test_gps_homography(self):
        """GPS homography should return a valid matrix."""
        H, gw, gh = gps_homography(
            drone_lat=51.0, drone_lon=7.0, drone_altitude=50,
            pitch_center_lat=51.0, pitch_center_lon=7.0,
            pitch_orientation_deg=0
        )
        assert H is not None
        assert H.shape == (3, 3)
        assert gw > 0
        assert gh > 0
