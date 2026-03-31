"""
Pitch line detection and homography computation.
Drone perspective simplifies this: near-orthogonal top-down view.
Standard pitch dimensions: 105m x 68m (FIFA).
"""
import cv2
import numpy as np


class PitchDetector:
    PITCH_LENGTH = 105.0
    PITCH_WIDTH = 68.0
    PENALTY_AREA_LENGTH = 16.5
    PENALTY_AREA_WIDTH = 40.3
    GOAL_AREA_LENGTH = 5.5
    GOAL_AREA_WIDTH = 18.3
    CENTER_CIRCLE_RADIUS = 9.15
    PENALTY_SPOT_DISTANCE = 11.0

    def detect_lines(self, frame):
        """Detect pitch lines via Hough Transform."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([30, 30, 30])
        upper_green = np.array([90, 255, 255])
        grass_mask = cv2.inRange(hsv, lower_green, upper_green)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        line_mask = cv2.bitwise_and(white_mask, grass_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, kernel)
        line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_OPEN, kernel)

        edges = cv2.Canny(line_mask, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                                minLineLength=50, maxLineGap=20)

        return lines, line_mask

    def find_keypoints(self, frame):
        """Find key points: corners, center circle, penalty area corners."""
        lines, line_mask = self.detect_lines(frame)

        if lines is None:
            return None

        intersections = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                pt = self._line_intersection(lines[i][0], lines[j][0])
                if pt is not None:
                    x, y = pt
                    if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                        intersections.append(pt)

        circles = cv2.HoughCircles(line_mask, cv2.HOUGH_GRADIENT,
                                   dp=1.2, minDist=100,
                                   param1=50, param2=30,
                                   minRadius=30, maxRadius=200)

        return {
            'intersections': intersections,
            'circles': circles
        }

    def compute_homography(self, src_points, dst_points=None):
        """
        Compute homography matrix: Pixel -> Pitch coordinates (meters).
        Requires at least 4 point correspondences.
        """
        if dst_points is None:
            dst_points = np.array([
                [0, 0],
                [self.PITCH_LENGTH, 0],
                [self.PITCH_LENGTH, self.PITCH_WIDTH],
                [0, self.PITCH_WIDTH],
                [self.PITCH_LENGTH / 2, self.PITCH_WIDTH / 2],
            ], dtype=np.float32)

        src_points = np.array(src_points[:len(dst_points)], dtype=np.float32)
        dst_points = np.array(dst_points[:len(src_points)], dtype=np.float32)

        if len(src_points) < 4:
            return None

        H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
        return H

    def pixel_to_pitch(self, H, pixel_coords):
        """Transform pixel coordinates to pitch coordinates (meters)."""
        px = np.array([pixel_coords[0], pixel_coords[1], 1.0])
        pitch = H @ px
        pitch = pitch / pitch[2]
        return pitch[0], pitch[1]

    @staticmethod
    def _line_intersection(line1, line2):
        """Compute intersection point of two lines."""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        return (ix, iy)
