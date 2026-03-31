"""
Team classification via unsupervised KMeans clustering on jersey colors.
Workflow:
1. Crop bounding box
2. Take upper 40% (jersey region)
3. Convert to HSV, mask out grass and skin
4. KMeans (k=2) for 2 teams
"""
import cv2
import numpy as np
from sklearn.cluster import KMeans


class TeamClassifier:
    def __init__(self, n_teams=2):
        self.n_teams = n_teams
        self.kmeans = None
        self.team_colors = None

    def extract_jersey_color(self, frame, bbox):
        """Extract dominant jersey color from a bounding box crop."""
        x1, y1, x2, y2 = map(int, bbox[:4])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        h = crop.shape[0]
        jersey_region = crop[:int(h * 0.4), :]

        if jersey_region.size == 0 or jersey_region.shape[0] < 2 or jersey_region.shape[1] < 2:
            return None

        hsv = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2HSV)

        # Mask out grass (green)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Mask out skin tones
        lower_skin = np.array([0, 40, 80])
        upper_skin = np.array([25, 170, 255])
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

        exclude_mask = cv2.bitwise_or(green_mask, skin_mask)
        jersey_mask = cv2.bitwise_not(exclude_mask)

        jersey_pixels = hsv[jersey_mask > 0]
        if len(jersey_pixels) < 10:
            return None

        return np.mean(jersey_pixels, axis=0)

    def classify_teams(self, frame, detections):
        """Classify all player detections into teams."""
        colors = []
        valid_indices = []

        for i, det in enumerate(detections):
            if det.get('class', 0) == 0:  # player class only
                color = self.extract_jersey_color(frame, det['bbox'])
                if color is not None:
                    colors.append(color)
                    valid_indices.append(i)

        if len(colors) < 4:
            return {}

        colors = np.array(colors)

        if self.kmeans is None:
            self.kmeans = KMeans(n_clusters=self.n_teams, random_state=42, n_init=10)
            self.kmeans.fit(colors)
            self.team_colors = self.kmeans.cluster_centers_

        labels = self.kmeans.predict(colors)
        team_assignments = {}
        for idx, label in zip(valid_indices, labels):
            team_assignments[idx] = int(label)

        return team_assignments

    def calibrate(self, frames, all_detections, n_calibration_frames=100):
        """Calibrate team colors over the first N frames for stable assignment."""
        all_colors = []
        for frame_idx in range(min(n_calibration_frames, len(frames))):
            frame = frames[frame_idx]
            dets = all_detections[frame_idx] if frame_idx < len(all_detections) else []
            for det in dets:
                if det.get('class', 0) == 0:
                    color = self.extract_jersey_color(frame, det['bbox'])
                    if color is not None:
                        all_colors.append(color)

        if len(all_colors) > 10:
            self.kmeans = KMeans(n_clusters=self.n_teams, random_state=42, n_init=10)
            self.kmeans.fit(np.array(all_colors))
            self.team_colors = self.kmeans.cluster_centers_
