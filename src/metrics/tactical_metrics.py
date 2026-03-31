"""
Tactical metrics from team positions:
Formation detection, compactness, zone control via Voronoi.
"""
import numpy as np
from scipy.spatial import ConvexHull, Voronoi
from sklearn.cluster import KMeans


class TacticalMetrics:

    def formation_detection(self, positions):
        """
        Detect formation (e.g., 4-3-3, 4-4-2) via clustering.
        positions: np.array shape (10, 2) — 10 outfield players
        """
        if len(positions) < 3:
            return None, float('inf')

        sorted_positions = positions[positions[:, 1].argsort()]

        formations = {
            '4-4-2': [4, 4, 2],
            '4-3-3': [4, 3, 3],
            '3-5-2': [3, 5, 2],
            '4-2-3-1': [4, 2, 3, 1],
            '4-1-4-1': [4, 1, 4, 1],
            '3-4-3': [3, 4, 3],
        }

        best_formation = None
        best_score = float('inf')

        for name, line_sizes in formations.items():
            if sum(line_sizes) != len(positions):
                continue
            score = self._formation_fit_score(sorted_positions, line_sizes)
            if score < best_score:
                best_score = score
                best_formation = name

        return best_formation, best_score

    def compactness(self, positions):
        """Team compactness via Convex Hull area."""
        if len(positions) < 3:
            return None
        try:
            hull = ConvexHull(positions)
            return {
                'area_m2': float(hull.volume),
                'perimeter_m': float(hull.area),
            }
        except Exception:
            return None

    def team_width_depth(self, positions):
        """Team width and depth in meters."""
        if len(positions) < 2:
            return {'width_m': 0.0, 'depth_m': 0.0}
        return {
            'width_m': float(np.max(positions[:, 0]) - np.min(positions[:, 0])),
            'depth_m': float(np.max(positions[:, 1]) - np.min(positions[:, 1])),
        }

    def defensive_line_height(self, defender_positions):
        """Average position of the defensive line."""
        if len(defender_positions) == 0:
            return 0.0
        return float(np.mean(defender_positions[:, 1]))

    def zone_control(self, team_a_positions, team_b_positions, pitch_length=105, pitch_width=68):
        """Zone control via Voronoi tessellation."""
        if len(team_a_positions) < 1 or len(team_b_positions) < 1:
            return {'team_a': 50.0, 'team_b': 50.0}

        all_positions = np.vstack([team_a_positions, team_b_positions])
        n_a = len(team_a_positions)

        if len(all_positions) < 3:
            return {'team_a': 50.0, 'team_b': 50.0}

        try:
            vor = Voronoi(all_positions)
        except Exception:
            return {'team_a': 50.0, 'team_b': 50.0}

        team_a_area = 0
        team_b_area = 0

        for i, region_idx in enumerate(vor.point_region):
            region = vor.regions[region_idx]
            if -1 in region or len(region) == 0:
                continue
            polygon = vor.vertices[region]
            area = self._polygon_area(polygon)
            if i < n_a:
                team_a_area += area
            else:
                team_b_area += area

        total = team_a_area + team_b_area
        if total == 0:
            return {'team_a': 50.0, 'team_b': 50.0}

        return {
            'team_a': float(team_a_area / total * 100),
            'team_b': float(team_b_area / total * 100),
        }

    def _formation_fit_score(self, sorted_positions, line_sizes):
        n_lines = len(line_sizes)
        y_coords = sorted_positions[:, 1].reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_lines, random_state=42, n_init=10)
        labels = kmeans.fit_predict(y_coords)

        cluster_means = [(i, kmeans.cluster_centers_[i][0]) for i in range(n_lines)]
        cluster_means.sort(key=lambda x: x[1])

        score = 0
        for idx, (cluster_id, _) in enumerate(cluster_means):
            actual_size = int(np.sum(labels == cluster_id))
            expected_size = line_sizes[idx]
            score += abs(actual_size - expected_size)

        return score

    @staticmethod
    def _polygon_area(polygon):
        n = len(polygon)
        if n < 3:
            return 0
        area = 0
        for i in range(n):
            j = (i + 1) % n
            area += polygon[i][0] * polygon[j][1]
            area -= polygon[j][0] * polygon[i][1]
        return abs(area) / 2
