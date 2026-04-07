"""
Per-track Kalman filter for pitch-coordinate smoothing and outlier rejection.

Uses a constant-velocity motion model on (x, y) pitch coordinates.
Outlier measurements (from tracker ID switches, homography noise, etc.)
are detected via Mahalanobis gating and replaced with the filter prediction.

State vector:  [x, y, vx, vy]
Measurement:   [x, y]
"""
import numpy as np
from collections import defaultdict


class PlayerKalmanFilter:
    """Single-player Kalman filter with constant-velocity model."""

    def __init__(self, dt=1.0 / 30, process_noise_std=0.5, measurement_noise_std=0.8,
                 gate_threshold=4.0):
        """
        Parameters
        ----------
        dt : float
            Time step between frames (1/fps).
        process_noise_std : float
            Std of acceleration noise (m/s^2). Controls how much the filter
            trusts the motion model vs measurements.  0.5 m/s^2 is a moderate
            value for football players who can accelerate ~3-4 m/s^2.
        measurement_noise_std : float
            Std of measurement noise in meters. Accounts for homography
            reprojection error (~0.5-1.0 m typical).
        gate_threshold : float
            Mahalanobis distance threshold for outlier rejection. 4.0 ≈ 99.99%
            of a chi-squared(2) distribution, so only truly wild jumps are
            rejected.
        """
        self.dt = dt
        self.gate_threshold = gate_threshold

        # State transition: constant velocity
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float64)

        # Measurement matrix: observe position only
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        # Process noise (discrete white-noise acceleration model)
        q = process_noise_std ** 2
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        self.Q = q * np.array([
            [dt4/4, 0,     dt3/2, 0    ],
            [0,     dt4/4, 0,     dt3/2],
            [dt3/2, 0,     dt2,   0    ],
            [0,     dt3/2, 0,     dt2  ],
        ], dtype=np.float64)

        # Measurement noise
        r = measurement_noise_std ** 2
        self.R = np.array([
            [r, 0],
            [0, r],
        ], dtype=np.float64)

        # State and covariance (initialized on first measurement)
        self.x = None  # state [x, y, vx, vy]
        self.P = None  # covariance (4x4)
        self._initialized = False

    def initialize(self, z):
        """Initialize state from first measurement."""
        self.x = np.array([z[0], z[1], 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([1.0, 1.0, 5.0, 5.0])  # uncertain velocity
        self._initialized = True

    def predict(self):
        """Predict next state."""
        if not self._initialized:
            return None
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].copy()

    def update(self, z):
        """Update state with measurement. Returns (filtered_pos, was_gated).

        If measurement is an outlier (Mahalanobis distance > gate), the
        measurement is rejected and the prediction is used instead.
        """
        if not self._initialized:
            self.initialize(z)
            return self.x[:2].copy(), False

        # Innovation (measurement residual)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R  # innovation covariance

        # Mahalanobis gating
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Degenerate covariance — accept measurement
            S_inv = np.eye(2)

        mahal_dist = np.sqrt(float(y.T @ S_inv @ y))

        if mahal_dist > self.gate_threshold:
            # Outlier: reject measurement, use prediction only
            return self.x[:2].copy(), True

        # Standard Kalman update
        K = self.P @ self.H.T @ S_inv
        self.x = self.x + K @ y
        I_KH = np.eye(4) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T  # Joseph form

        return self.x[:2].copy(), False

    def get_velocity(self):
        """Return current estimated velocity (vx, vy) in m/frame."""
        if not self._initialized:
            return np.array([0.0, 0.0])
        return self.x[2:4].copy() * self.dt  # convert from m/s to m/frame

    @property
    def speed_ms(self):
        """Current estimated speed in m/s."""
        if not self._initialized:
            return 0.0
        return float(np.linalg.norm(self.x[2:4]))


def smooth_tracks(all_tracks, fps=30.0, process_noise_std=0.5,
                  measurement_noise_std=0.8, gate_threshold=4.0):
    """Apply per-track Kalman filtering to all tracks.

    Replaces raw pitch_pos with smoothed positions and rejects outlier
    measurements (position jumps from ID switches, etc.).

    Parameters
    ----------
    all_tracks : list[list[dict]]
        Per-frame list of track dicts. Each dict must have 'track_id' and
        'pitch_pos' fields.
    fps : float
        Video frame rate.
    process_noise_std : float
        Acceleration noise std (m/s^2).
    measurement_noise_std : float
        Measurement noise std (m).
    gate_threshold : float
        Mahalanobis distance for outlier gating.

    Returns
    -------
    all_tracks : list[list[dict]]
        Tracks with smoothed pitch_pos and added 'pitch_pos_raw' field.
    stats : dict
        Smoothing statistics.
    """
    dt = 1.0 / fps
    filters = {}  # track_id -> PlayerKalmanFilter
    total_updated = 0
    total_gated = 0
    total_predicted = 0

    for frame_tracks in all_tracks:
        # First pass: predict all active filters
        predicted_this_frame = set()
        for t in frame_tracks:
            tid = t['track_id']
            if tid in filters:
                filters[tid].predict()
                predicted_this_frame.add(tid)

        # Second pass: update with measurements
        for t in frame_tracks:
            tid = t['track_id']
            if t.get('pitch_pos') is None:
                continue

            raw_pos = t['pitch_pos']
            t['pitch_pos_raw'] = list(raw_pos)  # keep raw for reference

            z = np.array(raw_pos, dtype=np.float64)

            if tid not in filters:
                filters[tid] = PlayerKalmanFilter(
                    dt=dt,
                    process_noise_std=process_noise_std,
                    measurement_noise_std=measurement_noise_std,
                    gate_threshold=gate_threshold,
                )

            filtered_pos, was_gated = filters[tid].update(z)

            # Clamp to pitch boundaries
            filtered_pos[0] = max(0.0, min(105.0, filtered_pos[0]))
            filtered_pos[1] = max(0.0, min(68.0, filtered_pos[1]))

            t['pitch_pos'] = [float(filtered_pos[0]), float(filtered_pos[1])]

            total_updated += 1
            if was_gated:
                total_gated += 1

    stats = {
        'total_measurements': total_updated,
        'outliers_rejected': total_gated,
        'rejection_rate_pct': (total_gated / total_updated * 100) if total_updated > 0 else 0,
        'active_filters': len(filters),
    }
    return all_tracks, stats
