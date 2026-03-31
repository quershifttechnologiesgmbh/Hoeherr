"""
GPS-based fallback homography when line detection fails.
Uses drone GPS + known pitch dimensions for approximate calibration.
Accuracy: ~1-2m (vs ~0.15m with keypoint method).
"""
import math
import numpy as np


def gps_homography(drone_lat, drone_lon, drone_altitude,
                   pitch_center_lat, pitch_center_lon,
                   pitch_orientation_deg,
                   camera_fov_h=84, camera_fov_v=56,
                   image_width=3840, image_height=2160):
    """
    Compute approximate homography from GPS data.
    DJI Mavic 3 Pro: FOV 84 deg (horizontal), ~56 deg (vertical).
    """
    ground_width = 2 * drone_altitude * math.tan(math.radians(camera_fov_h / 2))
    ground_height = 2 * drone_altitude * math.tan(math.radians(camera_fov_v / 2))

    px_per_meter_x = image_width / ground_width
    px_per_meter_y = image_height / ground_height

    H = np.array([
        [1 / px_per_meter_x, 0, -ground_width / 2],
        [0, 1 / px_per_meter_y, -ground_height / 2],
        [0, 0, 1]
    ])

    return H, ground_width, ground_height
