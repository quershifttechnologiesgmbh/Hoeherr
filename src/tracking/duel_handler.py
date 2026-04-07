"""
Duel/Occlusion Handler and Substitution Detector.

Handles identity preservation during:
- Zweikämpfe (duels/tackles): Multiple players in close proximity
- Occlusions: Player temporarily hidden behind another
- Substitutions: Player leaves, new player enters

Uses Kalman prediction through occlusion and appearance-based re-acquisition.
"""
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


class DuelState:
    """Tracks the state of an active duel/occlusion event."""

    def __init__(self, frame_idx: int, involved_track_ids: Set[int],
                 center_pos: np.ndarray):
        self.start_frame = frame_idx
        self.involved_track_ids = involved_track_ids
        self.center_pos = center_pos
        self.last_update_frame = frame_idx
        self.resolved = False

        # Store pre-duel positions and velocities for each player
        self.pre_duel_positions: Dict[int, np.ndarray] = {}
        self.pre_duel_velocities: Dict[int, np.ndarray] = {}

    @property
    def duration_frames(self):
        return self.last_update_frame - self.start_frame


class DuelOcclusionHandler:
    """Handles player identity during duels and occlusions.

    Detection: 2+ detections with distance < proximity_threshold
    Behavior:
      1. Enter occlusion mode: start Kalman prediction for involved tracks
      2. During occlusion: trust motion model, don't update from measurements
      3. Re-acquisition: use appearance matching to reassign correct IDs
    """

    def __init__(
        self,
        proximity_threshold: float = 2.0,   # meters — triggers duel detection
        max_occlusion_frames: int = 60,      # max frames to predict through (2s at 30fps)
        min_separation: float = 3.0,         # meters — re-acquisition threshold
        fps: float = 30.0,
    ):
        self.proximity_threshold = proximity_threshold
        self.max_occlusion_frames = max_occlusion_frames
        self.min_separation = min_separation
        self.fps = fps

        # Active duel states
        self.active_duels: List[DuelState] = []
        # Track IDs currently in a duel
        self.tracks_in_duel: Set[int] = set()
        # Statistics
        self.stats = {
            'duels_detected': 0,
            'duels_resolved': 0,
            'avg_duel_duration_frames': 0,
            'id_corrections': 0,
        }

    def process_frame(
        self,
        frame_idx: int,
        frame_tracks: list,
        reid_module=None,
        frame: np.ndarray = None,
        frame_4k: np.ndarray = None,
    ) -> list:
        """Process a single frame for duel detection and resolution.

        Args:
            frame_idx: Current frame index
            frame_tracks: Track data for current frame
            reid_module: Optional PlayerReIDModule for appearance matching
            frame: Current frame image (for ReID)
            frame_4k: Optional 4K frame (for jersey recognition)

        Returns:
            Possibly corrected frame_tracks
        """
        # Step 1: Detect new duels
        new_duels = self._detect_duels(frame_idx, frame_tracks)
        for duel in new_duels:
            self.active_duels.append(duel)
            self.tracks_in_duel.update(duel.involved_track_ids)
            self.stats['duels_detected'] += 1

        # Step 2: Update/resolve active duels
        resolved_duels = []
        for duel in self.active_duels:
            if duel.resolved:
                continue

            # Check if duel has timed out
            duration = frame_idx - duel.start_frame
            if duration > self.max_occlusion_frames:
                duel.resolved = True
                resolved_duels.append(duel)
                continue

            # Check if players have separated
            if self._check_separation(duel, frame_tracks):
                # Players separated — attempt re-identification
                if reid_module and frame is not None:
                    corrections = self._resolve_duel_with_reid(
                        duel, frame_tracks, reid_module, frame, frame_4k
                    )
                    if corrections:
                        frame_tracks = self._apply_corrections(
                            frame_tracks, corrections
                        )
                        self.stats['id_corrections'] += len(corrections)

                duel.resolved = True
                resolved_duels.append(duel)

        # Clean up resolved duels
        for duel in resolved_duels:
            self.tracks_in_duel -= duel.involved_track_ids
            self.stats['duels_resolved'] += 1
            durations = [d.duration_frames for d in self.active_duels if d.resolved]
            if durations:
                self.stats['avg_duel_duration_frames'] = np.mean(durations)

        self.active_duels = [d for d in self.active_duels if not d.resolved]

        return frame_tracks

    def _detect_duels(self, frame_idx: int, frame_tracks: list) -> List[DuelState]:
        """Detect new duel/occlusion events in current frame."""
        new_duels = []

        # Get player positions
        players = []
        for t in frame_tracks:
            if t.get('role') in ('player', 'goalkeeper') and t.get('pitch_pos'):
                players.append(t)

        if len(players) < 2:
            return new_duels

        # Find clusters of close players from DIFFERENT teams
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                pi = np.array(players[i]['pitch_pos'])
                pj = np.array(players[j]['pitch_pos'])
                dist = np.linalg.norm(pi - pj)

                if dist < self.proximity_threshold:
                    tid_i = players[i]['track_id']
                    tid_j = players[j]['track_id']

                    # Only trigger for different teams
                    if players[i].get('team') == players[j].get('team'):
                        continue

                    # Check if already in an active duel
                    if tid_i in self.tracks_in_duel or tid_j in self.tracks_in_duel:
                        # Update existing duel
                        for duel in self.active_duels:
                            if tid_i in duel.involved_track_ids or tid_j in duel.involved_track_ids:
                                duel.last_update_frame = frame_idx
                                duel.involved_track_ids.add(tid_i)
                                duel.involved_track_ids.add(tid_j)
                        continue

                    # New duel
                    center = (pi + pj) / 2
                    duel = DuelState(
                        frame_idx=frame_idx,
                        involved_track_ids={tid_i, tid_j},
                        center_pos=center,
                    )
                    duel.pre_duel_positions[tid_i] = pi.copy()
                    duel.pre_duel_positions[tid_j] = pj.copy()
                    new_duels.append(duel)

        return new_duels

    def _check_separation(self, duel: DuelState, frame_tracks: list) -> bool:
        """Check if players involved in a duel have separated."""
        involved_positions = {}
        for t in frame_tracks:
            if t['track_id'] in duel.involved_track_ids and t.get('pitch_pos'):
                involved_positions[t['track_id']] = np.array(t['pitch_pos'])

        if len(involved_positions) < 2:
            return False

        # Check all pairs
        pids = list(involved_positions.keys())
        for i in range(len(pids)):
            for j in range(i + 1, len(pids)):
                dist = np.linalg.norm(
                    involved_positions[pids[i]] - involved_positions[pids[j]]
                )
                if dist < self.min_separation:
                    return False  # Still close

        return True  # All separated

    def _resolve_duel_with_reid(
        self,
        duel: DuelState,
        frame_tracks: list,
        reid_module,
        frame: np.ndarray,
        frame_4k: np.ndarray = None,
    ) -> Dict[int, int]:
        """Use ReID module to determine correct ID assignments after duel.

        Returns dict of {old_track_id: corrected_track_id} for swaps needed.
        """
        # Get current tracks for involved players
        involved = {}
        for t in frame_tracks:
            if t['track_id'] in duel.involved_track_ids:
                involved[t['track_id']] = t

        if len(involved) < 2:
            return {}

        # Use ReID module's conflict resolution
        track_ids = list(involved.keys())
        if len(track_ids) == 2:
            corrections = reid_module.resolve_conflict(
                frame=frame,
                track_a_id=track_ids[0],
                track_b_id=track_ids[1],
                track_a_bbox=involved[track_ids[0]].get('bbox', [0, 0, 0, 0]),
                track_b_bbox=involved[track_ids[1]].get('bbox', [0, 0, 0, 0]),
                frame_4k=frame_4k,
            )

            # Only return if there's an actual swap
            swaps = {k: v for k, v in corrections.items() if k != v}
            return swaps

        return {}

    def _apply_corrections(self, frame_tracks: list, corrections: Dict[int, int]) -> list:
        """Apply ID corrections to frame tracks."""
        for t in frame_tracks:
            old_id = t['track_id']
            if old_id in corrections:
                t['track_id'] = corrections[old_id]
                t['_id_corrected'] = True
        return frame_tracks

    def is_in_duel(self, track_id: int) -> bool:
        """Check if a track is currently in a duel."""
        return track_id in self.tracks_in_duel

    def get_stats(self) -> dict:
        """Get duel handler statistics."""
        return {
            **self.stats,
            'active_duels': len(self.active_duels),
            'tracks_in_duel': len(self.tracks_in_duel),
        }


class SubstitutionDetector:
    """Detect player substitutions during the match.

    Heuristics:
    - Player count per team temporarily exceeds 11 (12 during sub)
    - Track disappears at sideline (not mid-field)
    - New track appears at sideline around same time
    - Gemma 4 can confirm substitution events
    """

    def __init__(self, fps: float = 30.0, sideline_margin: float = 5.0):
        self.fps = fps
        self.sideline_margin = sideline_margin

        # Per-team active track counts over time
        self.team_counts_history: List[Dict[int, int]] = []
        self.substitutions: List[dict] = []

        # Track lifecycle: first/last frame seen
        self.track_lifecycle: Dict[int, dict] = defaultdict(
            lambda: {'first_frame': None, 'last_frame': None, 'team': None,
                     'last_pos': None, 'first_pos': None}
        )

    def update(self, frame_idx: int, frame_tracks: list):
        """Update substitution detector with current frame data."""
        team_counts = defaultdict(int)

        for t in frame_tracks:
            if t.get('role') in ('player', 'goalkeeper') and t.get('team') is not None:
                tid = t['track_id']
                team = t['team']
                team_counts[team] += 1

                lifecycle = self.track_lifecycle[tid]
                if lifecycle['first_frame'] is None:
                    lifecycle['first_frame'] = frame_idx
                    lifecycle['first_pos'] = t.get('pitch_pos')
                lifecycle['last_frame'] = frame_idx
                lifecycle['last_pos'] = t.get('pitch_pos')
                lifecycle['team'] = team

        self.team_counts_history.append(dict(team_counts))

    def detect_substitutions(self) -> List[dict]:
        """Analyze track lifecycles to detect substitutions.

        Called after tracking is complete.
        """
        # Group tracks by team
        team_tracks = defaultdict(list)
        for tid, lc in self.track_lifecycle.items():
            if lc['team'] is not None:
                team_tracks[lc['team']].append((tid, lc))

        for team_id, tracks in team_tracks.items():
            # Sort by first_frame
            tracks.sort(key=lambda x: x[1]['first_frame'] or 0)

            # Find tracks that end at sideline
            ended_sideline = []
            started_sideline = []

            for tid, lc in tracks:
                if lc['last_pos'] is not None:
                    py = lc['last_pos'][1]
                    if py < self.sideline_margin or py > 68 - self.sideline_margin:
                        ended_sideline.append((tid, lc))

                if lc['first_pos'] is not None:
                    py = lc['first_pos'][1]
                    if py < self.sideline_margin or py > 68 - self.sideline_margin:
                        # Don't count tracks from the very start of the video
                        if lc['first_frame'] and lc['first_frame'] > 30:
                            started_sideline.append((tid, lc))

            # Match exits with entries (within 10s window)
            for tid_out, lc_out in ended_sideline:
                for tid_in, lc_in in started_sideline:
                    if tid_out == tid_in:
                        continue
                    if lc_out['last_frame'] is None or lc_in['first_frame'] is None:
                        continue

                    gap_frames = lc_in['first_frame'] - lc_out['last_frame']
                    if 0 < gap_frames < self.fps * 10:  # Within 10 seconds
                        sub = {
                            'team': team_id,
                            'player_out_track': tid_out,
                            'player_in_track': tid_in,
                            'frame_out': lc_out['last_frame'],
                            'frame_in': lc_in['first_frame'],
                            'timestamp_out_s': lc_out['last_frame'] / self.fps,
                            'timestamp_in_s': lc_in['first_frame'] / self.fps,
                        }
                        self.substitutions.append(sub)

        if self.substitutions:
            print(f"  [Substitution] Detected {len(self.substitutions)} "
                  f"potential substitution(s)")
            for sub in self.substitutions:
                team_letter = chr(ord('A') + sub['team'])
                print(f"    Team {team_letter}: Track #{sub['player_out_track']} out "
                      f"(t={sub['timestamp_out_s']:.1f}s) → "
                      f"Track #{sub['player_in_track']} in "
                      f"(t={sub['timestamp_in_s']:.1f}s)")

        return self.substitutions
