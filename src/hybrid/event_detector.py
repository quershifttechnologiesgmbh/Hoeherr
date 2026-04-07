"""
Gemma 4-based Event Detection.

Uses YOLO tracking heuristics to identify candidate events (ball near goal,
player convergence, ball direction changes), then Gemma 4 to classify and
describe them with visual context.
"""
import time
from collections import defaultdict

import numpy as np


# Pitch geometry constants
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
GOAL_Y_MIN = (PITCH_WIDTH - 7.32) / 2   # ~30.34m
GOAL_Y_MAX = (PITCH_WIDTH + 7.32) / 2   # ~37.66m
PENALTY_AREA_X = 16.5
GOAL_AREA_X = 5.5


class Gemma4EventDetector:
    """Detect football events using YOLO heuristics + Gemma 4 classification."""

    def __init__(self, analyzer=None):
        """
        Args:
            analyzer: ContextAwareGemma4Analyzer instance (already loaded).
        """
        self.analyzer = analyzer

    def detect_candidate_events(self, all_tracks: list, fps: float) -> list:
        """Detect candidate events from YOLO tracking data using heuristics.

        Returns list of candidate event dicts:
            {frame_idx, event_hint, confidence, details}
        """
        candidates = []

        # 1. Ball near goal line
        candidates.extend(self._detect_goal_proximity(all_tracks, fps))

        # 2. Ball velocity spikes (shots/passes)
        candidates.extend(self._detect_ball_velocity_events(all_tracks, fps))

        # 3. Player convergence (tackles/duels)
        candidates.extend(self._detect_player_convergence_events(all_tracks, fps))

        # 4. Ball direction changes (deflections, saves)
        candidates.extend(self._detect_ball_direction_changes(all_tracks, fps))

        # Deduplicate: merge candidates within 1 second of each other
        candidates.sort(key=lambda c: c['frame_idx'])
        deduped = []
        min_gap = int(fps)
        for c in candidates:
            if deduped and c['frame_idx'] - deduped[-1]['frame_idx'] < min_gap:
                # Keep the one with higher confidence
                if c['confidence'] > deduped[-1]['confidence']:
                    deduped[-1] = c
            else:
                deduped.append(c)

        print(f"  [Events] {len(deduped)} candidate events detected "
              f"(from {len(candidates)} raw)")
        return deduped

    def _detect_goal_proximity(self, all_tracks, fps):
        """Detect when ball is near goal line."""
        candidates = []
        cooldown = 0

        for fi, frame_tracks in enumerate(all_tracks):
            if cooldown > 0:
                cooldown -= 1
                continue

            ball_pos = self._get_ball_position(frame_tracks)
            if ball_pos is None:
                continue

            bx, by = ball_pos

            # Near left goal (x < GOAL_AREA_X)
            near_left = bx < PENALTY_AREA_X and GOAL_Y_MIN - 5 < by < GOAL_Y_MAX + 5
            # Near right goal (x > 105 - GOAL_AREA_X)
            near_right = bx > PITCH_LENGTH - PENALTY_AREA_X and GOAL_Y_MIN - 5 < by < GOAL_Y_MAX + 5

            if near_left or near_right:
                # Count players in area
                players_nearby = 0
                for t in frame_tracks:
                    if t.get('role') in ('player', 'goalkeeper') and t.get('pitch_pos'):
                        px = t['pitch_pos'][0]
                        if (near_left and px < PENALTY_AREA_X + 5) or \
                           (near_right and px > PITCH_LENGTH - PENALTY_AREA_X - 5):
                            players_nearby += 1

                goal_side = "left" if near_left else "right"
                conf = min(0.9, 0.5 + players_nearby * 0.1)

                candidates.append({
                    'frame_idx': fi,
                    'timestamp_s': fi / fps,
                    'event_hint': f"Ball in penalty area ({goal_side} goal), "
                                  f"{players_nearby} players nearby — potential goal attempt",
                    'confidence': conf,
                    'heuristic': 'goal_proximity',
                })
                cooldown = int(fps * 2)  # 2s cooldown

        return candidates

    def _detect_ball_velocity_events(self, all_tracks, fps):
        """Detect ball velocity spikes indicating shots or long passes."""
        candidates = []
        prev_ball = None
        cooldown = 0

        for fi, frame_tracks in enumerate(all_tracks):
            if cooldown > 0:
                cooldown -= 1
                prev_ball = self._get_ball_position(frame_tracks)
                continue

            ball_pos = self._get_ball_position(frame_tracks)
            if ball_pos is None or prev_ball is None:
                prev_ball = ball_pos
                continue

            dist = np.linalg.norm(np.array(ball_pos) - np.array(prev_ball))
            dt = 1.0 / fps
            velocity = dist / dt  # m/s

            if velocity > 15.0:  # > 54 km/h — likely a shot or long pass
                # Determine direction towards goal
                bx = ball_pos[0]
                towards_goal = bx < 20 or bx > 85
                hint = "High-speed ball movement"
                if towards_goal:
                    hint += " towards goal — potential shot"
                else:
                    hint += " — potential long pass or clearance"

                candidates.append({
                    'frame_idx': fi,
                    'timestamp_s': fi / fps,
                    'event_hint': hint,
                    'confidence': min(0.85, 0.5 + velocity / 50),
                    'heuristic': 'ball_velocity',
                    'velocity_ms': velocity,
                })
                cooldown = int(fps * 1.5)

            prev_ball = ball_pos

        return candidates

    def _detect_player_convergence_events(self, all_tracks, fps):
        """Detect when multiple players from different teams converge (tackles/duels)."""
        candidates = []
        cooldown = 0

        for fi in range(0, len(all_tracks), max(1, int(fps / 6))):  # Sample at ~6Hz
            if cooldown > 0:
                cooldown -= 1
                continue

            frame_tracks = all_tracks[fi]
            team_positions = defaultdict(list)

            for t in frame_tracks:
                if t.get('role') in ('player', 'goalkeeper') and t.get('pitch_pos'):
                    team = t.get('team', -1)
                    team_positions[team].append(
                        (t['pitch_pos'], t['track_id'])
                    )

            if len(team_positions) < 2:
                continue

            teams = sorted(team_positions.keys())
            if len(teams) < 2:
                continue

            # Check for inter-team proximity
            max_convergence = 0
            involved = []
            for pos_a, tid_a in team_positions[teams[0]]:
                for pos_b, tid_b in team_positions[teams[1]]:
                    dist = np.linalg.norm(
                        np.array(pos_a) - np.array(pos_b)
                    )
                    if dist < 2.0:  # Within 2 meters
                        max_convergence += 1
                        involved.extend([tid_a, tid_b])

            if max_convergence >= 2:
                candidates.append({
                    'frame_idx': fi,
                    'timestamp_s': fi / fps,
                    'event_hint': f"Multiple players from different teams within 2m "
                                  f"({max_convergence} pairs) — potential tackle/duel/foul",
                    'confidence': min(0.8, 0.4 + max_convergence * 0.15),
                    'heuristic': 'player_convergence',
                    'involved_tracks': list(set(involved)),
                })
                cooldown = int(fps * 2)

        return candidates

    def _detect_ball_direction_changes(self, all_tracks, fps):
        """Detect sudden ball direction changes (deflections, saves, interceptions)."""
        candidates = []
        history = []  # (frame_idx, position)
        cooldown = 0

        for fi, frame_tracks in enumerate(all_tracks):
            if cooldown > 0:
                cooldown -= 1

            ball_pos = self._get_ball_position(frame_tracks)
            if ball_pos is not None:
                history.append((fi, np.array(ball_pos)))

            if len(history) < 3:
                continue

            # Keep only recent history
            if len(history) > 10:
                history = history[-10:]

            if cooldown > 0:
                continue

            # Compare last 3 velocity vectors
            fi1, p1 = history[-3]
            fi2, p2 = history[-2]
            fi3, p3 = history[-1]

            v1 = p2 - p1
            v2 = p3 - p2

            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 < 0.5 or norm2 < 0.5:
                continue

            # Angle between velocity vectors
            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle_deg = np.degrees(np.arccos(cos_angle))

            if angle_deg > 60 and (norm1 > 2 or norm2 > 2):
                candidates.append({
                    'frame_idx': fi,
                    'timestamp_s': fi / fps,
                    'event_hint': f"Ball direction change of {angle_deg:.0f}° — "
                                  f"potential deflection, save, or interception",
                    'confidence': min(0.75, 0.3 + angle_deg / 180),
                    'heuristic': 'direction_change',
                    'angle_deg': angle_deg,
                })
                cooldown = int(fps * 2)

        return candidates

    def detect_and_classify(
        self,
        all_tracks: list,
        clip_info: list,
        fps: float,
        max_events: int = 50,
    ) -> list:
        """Full event detection pipeline: YOLO heuristics → Gemma 4 classification.

        Returns list of classified events.
        """
        # Step 1: Detect candidates from YOLO data
        candidates = self.detect_candidate_events(all_tracks, fps)

        if not candidates:
            return []

        # Limit to top candidates by confidence
        candidates.sort(key=lambda c: c['confidence'], reverse=True)
        candidates = candidates[:max_events]
        candidates.sort(key=lambda c: c['frame_idx'])

        # Step 2: Classify with Gemma 4 (if analyzer available)
        if self.analyzer is None:
            print("  [Events] No Gemma 4 analyzer — returning heuristic events only")
            return [self._heuristic_to_event(c) for c in candidates]

        events = []
        for i, candidate in enumerate(candidates):
            fi = candidate['frame_idx']
            print(f"  [Events] Classifying event {i + 1}/{len(candidates)} "
                  f"(frame {fi}, t={candidate['timestamp_s']:.1f}s)...", end=" ")

            t0 = time.time()

            # Build YOLO context
            frame_tracks = all_tracks[fi] if fi < len(all_tracks) else []
            yolo_context = self.analyzer.build_yolo_context(frame_tracks, fps, fi)

            # Extract frame image
            image = self.analyzer._extract_single_frame(fi, clip_info)
            if image is None:
                print("SKIP (no frame)")
                events.append(self._heuristic_to_event(candidate))
                continue

            # Classify with Gemma 4
            result = self.analyzer.analyze_event(
                image, candidate['event_hint'], yolo_context
            )
            elapsed = time.time() - t0

            if result and not result.get('raw_response'):
                event = {
                    'frame_idx': fi,
                    'timestamp_s': candidate['timestamp_s'],
                    'event_type': result.get('event_type', 'other'),
                    'confidence': result.get('confidence', candidate['confidence']),
                    'description': result.get('description', candidate['event_hint']),
                    'involved_players': result.get('involved_players', []),
                    'attacking_team': result.get('attacking_team'),
                    'outcome': result.get('outcome'),
                    'source': 'gemma4',
                    'heuristic': candidate['heuristic'],
                }
                events.append(event)
                print(f"{elapsed:.1f}s → {event['event_type']}")
            else:
                events.append(self._heuristic_to_event(candidate))
                print(f"{elapsed:.1f}s → fallback to heuristic")

        return events

    @staticmethod
    def _get_ball_position(frame_tracks):
        """Get ball position from frame tracks."""
        for t in frame_tracks:
            if t.get('class', 0) != 0 or t.get('role') == 'ball':
                if t.get('pitch_pos'):
                    return t['pitch_pos']
        return None

    @staticmethod
    def _heuristic_to_event(candidate):
        """Convert a heuristic candidate to an event dict (fallback)."""
        # Map heuristics to event types
        heuristic_map = {
            'goal_proximity': 'goal_attempt',
            'ball_velocity': 'pass',
            'player_convergence': 'tackle',
            'direction_change': 'interception',
        }
        return {
            'frame_idx': candidate['frame_idx'],
            'timestamp_s': candidate['timestamp_s'],
            'event_type': heuristic_map.get(candidate['heuristic'], 'other'),
            'confidence': candidate['confidence'],
            'description': candidate['event_hint'],
            'involved_players': candidate.get('involved_tracks', []),
            'source': 'heuristic',
            'heuristic': candidate['heuristic'],
        }
