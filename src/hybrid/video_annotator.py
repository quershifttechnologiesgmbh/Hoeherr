"""
Enhanced Video Annotator for Hybrid Pipeline.

Extends the standard YOLO annotated video with:
- Tactical HUD (top-left): Phase, Formation, Possession, Pressing
- Event banners (top-center): "GOAL ATTEMPT - #9" with fade-out
- Formation lines on mini-pitch overlay
"""
import cv2
import numpy as np
from collections import defaultdict


class HybridVideoAnnotator:
    """Add tactical overlays to annotated video frames."""

    # Team colors (BGR)
    TEAM_COLORS = {0: (255, 100, 0), 1: (0, 0, 255)}
    GK_COLORS = {0: (255, 200, 0), 1: (100, 100, 255)}
    BALL_COLOR = (0, 255, 255)
    REFEREE_COLOR = (0, 255, 0)
    DEFAULT_COLOR = (180, 180, 180)

    # Phase colors for HUD (BGR)
    PHASE_COLORS_BGR = {
        'attack': (80, 175, 76),
        'defense': (54, 67, 244),
        'transition_attack': (58, 195, 139),
        'transition_defense': (0, 152, 255),
        'build_up': (243, 150, 33),
        'pressing': (34, 87, 255),
        'set_piece': (176, 39, 156),
    }

    def __init__(
        self,
        gemma4_tactical_timeline: list = None,
        events: list = None,
        fps: float = 30.0,
    ):
        """
        Args:
            gemma4_tactical_timeline: List of tactical entries from Gemma 4
            events: List of classified events
            fps: Video FPS
        """
        self.fps = fps

        # Build frame-indexed lookups
        self.tactical_by_frame = {}
        if gemma4_tactical_timeline:
            for entry in sorted(gemma4_tactical_timeline, key=lambda e: e.get('frame_index', 0)):
                self.tactical_by_frame[entry.get('frame_index', 0)] = entry

        self.events_by_frame = {}
        if events:
            for event in events:
                fi = event.get('frame_idx', 0)
                self.events_by_frame[fi] = event

        # Current tactical state (persists between key frames)
        self._current_tactical = {}
        self._active_event = None
        self._event_fade_counter = 0
        self._event_fade_duration = int(fps * 3)  # 3 second fade

    def get_tactical_state(self, frame_idx: int) -> dict:
        """Get the current tactical state for a frame.

        Uses the most recent Gemma 4 analysis up to this frame.
        """
        if frame_idx in self.tactical_by_frame:
            self._current_tactical = self.tactical_by_frame[frame_idx]
        return self._current_tactical

    def get_active_event(self, frame_idx: int) -> tuple:
        """Get active event and its fade alpha for the current frame.

        Returns (event_dict, alpha) or (None, 0).
        """
        if frame_idx in self.events_by_frame:
            self._active_event = self.events_by_frame[frame_idx]
            self._event_fade_counter = self._event_fade_duration

        if self._active_event and self._event_fade_counter > 0:
            alpha = self._event_fade_counter / self._event_fade_duration
            self._event_fade_counter -= 1
            return self._active_event, alpha

        return None, 0

    def draw_tactical_hud(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Draw tactical HUD overlay on top-left of frame."""
        tactical = self.get_tactical_state(frame_idx)
        if not tactical:
            return frame

        h, w = frame.shape[:2]
        hud_w, hud_h = 320, 110
        margin = 10

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (margin, margin),
                       (margin + hud_w, margin + hud_h),
                       (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Text content
        y_offset = margin + 20
        x_offset = margin + 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        color = (255, 255, 255)

        # Phase of play
        phase = tactical.get('phase_of_play', '—')
        phase_color = self.PHASE_COLORS_BGR.get(phase, (200, 200, 200))
        cv2.putText(frame, f"Phase: {phase.replace('_', ' ').upper()}",
                     (x_offset, y_offset), font, font_scale, phase_color, 1)

        # Formation
        y_offset += 20
        form_a = tactical.get('formation_team_a', '—')
        form_b = tactical.get('formation_team_b', '—')
        cv2.putText(frame, f"Formation: A={form_a}  B={form_b}",
                     (x_offset, y_offset), font, font_scale, color, 1)

        # Possession
        y_offset += 20
        poss = tactical.get('ball_possession', '—')
        poss_color = self.TEAM_COLORS.get(0, color) if poss == 'A' else \
                     self.TEAM_COLORS.get(1, color) if poss == 'B' else color
        cv2.putText(frame, f"Possession: Team {poss}",
                     (x_offset, y_offset), font, font_scale, poss_color, 1)

        # Pressing
        y_offset += 20
        pressing = tactical.get('pressing_intensity', '—')
        pressing_colors = {'low': (0, 200, 0), 'medium': (0, 200, 255), 'high': (0, 0, 255)}
        pressing_color = pressing_colors.get(pressing, color)
        cv2.putText(frame, f"Pressing: {pressing.upper()}",
                     (x_offset, y_offset), font, font_scale, pressing_color, 1)

        return frame

    def draw_event_banner(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Draw event banner at top-center with fade-out effect."""
        event, alpha = self.get_active_event(frame_idx)
        if event is None or alpha <= 0:
            return frame

        h, w = frame.shape[:2]
        event_type = event.get('event_type', 'event').replace('_', ' ').upper()
        players = event.get('involved_players', [])
        player_text = " - " + ", ".join(f"#{p}" for p in players) if players else ""
        text = f"{event_type}{player_text}"

        # Calculate text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Banner position (top center)
        banner_w = text_w + 40
        banner_h = text_h + 20
        bx = (w - banner_w) // 2
        by = 50

        # Draw with alpha
        overlay = frame.copy()
        cv2.rectangle(overlay, (bx, by), (bx + banner_w, by + banner_h),
                       (0, 0, 100), -1)
        cv2.rectangle(overlay, (bx, by), (bx + banner_w, by + banner_h),
                       (0, 0, 255), 2)
        frame = cv2.addWeighted(overlay, alpha * 0.7, frame, 1 - alpha * 0.7, 0)

        # Text (also faded)
        text_color = (255, 255, 255)
        text_x = bx + 20
        text_y = by + text_h + 8
        cv2.putText(frame, text, (text_x, text_y), font, font_scale,
                     tuple(int(c * alpha) for c in text_color), thickness)

        return frame

    def draw_formation_lines(self, pitch_overlay: np.ndarray, frame_tracks: list,
                              pitch_w: int, pitch_h: int) -> np.ndarray:
        """Draw formation lines on mini-pitch overlay.

        Groups players into defensive/midfield/attack lines and draws
        connecting lines between them.
        """
        team_positions = defaultdict(list)
        for t in frame_tracks:
            if t.get('pitch_pos') and t.get('team') is not None:
                if t.get('role') in ('player', 'goalkeeper'):
                    team_positions[t['team']].append(t['pitch_pos'])

        line_colors = {0: (255, 150, 50), 1: (50, 50, 255)}

        for team_id, positions in team_positions.items():
            if len(positions) < 3:
                continue

            color = line_colors.get(team_id, (200, 200, 200))
            pts = np.array(positions)

            # Sort by x to identify lines
            sorted_pts = pts[np.argsort(pts[:, 0])]

            # Convert to mini-pitch coordinates
            mini_pts = []
            for px, py in sorted_pts:
                mx = int(px / 105 * (pitch_w - 4)) + 2
                my = int(py / 68 * (pitch_h - 4)) + 2
                mx = max(2, min(pitch_w - 3, mx))
                my = max(2, min(pitch_h - 3, my))
                mini_pts.append((mx, my))

            # Draw lines between adjacent players (sorted by x)
            for i in range(len(mini_pts) - 1):
                # Only connect players in similar x-zones (same line)
                dx = abs(sorted_pts[i][0] - sorted_pts[i + 1][0])
                if dx < 12:  # Within ~12m = same formation line
                    cv2.line(pitch_overlay, mini_pts[i], mini_pts[i + 1],
                              color, 1, cv2.LINE_AA)

        return pitch_overlay

    def annotate_frame(self, frame: np.ndarray, frame_idx: int,
                        frame_tracks: list) -> np.ndarray:
        """Apply all hybrid annotations to a single frame.

        Should be called AFTER standard YOLO annotations (bboxes, labels, mini-pitch).
        """
        # Tactical HUD
        frame = self.draw_tactical_hud(frame, frame_idx)

        # Event banner
        frame = self.draw_event_banner(frame, frame_idx)

        return frame
