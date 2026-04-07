"""
Context-Aware Gemma 4 Tactical Analyzer.

Uses YOLO tracking data as context to dramatically improve Gemma 4's
tactical analysis accuracy. Instead of pure vision-based analysis,
Gemma 4 receives structured player positions in meters, team assignments,
and roles alongside the video frame.
"""
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class ContextAwareGemma4Analyzer:
    """Gemma 4 tactical analyzer that uses YOLO tracking context."""

    def __init__(
        self,
        model_path: str = "/home/jovyan/.local/models/gemma4-e4b",
        device: str = "cuda:1",
        image_size: int = 960,
    ):
        self.model_path = model_path
        self.device = device
        self.image_size = image_size
        self.model = None
        self.processor = None

    def load_model(self):
        """Load Gemma 4 model and processor."""
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        print(f"  [Gemma4] Loading model from {self.model_path} on {self.device}...")
        t0 = time.time()

        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            device_map=self.device,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        print(f"  [Gemma4] Model loaded in {time.time() - t0:.1f}s")

    def _query_model(self, image: Image.Image, prompt: str, max_tokens: int = 2048):
        """Send image + prompt to Gemma 4."""
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        input_len = inputs["input_ids"].shape[-1]
        generated = output_ids[0][input_len:]
        response = self.processor.decode(generated, skip_special_tokens=True)
        return response.strip()

    def select_key_frames(
        self,
        all_tracks: list,
        fps: float,
        interval: int = 90,
        max_frames: int = 0,
    ) -> list:
        """Select key frames for Gemma 4 analysis using YOLO tracking intelligence.

        Instead of fixed intervals, also selects frames where:
        - Ball velocity has a spike (pass/shot)
        - Team compactness changes significantly (formation shift)
        - Player convergence occurs (potential duel/event)

        Returns list of frame indices.
        """
        total_frames = len(all_tracks)
        if total_frames == 0:
            return []

        # Start with regular interval frames
        regular_frames = set(range(0, total_frames, interval))

        # Add first and last frame
        regular_frames.add(0)
        regular_frames.add(total_frames - 1)

        # Detect ball velocity spikes
        ball_velocity_frames = self._detect_ball_velocity_spikes(all_tracks, fps)
        regular_frames.update(ball_velocity_frames)

        # Detect compactness changes
        compactness_change_frames = self._detect_compactness_changes(all_tracks)
        regular_frames.update(compactness_change_frames)

        # Detect player convergence events
        convergence_frames = self._detect_player_convergence(all_tracks)
        regular_frames.update(convergence_frames)

        # Sort and limit
        key_frames = sorted(regular_frames)

        # Enforce minimum spacing (don't analyze frames < 15 apart)
        min_spacing = max(15, int(fps * 0.5))
        filtered = [key_frames[0]]
        for f in key_frames[1:]:
            if f - filtered[-1] >= min_spacing:
                filtered.append(f)
        key_frames = filtered

        if max_frames > 0 and len(key_frames) > max_frames:
            # Prioritize: keep first, last, and evenly spaced subset
            step = len(key_frames) / max_frames
            indices = [int(i * step) for i in range(max_frames)]
            key_frames = [key_frames[i] for i in indices]

        print(f"  [Gemma4] Selected {len(key_frames)} key frames "
              f"(interval={interval}, ball_spikes={len(ball_velocity_frames)}, "
              f"compactness_changes={len(compactness_change_frames)}, "
              f"convergences={len(convergence_frames)})")

        return key_frames

    def _detect_ball_velocity_spikes(self, all_tracks, fps, threshold_factor=2.0):
        """Detect frames where ball velocity spikes (pass/shot events)."""
        ball_positions = []
        for fi, frame_tracks in enumerate(all_tracks):
            ball_pos = None
            for t in frame_tracks:
                if t.get('class', 0) != 0 or t.get('role') == 'ball':
                    if t.get('pitch_pos'):
                        ball_pos = t['pitch_pos']
                        break
            ball_positions.append((fi, ball_pos))

        if len(ball_positions) < 3:
            return set()

        # Compute velocities
        velocities = []
        for i in range(1, len(ball_positions)):
            fi_prev, pos_prev = ball_positions[i - 1]
            fi_curr, pos_curr = ball_positions[i]
            if pos_prev is not None and pos_curr is not None:
                dist = np.linalg.norm(
                    np.array(pos_curr) - np.array(pos_prev)
                )
                dt = (fi_curr - fi_prev) / fps
                vel = dist / dt if dt > 0 else 0
                velocities.append((fi_curr, vel))

        if not velocities:
            return set()

        # Find spikes above threshold
        vel_values = [v for _, v in velocities]
        mean_vel = np.mean(vel_values)
        std_vel = np.std(vel_values) if len(vel_values) > 1 else 1.0
        threshold = mean_vel + threshold_factor * std_vel

        spike_frames = set()
        for fi, vel in velocities:
            if vel > threshold and vel > 5.0:  # At least 5 m/s
                spike_frames.add(fi)

        return spike_frames

    def _detect_compactness_changes(self, all_tracks, window=30, threshold=0.3):
        """Detect frames where team compactness changes significantly."""
        change_frames = set()

        # Sample compactness every 'window' frames
        compactness_history = []
        for fi in range(0, len(all_tracks), window):
            team_positions = defaultdict(list)
            for t in all_tracks[fi]:
                if t.get('pitch_pos') and t.get('team') is not None:
                    team_positions[t['team']].append(t['pitch_pos'])

            total_area = 0
            for team_id, positions in team_positions.items():
                if len(positions) >= 3:
                    pts = np.array(positions)
                    hull_area = self._convex_hull_area(pts)
                    total_area += hull_area

            compactness_history.append((fi, total_area))

        # Detect significant changes
        for i in range(1, len(compactness_history)):
            fi_prev, area_prev = compactness_history[i - 1]
            fi_curr, area_curr = compactness_history[i]
            if area_prev > 0:
                change = abs(area_curr - area_prev) / area_prev
                if change > threshold:
                    change_frames.add(fi_curr)

        return change_frames

    def _detect_player_convergence(self, all_tracks, proximity_threshold=3.0, min_players=3):
        """Detect frames where multiple players converge (duels, set pieces)."""
        convergence_frames = set()

        # Sample every 30 frames for efficiency
        for fi in range(0, len(all_tracks), 30):
            positions = []
            for t in all_tracks[fi]:
                if t.get('pitch_pos') and t.get('role') in ('player', 'goalkeeper'):
                    positions.append(t['pitch_pos'])

            if len(positions) < min_players:
                continue

            # Count how many players are within proximity_threshold of each other
            pts = np.array(positions)
            max_cluster = 0
            for i in range(len(pts)):
                nearby = np.sum(
                    np.linalg.norm(pts - pts[i], axis=1) < proximity_threshold
                )
                max_cluster = max(max_cluster, nearby)

            if max_cluster >= min_players:
                convergence_frames.add(fi)

        return convergence_frames

    @staticmethod
    def _convex_hull_area(points):
        """Compute convex hull area for a set of 2D points."""
        if len(points) < 3:
            return 0.0
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(points)
            return hull.volume  # 2D: volume = area
        except Exception:
            return 0.0

    def build_yolo_context(self, frame_tracks: list, fps: float, frame_idx: int) -> str:
        """Build text context from YOLO tracking data for a single frame.

        Creates a structured text description of player positions, teams,
        roles, and ball position that Gemma 4 can use alongside the image.
        """
        timestamp = frame_idx / fps
        lines = [
            f"YOLO TRACKING CONTEXT (t={timestamp:.1f}s, frame={frame_idx}):",
            f"Pitch: 105m x 68m (FIFA standard)",
            "",
        ]

        team_players = defaultdict(list)
        ball_info = None
        referees = []

        for t in frame_tracks:
            role = t.get('role', 'player')
            pos = t.get('pitch_pos')
            if pos is None:
                continue

            if role == 'ball':
                ball_info = {'pos': pos, 'track_id': t['track_id']}
            elif role == 'referee':
                referees.append({'pos': pos, 'track_id': t['track_id']})
            else:
                team = t.get('team', -1)
                team_players[team].append({
                    'pos': pos,
                    'track_id': t['track_id'],
                    'role': role,
                })

        # Ball position
        if ball_info:
            bx, by = ball_info['pos']
            third = "left" if bx < 35 else "right" if bx > 70 else "center"
            lines.append(f"Ball: ({bx:.1f}m, {by:.1f}m) — {third} third")
        else:
            lines.append("Ball: not detected")
        lines.append("")

        # Team positions
        for team_id in sorted(team_players.keys()):
            team_letter = chr(ord('A') + team_id) if team_id >= 0 else '?'
            players = team_players[team_id]
            lines.append(f"Team {team_letter} ({len(players)} players):")

            # Sort by x position (defensive to attacking)
            players.sort(key=lambda p: p['pos'][0])
            for p in players:
                px, py = p['pos']
                role_tag = f" [{p['role'].upper()}]" if p['role'] != 'player' else ""
                lines.append(f"  #{p['track_id']}: ({px:.1f}m, {py:.1f}m){role_tag}")
            lines.append("")

        # Referees
        if referees:
            lines.append(f"Referees: {len(referees)}")
            for r in referees:
                rx, ry = r['pos']
                lines.append(f"  #{r['track_id']}: ({rx:.1f}m, {ry:.1f}m)")

        return "\n".join(lines)

    def analyze_frame_tactical_with_context(
        self, image: Image.Image, yolo_context: str
    ) -> dict:
        """Analyze tactical situation with YOLO position context.

        This produces significantly better results than pure vision analysis
        because Gemma 4 gets exact meter-positions from YOLO tracking.
        """
        prompt = f"""You are analyzing a top-down drone football (soccer) image. You have PRECISE player position data from computer vision tracking:

{yolo_context}

Using BOTH the image AND the tracking data above, analyze the tactical situation.

Return ONLY valid JSON:
{{
  "formation_team_a": "4-4-2",
  "formation_team_b": "4-3-3",
  "ball_possession": "A",
  "phase_of_play": "attack",
  "pressing_intensity": "low|medium|high",
  "defensive_line_height": "low|medium|high",
  "offside_line_m": 35.0,
  "team_a_compactness": "compact|moderate|spread",
  "team_b_compactness": "compact|moderate|spread",
  "tactical_observation": "Detailed tactical description of what is happening",
  "key_patterns": ["list", "of", "observed", "patterns"]
}}

phase_of_play: attack, defense, transition_attack, transition_defense, build_up, set_piece, goal_kick, corner, free_kick, throw_in, pressing
Be specific about formations — use the actual player positions to determine lines."""

        response = self._query_model(image, prompt, max_tokens=1024)
        return self._parse_json_response(response)

    def analyze_event(
        self, image: Image.Image, event_hint: str, yolo_context: str
    ) -> dict:
        """Classify and describe a candidate event detected by YOLO heuristics."""
        prompt = f"""You are analyzing a football moment from drone footage. The computer vision system detected a potential event:

EVENT HINT: {event_hint}

{yolo_context}

Look at the image and the tracking data. Classify this event and describe what happened.

Return ONLY valid JSON:
{{
  "event_type": "goal_attempt|pass|cross|tackle|foul|corner|free_kick|goal_kick|throw_in|offside|substitution|other",
  "confidence": 0.85,
  "description": "Detailed description of the event",
  "involved_players": [9, 5],
  "attacking_team": "A|B",
  "outcome": "successful|unsuccessful|unclear"
}}"""

        response = self._query_model(image, prompt, max_tokens=512)
        return self._parse_json_response(response)

    def analyze_key_frames(
        self,
        key_frames: list,
        all_tracks: list,
        clip_info: list,
        fps: float,
    ) -> dict:
        """Analyze all selected key frames with YOLO context.

        Returns dict with tactical_timeline and summary.
        """
        # Build frame-to-clip mapping for video reading
        frame_to_clip = {}
        for ci, info in enumerate(clip_info):
            offset = info['frame_offset']
            for local_f in range(info['n_frames']):
                frame_to_clip[offset + local_f] = (ci, local_f)

        tactical_timeline = []
        total_frames = len(key_frames)

        for idx, frame_idx in enumerate(key_frames):
            if frame_idx >= len(all_tracks):
                continue

            print(f"  [Gemma4] Analyzing frame {idx + 1}/{total_frames} "
                  f"(frame {frame_idx}, t={frame_idx / fps:.1f}s)...", end=" ")
            t0 = time.time()

            # Build YOLO context for this frame
            frame_tracks = all_tracks[frame_idx]
            yolo_context = self.build_yolo_context(frame_tracks, fps, frame_idx)

            # Extract actual video frame
            image = self._extract_single_frame(frame_idx, clip_info)
            if image is None:
                print("SKIP (no frame)")
                continue

            # Analyze with context
            result = self.analyze_frame_tactical_with_context(image, yolo_context)
            elapsed = time.time() - t0

            if result:
                result['frame_index'] = frame_idx
                result['timestamp_s'] = round(frame_idx / fps, 2)
                tactical_timeline.append(result)

            print(f"{elapsed:.1f}s")

        # Compile summary
        formations_a = defaultdict(int)
        formations_b = defaultdict(int)
        phases = defaultdict(int)

        for entry in tactical_timeline:
            if not entry:
                continue
            fa = entry.get('formation_team_a')
            fb = entry.get('formation_team_b')
            phase = entry.get('phase_of_play')
            if fa:
                formations_a[fa] += 1
            if fb:
                formations_b[fb] += 1
            if phase:
                phases[phase] += 1

        return {
            'tactical_timeline': tactical_timeline,
            'summary': {
                'frames_analyzed': len(tactical_timeline),
                'formations_team_a': dict(formations_a),
                'formations_team_b': dict(formations_b),
                'phases_of_play': dict(phases),
            },
        }

    def _extract_single_frame(self, frame_idx: int, clip_info: list) -> Image.Image:
        """Extract a single frame from the correct video clip."""
        # Find which clip contains this frame
        target_clip = None
        local_frame = frame_idx

        for ci, info in enumerate(clip_info):
            offset = info['frame_offset']
            n_frames = info['n_frames']
            if offset <= frame_idx < offset + n_frames:
                target_clip = info
                local_frame = frame_idx - offset
                break

        if target_clip is None:
            return None

        cap = cv2.VideoCapture(target_clip['path'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, local_frame)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None

        # Resize for Gemma 4
        h, w = frame.shape[:2]
        new_w = self.image_size
        new_h = int(h * new_w / w)
        resized = cv2.resize(frame, (new_w, new_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    @staticmethod
    def _parse_json_response(response: str) -> dict:
        """Parse JSON from model response."""
        import json

        if not response:
            return None

        text = response.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

        return {"raw_response": text}
