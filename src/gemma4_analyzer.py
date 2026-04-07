"""
Gemma 4 Vision-based Football Video Analyzer.

Replaces the YOLO-based detection pipeline with Gemma 4's multimodal capabilities
for end-to-end football video analysis: player detection, team classification,
tactical analysis, and event recognition from drone footage.

Uses the Gemma 4 26B-A4B (Mixture of Experts) model via HuggingFace transformers.
"""
import base64
import json
import time
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class Gemma4FootballAnalyzer:
    """Football video analyzer using Gemma 4 vision model."""

    def __init__(
        self,
        model_path: str = "/home/jovyan/.local/models/gemma4-e4b",
        device: str = "cuda:1",
        frame_sample_interval: int = 30,  # Analyze every Nth frame
        image_size: int = 960,  # Resize frames to this width
    ):
        self.model_path = model_path
        self.device = device
        self.frame_sample_interval = frame_sample_interval
        self.image_size = image_size
        self.model = None
        self.processor = None

    def load_model(self):
        """Load Gemma 4 model and processor."""
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        print(f"Loading Gemma 4 from {self.model_path} on {self.device}...")
        t0 = time.time()

        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            device_map=self.device,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

        elapsed = time.time() - t0
        print(f"Model loaded in {elapsed:.1f}s")

    def _extract_frames(self, video_path: str, max_frames: int = 0):
        """Extract frames from video at specified interval."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames, "
              f"{total_frames/fps:.1f}s")

        frames = []
        frame_indices = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self.frame_sample_interval == 0:
                # Resize for model input
                h, w = frame.shape[:2]
                new_w = self.image_size
                new_h = int(h * new_w / w)
                resized = cv2.resize(frame, (new_w, new_h))
                # Convert BGR to RGB
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                frames.append(pil_img)
                frame_indices.append(frame_idx)

                if max_frames > 0 and len(frames) >= max_frames:
                    break

            frame_idx += 1

        cap.release()
        print(f"Extracted {len(frames)} frames (every {self.frame_sample_interval} frames)")

        return frames, frame_indices, {
            'fps': fps,
            'total_frames': total_frames,
            'width': width,
            'height': height,
            'duration_s': total_frames / fps if fps > 0 else 0,
        }

    def _query_model(self, image: Image.Image, prompt: str, max_tokens: int = 2048):
        """Send an image + prompt to Gemma 4 and get a text response."""
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

        # Decode only the generated tokens (skip input)
        input_len = inputs["input_ids"].shape[-1]
        generated = output_ids[0][input_len:]
        response = self.processor.decode(generated, skip_special_tokens=True)
        return response.strip()

    def analyze_frame_detection(self, image: Image.Image):
        """Detect players, ball, and their positions in a single frame."""
        prompt = """Analyze this top-down drone football (soccer) image. For each player and the ball visible, estimate their approximate position on the pitch.

Return ONLY valid JSON in this exact format (no extra text):
{
  "players": [
    {"id": 1, "team": "A", "x_pct": 45.2, "y_pct": 30.1, "jersey_color": "red"},
    {"id": 2, "team": "B", "x_pct": 55.8, "y_pct": 60.3, "jersey_color": "white"}
  ],
  "ball": {"x_pct": 50.0, "y_pct": 45.0, "visible": true},
  "player_count": {"team_a": 11, "team_b": 11},
  "notes": "brief observation"
}

Where x_pct and y_pct are percentage positions on the pitch (0-100), with 0,0 being top-left of the pitch and 100,100 being bottom-right."""

        response = self._query_model(image, prompt)
        return self._parse_json_response(response)

    def analyze_frame_tactical(self, image: Image.Image):
        """Analyze tactical aspects of a single frame."""
        prompt = """Analyze the tactical situation in this top-down drone football (soccer) image.

Return ONLY valid JSON in this exact format (no extra text):
{
  "formation_team_a": "4-4-2",
  "formation_team_b": "4-3-3",
  "ball_possession": "A",
  "phase_of_play": "attack",
  "pressing_intensity": "medium",
  "defensive_line_height": "medium",
  "team_compactness": {"team_a": "compact", "team_b": "spread"},
  "tactical_observation": "Team A is building up from the back with short passes. Team B is pressing high with 3 forwards."
}

phase_of_play options: attack, defense, transition_attack, transition_defense, set_piece, goal_kick, corner, free_kick, throw_in
pressing_intensity: low, medium, high
defensive_line_height: low, medium, high"""

        response = self._query_model(image, prompt)
        return self._parse_json_response(response)

    def analyze_sequence(self, frames: list, frame_indices: list):
        """Analyze a sequence of frames for events and patterns."""
        # Take first, middle, and last frames for context
        if len(frames) < 3:
            return {}

        indices = [0, len(frames) // 2, -1]
        selected_frames = [frames[i] for i in indices]

        # For now, analyze individual frames and aggregate
        # Future: batch frames together for temporal understanding
        results = []
        for i, frame in enumerate(selected_frames):
            prompt = f"""This is frame {i+1} of 3 from a football match sequence (drone top-down view).

Describe what is happening tactically and identify any notable events (goal attempts, passes, defensive actions).

Return ONLY valid JSON:
{{
  "events": ["list of events observed"],
  "ball_zone": "left_third|center_third|right_third",
  "attacking_team": "A|B|none",
  "tempo": "slow|medium|fast",
  "observation": "brief tactical description"
}}"""
            response = self._query_model(frame, prompt, max_tokens=512)
            parsed = self._parse_json_response(response)
            if parsed:
                results.append(parsed)

        return {
            'sequence_analysis': results,
            'frame_indices': [frame_indices[i] for i in indices],
        }

    def analyze_video(self, video_path: str, output_dir: str, max_frames: int = 10):
        """Run full Gemma 4 analysis on a football video.

        Args:
            video_path: Path to drone football video.
            output_dir: Directory to save results.
            max_frames: Maximum number of frames to analyze (for testing).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.model is None:
            self.load_model()

        print(f"\n{'='*60}")
        print("GEMMA 4 FOOTBALL VIDEO ANALYSIS")
        print(f"{'='*60}")
        t_start = time.time()

        # Step 1: Extract frames
        print("\n[1/4] Extracting frames...")
        frames, frame_indices, video_info = self._extract_frames(
            video_path, max_frames=max_frames
        )

        if not frames:
            print("ERROR: No frames extracted!")
            return {}

        # Step 2: Frame-by-frame detection analysis
        print(f"\n[2/4] Analyzing {len(frames)} frames for player detection...")
        detection_results = []
        for i, (frame, fidx) in enumerate(zip(frames, frame_indices)):
            print(f"  Frame {i+1}/{len(frames)} (video frame {fidx})...", end=" ")
            t0 = time.time()
            result = self.analyze_frame_detection(frame)
            elapsed = time.time() - t0
            print(f"{elapsed:.1f}s")
            if result:
                result['frame_index'] = fidx
                result['timestamp_s'] = fidx / video_info['fps'] if video_info['fps'] > 0 else 0
            detection_results.append(result)

        # Step 3: Tactical analysis on selected frames
        print(f"\n[3/4] Tactical analysis on key frames...")
        # Analyze at start, 25%, 50%, 75%, end
        n = len(frames)
        tactical_indices = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))
        tactical_indices = [i for i in tactical_indices if i < n]

        tactical_results = []
        for i in tactical_indices:
            print(f"  Frame {i+1}/{n} (video frame {frame_indices[i]})...", end=" ")
            t0 = time.time()
            result = self.analyze_frame_tactical(frames[i])
            elapsed = time.time() - t0
            print(f"{elapsed:.1f}s")
            if result:
                result['frame_index'] = frame_indices[i]
                result['timestamp_s'] = frame_indices[i] / video_info['fps'] if video_info['fps'] > 0 else 0
            tactical_results.append(result)

        # Step 4: Sequence analysis
        print(f"\n[4/4] Sequence analysis...")
        sequence = self.analyze_sequence(frames, frame_indices)

        # Compile results
        total_time = time.time() - t_start
        report = {
            'video_info': video_info,
            'video_path': str(video_path),
            'analysis_config': {
                'model': self.model_path,
                'frame_sample_interval': self.frame_sample_interval,
                'image_size': self.image_size,
                'frames_analyzed': len(frames),
                'total_analysis_time_s': round(total_time, 1),
            },
            'detection_results': detection_results,
            'tactical_results': tactical_results,
            'sequence_analysis': sequence,
        }

        # Aggregate stats
        report['summary'] = self._compute_summary(detection_results, tactical_results)

        # Save results
        results_path = output_dir / "gemma4_analysis.json"
        with open(results_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nResults saved: {results_path}")

        # Save sample annotated frames
        self._save_annotated_frames(frames, frame_indices, detection_results, output_dir)

        print(f"\n{'='*60}")
        print(f"ANALYSIS COMPLETE — {total_time:.1f}s total")
        print(f"  Frames analyzed: {len(frames)}")
        print(f"  Avg time/frame: {total_time/len(frames):.1f}s")
        print(f"  Output: {output_dir}")
        print(f"{'='*60}")

        return report

    def _compute_summary(self, detection_results, tactical_results):
        """Compute summary statistics from analysis results."""
        summary = {
            'avg_players_detected': 0,
            'formations_observed': {},
            'phases_of_play': {},
        }

        # Detection summary
        player_counts = []
        for r in detection_results:
            if r and 'players' in r:
                player_counts.append(len(r['players']))
        if player_counts:
            summary['avg_players_detected'] = round(np.mean(player_counts), 1)
            summary['min_players_detected'] = min(player_counts)
            summary['max_players_detected'] = max(player_counts)

        # Tactical summary
        for r in tactical_results:
            if not r:
                continue
            for key in ['formation_team_a', 'formation_team_b']:
                if key in r:
                    formation = r[key]
                    summary['formations_observed'][formation] = \
                        summary['formations_observed'].get(formation, 0) + 1
            if 'phase_of_play' in r:
                phase = r['phase_of_play']
                summary['phases_of_play'][phase] = \
                    summary['phases_of_play'].get(phase, 0) + 1

        return summary

    def _save_annotated_frames(self, frames, frame_indices, detections, output_dir):
        """Save a few annotated frames as images for visual verification."""
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(exist_ok=True)

        for i, (frame, fidx, det) in enumerate(zip(frames, frame_indices, detections)):
            if i >= 5:  # Save max 5 frames
                break

            # Convert PIL to OpenCV
            cv_frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

            if det and 'players' in det:
                h, w = cv_frame.shape[:2]
                for player in det['players']:
                    x_pct = player.get('x_pct', 0)
                    y_pct = player.get('y_pct', 0)
                    px = int(x_pct / 100 * w)
                    py = int(y_pct / 100 * h)

                    # Color by team
                    team = player.get('team', '?')
                    color = (255, 0, 0) if team == 'A' else (0, 0, 255) if team == 'B' else (128, 128, 128)

                    cv2.circle(cv_frame, (px, py), 8, color, 2)
                    label = f"#{player.get('id', '?')} {team}"
                    cv2.putText(cv_frame, label, (px + 10, py - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                # Draw ball
                if det.get('ball', {}).get('visible', False):
                    bx = int(det['ball'].get('x_pct', 0) / 100 * w)
                    by = int(det['ball'].get('y_pct', 0) / 100 * h)
                    cv2.circle(cv_frame, (bx, by), 6, (0, 255, 255), -1)
                    cv2.putText(cv_frame, "BALL", (bx + 8, by - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            frame_path = frames_dir / f"frame_{fidx:06d}.jpg"
            cv2.imwrite(str(frame_path), cv_frame)

        print(f"  Saved {min(5, len(frames))} annotated frames to {frames_dir}")

    @staticmethod
    def _parse_json_response(response: str):
        """Parse JSON from model response, handling common formatting issues."""
        if not response:
            return None

        # Try to extract JSON from the response
        text = response.strip()

        # Remove markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in the text
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

        # Return raw text as fallback
        return {"raw_response": text}

    # ------------------------------------------------------------------
    # Context-aware methods for hybrid pipeline
    # ------------------------------------------------------------------

    def analyze_frame_tactical_with_context(self, image: Image.Image, yolo_context: str):
        """Analyze tactical situation using YOLO position context.

        Enhanced version of analyze_frame_tactical that receives precise
        player positions from YOLO tracking, producing much more accurate
        formation and phase-of-play detection.

        Args:
            image: PIL Image of the frame
            yolo_context: Structured text with YOLO player positions in meters

        Returns:
            Parsed JSON dict with tactical analysis
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
  "tactical_observation": "Detailed tactical description",
  "key_patterns": ["pattern1", "pattern2"]
}}

phase_of_play options: attack, defense, transition_attack, transition_defense, build_up, set_piece, goal_kick, corner, free_kick, throw_in, pressing
Use the actual player positions to determine formation lines."""

        response = self._query_model(image, prompt, max_tokens=1024)
        return self._parse_json_response(response)

    def analyze_event(self, image: Image.Image, event_hint: str, yolo_context: str):
        """Classify and describe a candidate event using visual + tracking context.

        Called by the hybrid pipeline when YOLO heuristics detect a potential
        event (ball near goal, player convergence, etc.).

        Args:
            image: PIL Image of the event frame
            event_hint: Description of why this frame was flagged
            yolo_context: YOLO tracking data as structured text

        Returns:
            Parsed JSON dict with event classification
        """
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


def main():
    """Run Gemma 4 football analysis from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Gemma 4 Football Video Analyzer")
    parser.add_argument("--video", required=True, help="Path to football video")
    parser.add_argument("--output", default="reports/gemma4_output/", help="Output directory")
    parser.add_argument("--model", default="/home/jovyan/.local/models/gemma4-awq",
                        help="Path to Gemma 4 model")
    parser.add_argument("--device", default="cuda:1", help="GPU device")
    parser.add_argument("--interval", type=int, default=30,
                        help="Frame sample interval (analyze every Nth frame)")
    parser.add_argument("--max-frames", type=int, default=10,
                        help="Max frames to analyze (0=all)")
    parser.add_argument("--image-size", type=int, default=960,
                        help="Resize frames to this width")
    args = parser.parse_args()

    analyzer = Gemma4FootballAnalyzer(
        model_path=args.model,
        device=args.device,
        frame_sample_interval=args.interval,
        image_size=args.image_size,
    )

    report = analyzer.analyze_video(
        video_path=args.video,
        output_dir=args.output,
        max_frames=args.max_frames,
    )

    if report:
        summary = report.get('summary', {})
        print(f"\nSummary:")
        print(f"  Avg players detected: {summary.get('avg_players_detected', 'N/A')}")
        print(f"  Formations: {summary.get('formations_observed', {})}")
        print(f"  Phases: {summary.get('phases_of_play', {})}")


if __name__ == "__main__":
    main()
