"""
Hybrid Pipeline Orchestrator.

Coordinates the YOLO + Gemma 4 analysis pipeline:
  Phase 1 (sequential): YOLO Detection → Tracking → Homography → Teams → Kalman
  Phase 2 (parallel):   Thread A: YOLO post-processing (roles, video, metrics)
                         Thread B: Gemma 4 tactical + event analysis (with YOLO context)
  Phase 3:              Merge results → Unified JSON + Enhanced HTML Report
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import YOLO pipeline steps from existing run_analysis.py
from scripts.run_analysis import (
    run_detection_tracking,
    apply_topk_filter,
    compute_homography_from_keypoints,
    classify_and_transform,
    apply_pitch_boundary_filter,
    apply_kalman_smoothing,
    consolidate_tracks,
    detect_roles,
    render_annotated_video,
    compute_metrics,
    generate_report,
)


class HybridPipelineOrchestrator:
    """Orchestrates YOLO + Gemma 4 hybrid analysis pipeline."""

    def __init__(
        self,
        # YOLO settings
        model_path: str = None,
        videos: list = None,
        keypoints_path: str = None,
        tracker_config: str = "configs/botsort.yaml",
        yolo_device: str = "cuda:2",
        imgsz: int = 1280,
        conf: float = 0.25,
        fps: float = 30.0,
        output_scale: float = 0.5,
        output_dir: str = "reports/hybrid_output/",
        # Gemma 4 settings
        gemma4_model: str = "/home/jovyan/.local/models/gemma4-e4b",
        gemma4_device: str = "cuda:1",
        gemma4_interval: int = 90,  # Analyze every Nth frame with Gemma 4
        gemma4_image_size: int = 960,
        # Pipeline control
        no_gemma4: bool = False,
        max_gemma4_frames: int = 0,  # 0 = all selected keyframes
        # Tiered detection
        use_tiered_detection: bool = False,
        detection_scale: int = 1920,  # First pass resolution
    ):
        self.model_path = model_path
        self.videos = videos or []
        self.keypoints_path = keypoints_path
        self.tracker_config = tracker_config
        self.yolo_device = yolo_device
        self.imgsz = imgsz
        self.conf = conf
        self.fps = fps
        self.output_scale = output_scale
        self.output_dir = output_dir

        self.gemma4_model = gemma4_model
        self.gemma4_device = gemma4_device
        self.gemma4_interval = gemma4_interval
        self.gemma4_image_size = gemma4_image_size
        self.no_gemma4 = no_gemma4
        self.max_gemma4_frames = max_gemma4_frames

        self.use_tiered_detection = use_tiered_detection
        self.detection_scale = detection_scale

        # Pipeline state
        self.all_tracks = None
        self.clip_info = None
        self.H = None
        self.metrics = None
        self.gemma4_results = None
        self.events = None
        self.merged_results = None

    def _make_args_namespace(self):
        """Create an argparse-compatible namespace for legacy step functions."""
        import argparse
        args = argparse.Namespace(
            model=self.model_path,
            videos=self.videos,
            keypoints=self.keypoints_path,
            tracker=self.tracker_config,
            device=self.yolo_device,
            imgsz=self.imgsz,
            conf=self.conf,
            fps=self.fps,
            output_scale=self.output_scale,
            output=self.output_dir,
        )
        return args

    def run(self):
        """Execute the full hybrid pipeline."""
        os.makedirs(self.output_dir, exist_ok=True)
        args = self._make_args_namespace()

        print("=" * 70)
        print("HYBRID YOLO + GEMMA 4 FOOTBALL ANALYSIS PIPELINE")
        print("=" * 70)
        print(f"  YOLO device:   {self.yolo_device}")
        print(f"  Gemma 4:       {'DISABLED' if self.no_gemma4 else self.gemma4_device}")
        print(f"  Output:        {self.output_dir}")
        print(f"  Videos:        {len(self.videos)} clip(s)")
        print()

        t_pipeline_start = time.time()

        # ===================================================================
        # PHASE 1: YOLO Sequential Pipeline (Detection → Tracking → Homography → Teams → Kalman)
        # ===================================================================
        print("\n" + "=" * 70)
        print("PHASE 1: YOLO Sequential Pipeline")
        print("=" * 70)

        t_phase1 = time.time()

        # Step 1: Detection + Tracking
        self.all_tracks, self.clip_info = run_detection_tracking(args)

        # Step 1a: Top-K detection filter
        print("\n  Applying world model constraints...")
        self.all_tracks = apply_topk_filter(self.all_tracks)

        # Step 2: Homography
        self.H = compute_homography_from_keypoints(self.keypoints_path)

        # Step 3: Team Classification + Position Transform
        self.all_tracks = classify_and_transform(
            self.all_tracks, self.clip_info, self.H, args
        )

        # Step 3a: Pitch boundary filter
        self.all_tracks = apply_pitch_boundary_filter(self.all_tracks)

        # Step 3b: Kalman filter smoothing
        apply_kalman_smoothing(self.all_tracks, self.fps)

        # Step 3c: Track consolidation
        self.all_tracks = consolidate_tracks(self.all_tracks, self.fps)

        phase1_time = time.time() - t_phase1
        print(f"\n  Phase 1 complete: {phase1_time:.1f}s")

        # ===================================================================
        # PHASE 2: Parallel Fork
        # ===================================================================
        print("\n" + "=" * 70)
        print("PHASE 2: Parallel Processing")
        print("=" * 70)

        t_phase2 = time.time()

        if self.no_gemma4:
            # Sequential YOLO-only path
            self._run_yolo_postprocessing(args)
        else:
            # Parallel: YOLO post-processing + Gemma 4
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_yolo = executor.submit(self._run_yolo_postprocessing, args)
                future_gemma4 = executor.submit(self._run_gemma4_analysis)

                # Collect results
                for future in as_completed([future_yolo, future_gemma4]):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"  ERROR in parallel task: {e}")
                        import traceback
                        traceback.print_exc()

        phase2_time = time.time() - t_phase2
        print(f"\n  Phase 2 complete: {phase2_time:.1f}s")

        # ===================================================================
        # PHASE 3: Merge + Report
        # ===================================================================
        print("\n" + "=" * 70)
        print("PHASE 3: Merge + Report")
        print("=" * 70)

        t_phase3 = time.time()

        if not self.no_gemma4 and self.gemma4_results:
            from src.hybrid.result_merger import HybridResultMerger
            merger = HybridResultMerger()
            self.merged_results = merger.merge(
                yolo_metrics=self.metrics,
                gemma4_results=self.gemma4_results,
                events=self.events or [],
                all_tracks=self.all_tracks,
                fps=self.fps,
            )

            # Save merged results
            merged_path = os.path.join(self.output_dir, "hybrid_analysis.json")
            with open(merged_path, 'w') as f:
                json.dump(self.merged_results, f, indent=2, default=str)
            print(f"  Hybrid results saved: {merged_path}")

            # Generate enhanced report
            from src.hybrid.report_generator import HybridReportGenerator
            report_gen = HybridReportGenerator()
            report_path = report_gen.generate(
                merged_results=self.merged_results,
                yolo_metrics=self.metrics,
                output_dir=self.output_dir,
            )
            print(f"  Enhanced report saved: {report_path}")
        else:
            # YOLO-only: save standard outputs
            self._save_yolo_outputs(args)

        phase3_time = time.time() - t_phase3
        total_time = time.time() - t_pipeline_start

        print(f"\n{'=' * 70}")
        print(f"HYBRID PIPELINE COMPLETE")
        print(f"{'=' * 70}")
        print(f"  Phase 1 (YOLO sequential): {phase1_time:.1f}s")
        print(f"  Phase 2 (parallel):        {phase2_time:.1f}s")
        print(f"  Phase 3 (merge + report):  {phase3_time:.1f}s")
        print(f"  Total:                     {total_time:.1f}s")
        print(f"\n  Outputs in {self.output_dir}:")
        print(f"    - annotated_tracking.mp4")
        print(f"    - tracks.json")
        print(f"    - metrics.json")
        if not self.no_gemma4:
            print(f"    - hybrid_analysis.json")
            print(f"    - hybrid_report.html")
        else:
            print(f"    - analysis_report.html")

        return self.merged_results if self.merged_results else self.metrics

    def _run_yolo_postprocessing(self, args):
        """Thread A: YOLO role detection, video rendering, metrics."""
        print("\n  [Thread A] YOLO Post-Processing...")

        # Step 3d: Role detection
        self.all_tracks = detect_roles(self.all_tracks, self.fps)

        # Step 4: Render annotated video
        render_annotated_video(self.all_tracks, self.clip_info, self.H, args)

        # Step 5: Compute metrics
        self.metrics = compute_metrics(self.all_tracks, args)

        # Save tracks + metrics
        self._save_yolo_outputs(args)

        print("  [Thread A] YOLO Post-Processing complete.")

    def _run_gemma4_analysis(self):
        """Thread B: Gemma 4 tactical + event analysis with YOLO context."""
        print("\n  [Thread B] Gemma 4 Tactical Analysis...")

        try:
            from src.hybrid.gemma4_tactical import ContextAwareGemma4Analyzer
            from src.hybrid.event_detector import Gemma4EventDetector

            # Initialize Gemma 4 analyzer with YOLO context
            tactical = ContextAwareGemma4Analyzer(
                model_path=self.gemma4_model,
                device=self.gemma4_device,
                image_size=self.gemma4_image_size,
            )
            tactical.load_model()

            # Select key frames using YOLO tracking data
            key_frames = tactical.select_key_frames(
                all_tracks=self.all_tracks,
                fps=self.fps,
                interval=self.gemma4_interval,
                max_frames=self.max_gemma4_frames,
            )

            # Analyze each key frame with YOLO context
            self.gemma4_results = tactical.analyze_key_frames(
                key_frames=key_frames,
                all_tracks=self.all_tracks,
                clip_info=self.clip_info,
                fps=self.fps,
            )

            # Event detection
            event_detector = Gemma4EventDetector(analyzer=tactical)
            self.events = event_detector.detect_and_classify(
                all_tracks=self.all_tracks,
                clip_info=self.clip_info,
                fps=self.fps,
            )

            print(f"  [Thread B] Gemma 4 analysis complete: "
                  f"{len(self.gemma4_results.get('tactical_timeline', []))} tactical frames, "
                  f"{len(self.events)} events")

        except Exception as e:
            print(f"  [Thread B] Gemma 4 ERROR: {e}")
            import traceback
            traceback.print_exc()
            self.gemma4_results = {}
            self.events = []

    def _save_yolo_outputs(self, args):
        """Save YOLO tracks and metrics JSON."""
        # Save tracks
        tracks_path = os.path.join(self.output_dir, "tracks.json")
        flat_tracks = []
        for ft in self.all_tracks:
            for t in ft:
                flat_tracks.append(t)
        with open(tracks_path, 'w') as f:
            json.dump(flat_tracks, f, indent=2)
        print(f"  Tracks saved: {tracks_path} ({len(flat_tracks)} detections)")

        # Save metrics (strip heatmap data)
        if self.metrics:
            metrics_path = os.path.join(self.output_dir, "metrics.json")
            metrics_save = json.loads(json.dumps(self.metrics))
            for pid in metrics_save.get('individual', {}):
                metrics_save['individual'][pid].pop('heatmap_data', None)
            with open(metrics_path, 'w') as f:
                json.dump(metrics_save, f, indent=2)
            print(f"  Metrics saved: {metrics_path}")

            # Standard YOLO report
            generate_report(self.metrics, args)
