#!/usr/bin/env python3
"""
Hybrid YOLO + Gemma 4 Football Analysis Pipeline — Entry Point.

Extends run_analysis.py with:
  - Gemma 4 tactical analysis with YOLO position context
  - Event detection (heuristic candidates → Gemma 4 classification)
  - Player Re-ID with appearance embeddings + jersey number recognition
  - Duel/occlusion handling with Kalman prediction through proximity events
  - Tiered detection (1920px detection + 4K crops for Re-ID)
  - Enhanced HTML report with timeline, events, tactical insights
  - Annotated video with tactical HUD and event banners

Usage:
  python scripts/run_hybrid_analysis.py \\
    --model models/yolov8x_football.pt \\
    --videos api/uploads/clip1.mp4 api/uploads/clip2.mp4 \\
    --keypoints data/drone_keypoints.json \\
    --output reports/hybrid_output/ \\
    --gemma4-model /home/jovyan/.local/models/gemma4-e4b \\
    --gemma4-device cuda:1 \\
    --yolo-device cuda:2

  # YOLO-only (skip Gemma 4):
  python scripts/run_hybrid_analysis.py ... --no-gemma4
"""
import argparse
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hybrid YOLO + Gemma 4 Football Analysis Pipeline"
    )

    # --- YOLO settings (same as run_analysis.py) ---
    parser.add_argument("--model", required=True, help="Path to YOLO model")
    parser.add_argument("--videos", nargs="+", required=True, help="Input video paths")
    parser.add_argument("--keypoints", required=True, help="Path to drone keypoints JSON")
    parser.add_argument("--tracker", default="configs/botsort.yaml", help="Tracker config")
    parser.add_argument("--yolo-device", default="cuda:2", help="GPU for YOLO")
    parser.add_argument("--imgsz", type=int, default=1920,
                        help="Detection image size (default: 1920 for tiered)")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence")
    parser.add_argument("--output", default="reports/hybrid_output/", help="Output directory")
    parser.add_argument("--fps", type=float, default=30.0, help="Video FPS")
    parser.add_argument("--output-scale", type=float, default=0.5, help="Output video scale")

    # --- Gemma 4 settings ---
    parser.add_argument("--gemma4-model",
                        default="/home/jovyan/.local/models/gemma4-e4b",
                        help="Path to Gemma 4 model")
    parser.add_argument("--gemma4-device", default="cuda:1",
                        help="GPU for Gemma 4")
    parser.add_argument("--gemma4-interval", type=int, default=90,
                        help="Gemma 4 key frame interval (default: every 90 frames = 3s)")
    parser.add_argument("--gemma4-image-size", type=int, default=960,
                        help="Image size for Gemma 4 input")
    parser.add_argument("--max-gemma4-frames", type=int, default=0,
                        help="Max frames for Gemma 4 (0=auto)")
    parser.add_argument("--no-gemma4", action="store_true",
                        help="Disable Gemma 4 — pure YOLO pipeline")

    # --- Tiered detection ---
    parser.add_argument("--tiered-detection", action="store_true",
                        help="Enable tiered detection (1920px + 4K crops)")
    parser.add_argument("--detection-scale", type=int, default=1920,
                        help="First-pass detection resolution")
    parser.add_argument("--crop-interval", type=int, default=5,
                        help="Extract 4K crops every N frames")

    # --- Re-ID ---
    parser.add_argument("--enable-reid", action="store_true",
                        help="Enable appearance-based Re-ID")
    parser.add_argument("--jersey-ocr", action="store_true",
                        help="Enable jersey number OCR (requires PaddleOCR)")
    parser.add_argument("--reid-interval", type=int, default=150,
                        help="Jersey number check interval (frames)")

    # --- Duel handling ---
    parser.add_argument("--enable-duel-handler", action="store_true",
                        help="Enable duel/occlusion handler")
    parser.add_argument("--duel-proximity", type=float, default=2.0,
                        help="Duel detection proximity threshold (meters)")

    return parser.parse_args()


def main():
    args = parse_args()

    os.chdir(PROJECT_ROOT)
    os.makedirs(args.output, exist_ok=True)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output dir:   {args.output}")
    print(f"Model:        {args.model}")
    print(f"Videos:       {args.videos}")
    print(f"YOLO device:  {args.yolo_device}")
    print(f"Gemma 4:      {'DISABLED' if args.no_gemma4 else args.gemma4_device}")
    print(f"Tiered det:   {'ON' if args.tiered_detection else 'OFF'}")
    print(f"Re-ID:        {'ON' if args.enable_reid else 'OFF'}")
    print(f"Duel handler: {'ON' if args.enable_duel_handler else 'OFF'}")
    print()

    from src.hybrid.orchestrator import HybridPipelineOrchestrator

    orchestrator = HybridPipelineOrchestrator(
        # YOLO
        model_path=args.model,
        videos=args.videos,
        keypoints_path=args.keypoints,
        tracker_config=args.tracker,
        yolo_device=args.yolo_device,
        imgsz=args.imgsz,
        conf=args.conf,
        fps=args.fps,
        output_scale=args.output_scale,
        output_dir=args.output,
        # Gemma 4
        gemma4_model=args.gemma4_model,
        gemma4_device=args.gemma4_device,
        gemma4_interval=args.gemma4_interval,
        gemma4_image_size=args.gemma4_image_size,
        no_gemma4=args.no_gemma4,
        max_gemma4_frames=args.max_gemma4_frames,
        # Tiered detection
        use_tiered_detection=args.tiered_detection,
        detection_scale=args.detection_scale,
    )

    results = orchestrator.run()

    if results:
        print("\nPipeline completed successfully.")
    else:
        print("\nPipeline completed with no results.")


if __name__ == "__main__":
    main()
