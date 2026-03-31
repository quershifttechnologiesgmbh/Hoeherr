#!/bin/bash
# Launch all 3 training experiments in parallel on GPUs 1, 2, 3
# Must be run with clean environment to avoid NVIDIA_VISIBLE_DEVICES=void

cd /home/jovyan/work/hoeherr-football-analysis/hoeherr-ai

export NVIDIA_VISIBLE_DEVICES=all
export PATH=/opt/conda/bin:/usr/local/cuda-12.6/bin:/usr/local/bin:/usr/bin:/bin

# Clear old logs
> runs/train_v2.log
> runs/train_v3.log
> runs/train_v4.log

echo "Launching v2_combined on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python -u scripts/train/train_v2_combined.py > runs/train_v2.log 2>&1 &
PID_V2=$!
echo "  v2 PID: $PID_V2"

echo "Launching v3_unfrozen on GPU 2..."
CUDA_VISIBLE_DEVICES=2 python -u scripts/train/train_v3_unfrozen.py > runs/train_v3.log 2>&1 &
PID_V3=$!
echo "  v3 PID: $PID_V3"

echo "Launching v4_yolov8x on GPU 3..."
CUDA_VISIBLE_DEVICES=3 python -u scripts/train/train_v4_yolov8x.py > runs/train_v4.log 2>&1 &
PID_V4=$!
echo "  v4 PID: $PID_V4"

echo ""
echo "All training runs launched. PIDs: v2=$PID_V2 v3=$PID_V3 v4=$PID_V4"
echo "Monitor with: tail -f runs/train_v2.log runs/train_v3.log runs/train_v4.log"
echo ""

# Wait for all to finish
wait $PID_V2 $PID_V3 $PID_V4
echo "All training runs complete!"
