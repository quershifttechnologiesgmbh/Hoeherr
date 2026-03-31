"""
Evaluate MOT tracking performance.
Usage: python scripts/eval/evaluate_tracking.py --gt ground_truth.txt --pred predictions.txt
"""
import argparse
import numpy as np


def evaluate_mot(gt_file, pred_file):
    """Evaluate MOT performance against ground truth."""
    import motmetrics as mm

    from src.tracking.bot_sort_tracker import load_mot_format

    gt = load_mot_format(gt_file)
    pred = load_mot_format(pred_file)

    acc = mm.MOTAccumulator(auto_id=True)

    for frame_id in sorted(set(list(gt.keys()) + list(pred.keys()))):
        gt_boxes = gt.get(frame_id, [])
        pred_boxes = pred.get(frame_id, [])

        gt_ids = [b['id'] for b in gt_boxes]
        pred_ids = [b['id'] for b in pred_boxes]

        gt_centers = np.array([[b['x'] + b['w'] / 2, b['y'] + b['h'] / 2] for b in gt_boxes]) if gt_boxes else np.empty((0, 2))
        pred_centers = np.array([[b['x'] + b['w'] / 2, b['y'] + b['h'] / 2] for b in pred_boxes]) if pred_boxes else np.empty((0, 2))

        if len(gt_centers) > 0 and len(pred_centers) > 0:
            distances = mm.distances.norm2squared_matrix(gt_centers, pred_centers)
        else:
            distances = np.empty((len(gt_centers), len(pred_centers)))

        acc.update(gt_ids, pred_ids, distances)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=[
        'mota', 'idf1', 'num_switches',
        'mostly_tracked', 'mostly_lost',
        'num_false_positives', 'num_misses'
    ])
    print(summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True, help="Ground truth MOT file")
    parser.add_argument("--pred", required=True, help="Prediction MOT file")
    args = parser.parse_args()
    evaluate_mot(args.gt, args.pred)


if __name__ == "__main__":
    main()
