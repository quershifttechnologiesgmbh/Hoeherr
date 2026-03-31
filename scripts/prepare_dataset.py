"""
Dataset preparation: convert all datasets to unified YOLOv8 format.
Classes: 0=player, 1=goalkeeper, 2=referee, 3=ball
"""
import os
import shutil
import random
from pathlib import Path


def mot_to_yolo(mot_annotations_file, output_labels_dir, img_width, img_height):
    """Convert MOT format (SoccerTrack, SportsMOT) to YOLO format."""
    os.makedirs(output_labels_dir, exist_ok=True)

    with open(mot_annotations_file) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            frame_id = int(parts[0])
            bb_left = float(parts[2])
            bb_top = float(parts[3])
            bb_width = float(parts[4])
            bb_height = float(parts[5])

            cx = (bb_left + bb_width / 2) / img_width
            cy = (bb_top + bb_height / 2) / img_height
            w = bb_width / img_width
            h = bb_height / img_height

            label_file = os.path.join(output_labels_dir, f"frame_{frame_id:06d}.txt")
            with open(label_file, 'a') as lf:
                lf.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def create_dataset_yaml(dataset_dir, num_classes=4):
    """Create data.yaml for YOLOv8 training."""
    yaml_content = f"""path: {dataset_dir}
train: images/train
val: images/val
test: images/test

nc: {num_classes}
names:
  0: player
  1: goalkeeper
  2: referee
  3: ball
"""
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    return yaml_path


def merge_datasets(sources, target_dir, split_ratios=(0.8, 0.1, 0.1)):
    """Merge multiple datasets into a single YOLOv8 dataset."""
    all_samples = []

    for source in sources:
        img_dir = source['images_dir']
        lbl_dir = source['labels_dir']
        for img_file in Path(img_dir).glob("*.jpg"):
            label_file = Path(lbl_dir) / f"{img_file.stem}.txt"
            if label_file.exists():
                all_samples.append((str(img_file), str(label_file)))

    random.shuffle(all_samples)
    n = len(all_samples)
    train_end = int(n * split_ratios[0])
    val_end = train_end + int(n * split_ratios[1])

    splits = {
        'train': all_samples[:train_end],
        'val': all_samples[train_end:val_end],
        'test': all_samples[val_end:]
    }

    for split_name, samples in splits.items():
        img_out = os.path.join(target_dir, 'images', split_name)
        lbl_out = os.path.join(target_dir, 'labels', split_name)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for img_path, lbl_path in samples:
            shutil.copy2(img_path, img_out)
            shutil.copy2(lbl_path, lbl_out)

    print(f"Dataset merged: {len(splits['train'])} train, "
          f"{len(splits['val'])} val, {len(splits['test'])} test")

    return create_dataset_yaml(target_dir)


if __name__ == "__main__":
    print("Dataset preparation utilities loaded.")
    print("Usage: import and call mot_to_yolo(), merge_datasets(), etc.")
