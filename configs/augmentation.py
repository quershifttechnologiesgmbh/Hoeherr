"""
Augmentation strategy for bridging the domain gap:
Broadcast footage (sideline) -> Drone footage (top-down)
"""
import albumentations as A

train_augmentation = A.Compose([
    A.RandomResizedCrop(height=640, width=640, scale=(0.5, 1.0), ratio=(0.8, 1.2)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    A.Rotate(limit=180, p=0.3),
    A.Perspective(scale=(0.02, 0.08), p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=0.4),
    A.CLAHE(clip_limit=4.0, p=0.2),
    A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=0.2),
    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.1),
    A.GaussianBlur(blur_limit=(3, 7), p=0.2),
    A.GaussNoise(var_limit=(10, 50), p=0.2),
    A.ImageCompression(quality_lower=60, quality_upper=90, p=0.2),
    A.SmallestMaxSize(max_size=640),
], bbox_params=A.BboxParams(
    format='yolo',
    min_visibility=0.3,
    label_fields=['class_labels']
))

val_augmentation = A.Compose([
    A.SmallestMaxSize(max_size=640),
], bbox_params=A.BboxParams(
    format='yolo',
    min_visibility=0.3,
    label_fields=['class_labels']
))
