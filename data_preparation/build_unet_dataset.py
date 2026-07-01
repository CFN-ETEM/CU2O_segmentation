"""Build U-Net training arrays from image and binary-label TIFF/NPY files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from image_utils import center_crop, fit_to_square, read_image_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", default="data_preparation/HRTEM.tiff", help="Input HRTEM TIFF stack or NPY array.")
    parser.add_argument("--labels", default="outputs/labels/cu2o_mask.tiff", help="Binary label TIFF stack or NPY array.")
    parser.add_argument("--output-dir", default="data/custom", help="Directory for data_tem.npy and label.npy.")
    parser.add_argument("--size", type=int, default=512, help="Square U-Net input size.")
    parser.add_argument("--crop-height", type=int, default=None, help="Optional centered crop height before padding.")
    parser.add_argument("--crop-width", type=int, default=None, help="Optional centered crop width before padding.")
    parser.add_argument("--label-threshold", type=float, default=127, help="Threshold used to binarize labels.")
    parser.add_argument(
        "--repeat-single-label",
        action="store_true",
        help="Repeat one label frame across all image frames. Use only for controlled demonstrations.",
    )
    parser.add_argument(
        "--positive",
        choices=["bright", "dark"],
        default="bright",
        help="Whether selected atoms are bright or dark in the label image.",
    )
    parser.add_argument(
        "--normalize",
        choices=["none", "255", "minmax"],
        default="none",
        help="Optional image normalization. Use 'none' to match the provided example arrays.",
    )
    parser.add_argument("--image-pad-value", type=float, default=255, help="Padding value for image frames.")
    return parser.parse_args()


def normalize_image(frame: np.ndarray, mode: str) -> np.ndarray:
    frame = frame.astype(np.float32)
    if mode == "none":
        return frame
    if mode == "255":
        return frame / 255.0
    lo = float(frame.min())
    hi = float(frame.max())
    if hi <= lo:
        return np.zeros_like(frame, dtype=np.float32)
    return (frame - lo) / (hi - lo)


def main() -> None:
    args = parse_args()
    image_frames = read_image_frames(args.images)
    label_frames = read_image_frames(args.labels)

    if len(label_frames) == 1 and len(image_frames) > 1 and args.repeat_single_label:
        label_frames = np.repeat(label_frames, len(image_frames), axis=0)
    if len(image_frames) != len(label_frames):
        raise ValueError(f"Image and label frame counts differ: {len(image_frames)} vs {len(label_frames)}")

    images = []
    labels = []
    for image, label in zip(image_frames, label_frames):
        image = center_crop(image, args.crop_height, args.crop_width)
        label = center_crop(label, args.crop_height, args.crop_width)

        image = fit_to_square(image, args.size, pad_value=args.image_pad_value)
        label = fit_to_square(label, args.size, pad_value=0)

        image = normalize_image(image, args.normalize)
        if args.positive == "bright":
            label = label > args.label_threshold
        else:
            label = label < args.label_threshold

        images.append(image[..., np.newaxis].astype(np.float32))
        labels.append(label[..., np.newaxis].astype(np.float32))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_array = np.asarray(images, dtype=np.float32)
    label_array = np.asarray(labels, dtype=np.float32)
    np.save(output_dir / "data_tem.npy", image_array)
    np.save(output_dir / "label.npy", label_array)

    print(f"wrote {output_dir / 'data_tem.npy'} with shape {image_array.shape}")
    print(f"wrote {output_dir / 'label.npy'} with shape {label_array.shape}")


if __name__ == "__main__":
    main()
