"""Augment prepared Cu2O U-Net arrays.

The input directory must contain `data_tem.npy` and `label.npy` with shape
`(frames, height, width, 1)`. The script applies the same geometric
transforms to images and labels, then writes augmented arrays with the same
file names to a new directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, binary_propagation, gaussian_filter, rotate, zoom


def parse_number_list(value: str, cast_type: type = float) -> list:
    """Parse comma-separated values or range notation start:stop:step."""
    value = value.strip()
    if not value:
        return []
    if ":" in value:
        parts = [cast_type(part) for part in value.split(":")]
        if len(parts) != 3:
            raise argparse.ArgumentTypeError("Range notation must be start:stop:step.")
        start, stop, step = parts
        if step == 0:
            raise argparse.ArgumentTypeError("Range step cannot be zero.")
        return list(np.arange(start, stop, step, dtype=float))
    return [cast_type(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/custom", help="Directory containing data_tem.npy and label.npy.")
    parser.add_argument("--output-dir", default="data/custom_augmented", help="Directory for augmented arrays.")
    parser.add_argument(
        "--rotation-angles",
        default="0:360:10",
        help="Rotation angles in degrees. Use comma-separated values or start:stop:step notation.",
    )
    parser.add_argument(
        "--magnifications",
        default="90,95,105,110",
        help="Magnification percentages. Use an empty string to disable.",
    )
    parser.add_argument(
        "--blur-sigmas",
        default="1,2",
        help="Gaussian blur sigma values applied to images only. Use an empty string to disable.",
    )
    parser.add_argument(
        "--white-padding-threshold",
        type=float,
        default=240,
        help="Edge-connected pixels at or above this value are treated as white padding and filled before augmentation.",
    )
    parser.add_argument(
        "--edge-dilation-iterations",
        type=int,
        default=5,
        help="Number of dilation iterations used to remove interpolation halos around white padding.",
    )
    parser.add_argument("--label-threshold", type=float, default=0.5, help="Threshold for binarizing augmented labels.")
    return parser.parse_args()


def squeeze_frames(array: np.ndarray, name: str) -> np.ndarray:
    if array.ndim == 4 and array.shape[-1] == 1:
        return array[..., 0]
    if array.ndim == 3:
        return array
    raise ValueError(f"{name} must have shape (frames, height, width, 1) or (frames, height, width); got {array.shape}")


def fill_value(image: np.ndarray, white_padding_threshold: float = 240) -> float:
    """Use the mean non-padding intensity for geometric-transform corners."""
    image = np.asarray(image)
    non_white = image[image < white_padding_threshold]
    if non_white.size == 0:
        return float(np.mean(image))
    return float(np.mean(non_white))


def fill_white_padding_edges(
    image: np.ndarray,
    white_padding_threshold: float = 240,
    dilation_iterations: int = 5,
) -> np.ndarray:
    """Replace white padding connected to image edges with the image mean."""
    image = np.asarray(image, dtype=np.float32)
    high_mask = image >= white_padding_threshold
    if not np.any(high_mask):
        return image

    edge_seed = np.zeros(high_mask.shape, dtype=bool)
    edge_seed[0, :] = high_mask[0, :]
    edge_seed[-1, :] = high_mask[-1, :]
    edge_seed[:, 0] = high_mask[:, 0]
    edge_seed[:, -1] = high_mask[:, -1]
    if not np.any(edge_seed):
        return image

    edge_padding = binary_propagation(edge_seed, mask=high_mask)
    if dilation_iterations > 0:
        edge_padding = binary_dilation(edge_padding, iterations=dilation_iterations)

    filled = image.copy()
    filled[edge_padding] = fill_value(image, white_padding_threshold)
    return filled


def fit_to_shape(frame: np.ndarray, target_shape: tuple[int, int], pad_value: float) -> np.ndarray:
    """Center-crop or pad a 2-D frame to target shape."""
    target_h, target_w = target_shape
    h, w = frame.shape

    if h > target_h:
        top = (h - target_h) // 2
        frame = frame[top : top + target_h, :]
        h = target_h
    if w > target_w:
        left = (w - target_w) // 2
        frame = frame[:, left : left + target_w]
        w = target_w

    pad_top = (target_h - h) // 2
    pad_bottom = target_h - h - pad_top
    pad_left = (target_w - w) // 2
    pad_right = target_w - w - pad_left
    return np.pad(
        frame,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=pad_value,
    )


def rotate_pair(image: np.ndarray, label: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    image_fill = fill_value(image)
    image_aug = rotate(image, angle=angle, reshape=False, order=1, mode="constant", cval=image_fill)
    label_aug = rotate(label, angle=angle, reshape=False, order=0, mode="constant", cval=0)
    return image_aug, label_aug


def magnify_pair(image: np.ndarray, label: np.ndarray, percent: float) -> tuple[np.ndarray, np.ndarray]:
    factor = percent / 100.0
    image_fill = fill_value(image)
    image_aug = zoom(image, zoom=factor, order=1)
    label_aug = zoom(label, zoom=factor, order=0)
    image_aug = fit_to_shape(image_aug, image.shape, pad_value=image_fill)
    label_aug = fit_to_shape(label_aug, label.shape, pad_value=0)
    return image_aug, label_aug


def blur_pair(image: np.ndarray, label: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    return gaussian_filter(image, sigma=sigma), label.copy()


def augment_arrays(
    images: np.ndarray,
    labels: np.ndarray,
    rotation_angles: list[float],
    magnifications: list[float],
    blur_sigmas: list[float],
    label_threshold: float,
    white_padding_threshold: float,
    edge_dilation_iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    augmented_images = []
    augmented_labels = []

    for frame_index, (image, label) in enumerate(zip(images, labels)):
        image = fill_white_padding_edges(
            image,
            white_padding_threshold=white_padding_threshold,
            dilation_iterations=edge_dilation_iterations,
        )
        label = label > label_threshold

        for angle in rotation_angles:
            image_aug, label_aug = rotate_pair(image, label, angle)
            augmented_images.append(image_aug)
            augmented_labels.append(label_aug)

        for percent in magnifications:
            image_aug, label_aug = magnify_pair(image, label, percent)
            augmented_images.append(image_aug)
            augmented_labels.append(label_aug)

        for sigma in blur_sigmas:
            image_aug, label_aug = blur_pair(image, label, sigma)
            augmented_images.append(image_aug)
            augmented_labels.append(label_aug)

        print(
            f"frame {frame_index}: added "
            f"{len(rotation_angles)} rotations, {len(magnifications)} magnifications, "
            f"and {len(blur_sigmas)} blur variants"
        )

    image_array = np.asarray(augmented_images, dtype=np.float32)[..., np.newaxis]
    label_array = (np.asarray(augmented_labels) > label_threshold).astype(np.float32)[..., np.newaxis]
    return image_array, label_array


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = squeeze_frames(np.load(data_dir / "data_tem.npy"), "data_tem.npy")
    labels = squeeze_frames(np.load(data_dir / "label.npy"), "label.npy")
    if len(images) != len(labels):
        raise ValueError(f"Image and label frame counts differ: {len(images)} vs {len(labels)}")

    rotation_angles = parse_number_list(args.rotation_angles, float)
    magnifications = parse_number_list(args.magnifications, float)
    blur_sigmas = parse_number_list(args.blur_sigmas, float)
    if not rotation_angles and not magnifications and not blur_sigmas:
        raise ValueError("At least one augmentation type must be enabled.")

    image_array, label_array = augment_arrays(
        images=images,
        labels=labels,
        rotation_angles=rotation_angles,
        magnifications=magnifications,
        blur_sigmas=blur_sigmas,
        label_threshold=args.label_threshold,
        white_padding_threshold=args.white_padding_threshold,
        edge_dilation_iterations=args.edge_dilation_iterations,
    )
    np.save(output_dir / "data_tem.npy", image_array)
    np.save(output_dir / "label.npy", label_array)

    print(f"wrote {output_dir / 'data_tem.npy'} with shape {image_array.shape}")
    print(f"wrote {output_dir / 'label.npy'} with shape {label_array.shape}")


if __name__ == "__main__":
    main()
