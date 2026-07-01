"""Image loading and formatting helpers for the Cu2O U-Net workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile


def read_tiff_frames(path: str | Path) -> np.ndarray:
    """Read a 2-D TIFF or 3-D TIFF stack as frames with shape (n, h, w)."""
    array = tifffile.imread(path)
    array = np.asarray(array)

    if array.ndim == 2:
        return array[np.newaxis, ...]
    if array.ndim == 3:
        return array
    if array.ndim == 4 and array.shape[-1] == 1:
        return array[..., 0]
    raise ValueError(f"Expected a 2-D image or 3-D stack, got shape {array.shape} from {path}")


def read_image_frames(path: str | Path) -> np.ndarray:
    """Read TIFF or NPY image data as frames with shape (n, h, w)."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        array = np.load(path)
        if array.ndim == 2:
            return array[np.newaxis, ...]
        if array.ndim == 3:
            return array
        if array.ndim == 4 and array.shape[-1] == 1:
            return array[..., 0]
        raise ValueError(f"Expected NPY data with 2, 3, or singleton-channel 4 dimensions; got {array.shape}")
    return read_tiff_frames(path)


def center_crop(frame: np.ndarray, height: int | None, width: int | None) -> np.ndarray:
    """Center crop a frame when height and width are provided."""
    if height is None and width is None:
        return frame
    if height is None or width is None:
        raise ValueError("Both crop height and crop width must be provided.")

    h, w = frame.shape
    if height > h or width > w:
        raise ValueError(f"Crop size {(height, width)} is larger than frame size {(h, w)}.")

    top = (h - height) // 2
    left = (w - width) // 2
    return frame[top : top + height, left : left + width]


def fit_to_square(frame: np.ndarray, size: int, pad_value: float = 0) -> np.ndarray:
    """Center crop or pad a 2-D frame to a square target size."""
    h, w = frame.shape

    if h > size:
        top = (h - size) // 2
        frame = frame[top : top + size, :]
        h = size
    if w > size:
        left = (w - size) // 2
        frame = frame[:, left : left + size]
        w = size

    pad_top = (size - h) // 2
    pad_bottom = size - h - pad_top
    pad_left = (size - w) // 2
    pad_right = size - w - pad_left

    return np.pad(
        frame,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=pad_value,
    )


def write_tiff_frames(path: str | Path, frames: np.ndarray) -> None:
    """Write one frame as 2-D TIFF or multiple frames as a TIFF stack."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.asarray(frames)
    if frames.shape[0] == 1:
        tifffile.imwrite(path, frames[0])
    else:
        tifffile.imwrite(path, frames)


def to_uint8(frame: np.ndarray) -> np.ndarray:
    """Scale an image frame to uint8 for visualization outputs."""
    frame = np.asarray(frame, dtype=np.float32)
    lo = float(np.nanmin(frame))
    hi = float(np.nanmax(frame))
    if hi <= lo:
        return np.zeros(frame.shape, dtype=np.uint8)
    return np.clip((frame - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)

