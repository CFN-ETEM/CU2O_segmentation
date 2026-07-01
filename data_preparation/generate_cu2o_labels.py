"""Generate Cu2O atom labels from paired HRTEM and IFFT TIFF images.

The original notebook used determinant-of-Hessian blob detection on the IFFT
image, then removed Cu/background blobs using HRTEM intensity and nearest-
neighbor filters. This script exposes the same workflow as a reproducible CLI.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.spatial import cKDTree
from skimage.draw import disk
from skimage.feature import blob_doh

from image_utils import center_crop, read_tiff_frames, to_uint8, write_tiff_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hrtem", default="data_preparation/HRTEM.tiff", help="Raw HRTEM TIFF image or stack.")
    parser.add_argument("--ifft", default="data_preparation/IFFT.tiff", help="IFFT-filtered TIFF image or stack.")
    parser.add_argument("--output-dir", default="outputs/labels", help="Directory for mask, overlay, and CSV outputs.")
    parser.add_argument("--crop-height", type=int, default=None, help="Optional centered crop height before detection.")
    parser.add_argument("--crop-width", type=int, default=None, help="Optional centered crop width before detection.")
    parser.add_argument("--min-sigma", type=float, default=2.7, help="Minimum blob sigma for DoH detection.")
    parser.add_argument("--max-sigma", type=float, default=3.2, help="Maximum blob sigma for DoH detection.")
    parser.add_argument("--blob-threshold", type=float, default=2.5e-5, help="DoH response threshold.")
    parser.add_argument("--overlap", type=float, default=0.1, help="Maximum allowed blob overlap.")
    parser.add_argument("--median-size", type=int, default=5, help="Median filter size for IFFT detection image.")
    parser.add_argument("--intensity-median-size", type=int, default=11, help="Median filter size for HRTEM intensity filter.")
    parser.add_argument("--min-intensity", type=float, default=85, help="Minimum mean HRTEM intensity for Cu2O candidates.")
    parser.add_argument("--max-intensity", type=float, default=210, help="Maximum mean HRTEM intensity for Cu2O candidates.")
    parser.add_argument("--intensity-radius", type=int, default=3, help="Radius around each blob for intensity averaging.")
    parser.add_argument("--nn-min", type=float, default=8.6, help="Minimum nearest-neighbor distance in pixels.")
    parser.add_argument("--nn-max", type=float, default=12.0, help="Maximum nearest-neighbor distance in pixels.")
    parser.add_argument("--second-nn-max", type=float, default=15.0, help="Maximum second-nearest-neighbor distance in pixels.")
    parser.add_argument("--disable-nn-filter", action="store_true", help="Keep intensity-filtered blobs without NN filtering.")
    parser.add_argument("--mask-radius", type=int, default=3, help="Positive disk radius written into the binary mask.")
    return parser.parse_args()


def local_mean(image: np.ndarray, y: float, x: float, radius: int) -> float:
    h, w = image.shape
    y0 = max(0, int(round(y)) - radius)
    y1 = min(h, int(round(y)) + radius + 1)
    x0 = max(0, int(round(x)) - radius)
    x1 = min(w, int(round(x)) + radius + 1)
    return float(np.mean(image[y0:y1, x0:x1]))


def filter_by_neighbors(points: list[dict], nn_min: float, nn_max: float, second_nn_max: float) -> list[dict]:
    if len(points) < 3:
        return points

    coords = np.array([[p["y"], p["x"]] for p in points], dtype=float)
    distances, _ = cKDTree(coords).query(coords, k=min(3, len(points)))
    kept: list[dict] = []
    for point, distance_row in zip(points, distances):
        nearest = float(distance_row[1])
        second = float(distance_row[2]) if len(distance_row) > 2 else float("nan")
        point["nearest_px"] = nearest
        point["second_nearest_px"] = second
        if nn_min <= nearest <= nn_max and second <= second_nn_max:
            kept.append(point)
    return kept


def detect_frame(
    hrtem: np.ndarray,
    ifft: np.ndarray,
    args: argparse.Namespace,
    frame_index: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    hrtem = center_crop(hrtem, args.crop_height, args.crop_width)
    ifft = center_crop(ifft, args.crop_height, args.crop_width)

    detection_image = median_filter(ifft, size=args.median_size)
    intensity_image = median_filter(hrtem, size=args.intensity_median_size)
    blobs = blob_doh(
        detection_image,
        min_sigma=args.min_sigma,
        max_sigma=args.max_sigma,
        threshold=args.blob_threshold,
        overlap=args.overlap,
    )

    candidates: list[dict] = []
    for y, x, radius in blobs:
        intensity = local_mean(intensity_image, y, x, args.intensity_radius)
        if args.min_intensity <= intensity <= args.max_intensity:
            candidates.append(
                {
                    "frame": frame_index,
                    "y": float(y),
                    "x": float(x),
                    "radius_px": float(radius),
                    "intensity": intensity,
                    "nearest_px": float("nan"),
                    "second_nearest_px": float("nan"),
                }
            )

    points = candidates if args.disable_nn_filter else filter_by_neighbors(
        candidates,
        args.nn_min,
        args.nn_max,
        args.second_nn_max,
    )

    mask = np.zeros(hrtem.shape, dtype=np.uint8)
    overlay = to_uint8(hrtem)
    for point in points:
        rr, cc = disk((point["y"], point["x"]), args.mask_radius, shape=mask.shape)
        mask[rr, cc] = 255
        overlay[rr, cc] = 0

    return mask, overlay, points


def main() -> None:
    args = parse_args()
    hrtem_frames = read_tiff_frames(args.hrtem)
    ifft_frames = read_tiff_frames(args.ifft)
    if len(hrtem_frames) != len(ifft_frames):
        raise ValueError(f"HRTEM and IFFT frame counts differ: {len(hrtem_frames)} vs {len(ifft_frames)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    masks = []
    overlays = []
    all_points: list[dict] = []
    for frame_index, (hrtem, ifft) in enumerate(zip(hrtem_frames, ifft_frames)):
        mask, overlay, points = detect_frame(hrtem, ifft, args, frame_index)
        masks.append(mask)
        overlays.append(overlay)
        all_points.extend(points)
        print(f"frame {frame_index}: kept {len(points)} Cu2O atom candidates")

    write_tiff_frames(output_dir / "cu2o_mask.tiff", np.asarray(masks, dtype=np.uint8))
    write_tiff_frames(output_dir / "cu2o_overlay.tiff", np.asarray(overlays, dtype=np.uint8))

    csv_path = output_dir / "cu2o_points.csv"
    with csv_path.open("w", newline="") as handle:
        fieldnames = ["frame", "y", "x", "radius_px", "intensity", "nearest_px", "second_nearest_px"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_points)

    print(f"wrote {output_dir / 'cu2o_mask.tiff'}")
    print(f"wrote {output_dir / 'cu2o_overlay.tiff'}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()

