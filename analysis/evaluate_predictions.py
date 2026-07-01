"""Evaluate predicted Cu2O masks against binary labels using centroid matching."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

try:
    from analysisutil import centroids, match_centroids
except ModuleNotFoundError:
    from analysis.analysisutil import centroids, match_centroids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default="data/label.npy", help="Ground-truth label NPY array.")
    parser.add_argument("--predictions", default="results/predictions/prediction_mask.npy", help="Prediction NPY array.")
    parser.add_argument("--output", default="results/predictions/metrics.csv", help="CSV path for per-frame metrics.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for probability predictions.")
    parser.add_argument("--radius", type=float, default=2.5, help="Centroid matching radius in pixels.")
    return parser.parse_args()


def squeeze_frames(array: np.ndarray) -> np.ndarray:
    if array.ndim == 4 and array.shape[-1] == 1:
        return array[..., 0]
    if array.ndim == 3:
        return array
    if array.ndim == 2:
        return array[np.newaxis, ...]
    raise ValueError(f"Expected 2-D, 3-D, or singleton-channel 4-D array; got {array.shape}")


def main() -> None:
    args = parse_args()
    labels = squeeze_frames(np.load(args.labels)) > args.threshold
    predictions = squeeze_frames(np.load(args.predictions)) > args.threshold
    if len(labels) != len(predictions):
        raise ValueError(f"Label and prediction frame counts differ: {len(labels)} vs {len(predictions)}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    totals = {"tp": 0, "fp": 0, "fn": 0}
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame", "tp", "fp", "fn"])
        writer.writeheader()
        for frame_index, (label, prediction) in enumerate(zip(labels, predictions)):
            tp, fp, fn = match_centroids(centroids(label), centroids(prediction), r=args.radius)
            totals["tp"] += tp
            totals["fp"] += fp
            totals["fn"] += fn
            writer.writerow({"frame": frame_index, "tp": tp, "fp": fp, "fn": fn})

    precision = totals["tp"] / max(totals["tp"] + totals["fp"], 1)
    recall = totals["tp"] / max(totals["tp"] + totals["fn"], 1)
    print(f"wrote {output_path}")
    print(f"total TP={totals['tp']} FP={totals['fp']} FN={totals['fn']}")
    print(f"precision={precision:.4f} recall={recall:.4f}")


if __name__ == "__main__":
    main()
