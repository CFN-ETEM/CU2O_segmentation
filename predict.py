"""Run Cu2O U-Net inference on prepared arrays or TIFF images."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import tifffile

from data_preparation.image_utils import fit_to_square, read_image_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="pretrained/pretrained_cu2o.hdf5", help="Keras model file.")
    parser.add_argument("--input", default="data/data_tem.npy", help="Input NPY array or TIFF image stack.")
    parser.add_argument("--output-dir", default="results/predictions", help="Directory for prediction outputs.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability cutoff for binary masks.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--size", type=int, default=512, help="Square model input size for TIFF inputs.")
    parser.add_argument("--image-pad-value", type=float, default=255, help="Padding value for TIFF inputs.")
    parser.add_argument("--gpu", default="1", help="CUDA device id exposed to TensorFlow. Use '' for CPU.")
    return parser.parse_args()


def load_input(path: str, size: int, pad_value: float) -> np.ndarray:
    path_obj = Path(path)
    if path_obj.suffix.lower() == ".npy":
        array = np.load(path_obj)
        if array.ndim == 3:
            array = array[..., np.newaxis]
        return array.astype(np.float32)

    frames = read_image_frames(path_obj)
    fitted = [fit_to_square(frame, size, pad_value=pad_value)[..., np.newaxis] for frame in frames]
    return np.asarray(fitted, dtype=np.float32)


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import tensorflow as tf

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(args.model, compile=False)
    data = load_input(args.input, args.size, args.image_pad_value)
    probabilities = model.predict(data, batch_size=args.batch_size)
    masks = (probabilities >= args.threshold).astype(np.uint8)

    np.save(output_dir / "prediction_probability.npy", probabilities)
    np.save(output_dir / "prediction_mask.npy", masks)
    tifffile.imwrite(output_dir / "prediction_mask.tiff", (masks[..., 0] * 255).astype(np.uint8))

    print(f"wrote {output_dir / 'prediction_probability.npy'} with shape {probabilities.shape}")
    print(f"wrote {output_dir / 'prediction_mask.npy'} with shape {masks.shape}")
    print(f"wrote {output_dir / 'prediction_mask.tiff'}")


if __name__ == "__main__":
    main()

