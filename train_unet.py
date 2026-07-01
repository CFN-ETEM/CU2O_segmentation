"""Train the Cu2O U-Net on prepared NumPy arrays."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Directory containing data_tem.npy and label.npy.")
    parser.add_argument("--output-dir", default="results/training", help="Directory for checkpoints and logs.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--save-every", type=int, default=10, help="Checkpoint interval in epochs.")
    parser.add_argument("--gpu", default="1", help="CUDA device id exposed to TensorFlow. Use '' for CPU.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import tensorflow as tf
    from tensorflow.keras.callbacks import CSVLogger
    from tensorflow.keras.optimizers import Adam

    from unetmodule import CustomModelCheckpoint, get_unet_with_batchnorm

    tf.keras.backend.clear_session()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data = np.load(data_dir / "data_tem.npy")
    label_data = np.load(data_dir / "label.npy")

    model = get_unet_with_batchnorm()
    model.compile(optimizer=Adam(learning_rate=args.learning_rate), loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    checkpoint_callback = CustomModelCheckpoint(
        save_path=str(output_dir / "model_{epoch}.hdf5"),
        save_every=args.save_every,
    )
    csv_logger = CSVLogger(str(output_dir / "log.csv"), append=False)
    model.fit(
        train_data,
        label_data,
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=1,
        callbacks=[checkpoint_callback, csv_logger],
    )
    model.save(str(output_dir / "model_final.hdf5"))
    print(f"wrote {output_dir / 'model_final.hdf5'}")


if __name__ == "__main__":
    main()
