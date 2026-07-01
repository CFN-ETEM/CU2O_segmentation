# Data Preparation

This folder contains the label-generation, dataset-building, and data-augmentation tools for Cu2O U-Net training. The bundled `HRTEM.tiff` and `IFFT.tiff` files are a small demonstration pair; custom datasets should use aligned HRTEM and IFFT TIFF images or stacks with the same frame count.

## Label Generation

Run:

```bash
python data_preparation/generate_cu2o_labels.py \
  --hrtem data_preparation/HRTEM.tiff \
  --ifft data_preparation/IFFT.tiff \
  --output-dir outputs/demo_labels
```

The script applies determinant-of-Hessian blob detection to the IFFT image, filters candidate atoms by local HRTEM intensity, and removes candidates whose nearest-neighbor spacing is inconsistent with the Cu2O lattice. Outputs are:

- `cu2o_mask.tiff`: binary mask with Cu2O atom disks as positive pixels.
- `cu2o_overlay.tiff`: HRTEM image with accepted atom locations marked.
- `cu2o_points.csv`: frame, coordinate, intensity, and neighbor-distance table.

Useful parameters:

- `--min-intensity` and `--max-intensity`: reject Cu atoms or background blobs by raw-image contrast.
- `--nn-min`, `--nn-max`, `--second-nn-max`: tune expected atom spacing in pixels.
- `--crop-height`, `--crop-width`: center-crop an interface region before detection.
- `--disable-nn-filter`: inspect intensity-filtered detections before spacing cleanup.

## U-Net Array Generation

Convert images and labels into arrays expected by `unetmodule.py`:

```bash
python data_preparation/build_unet_dataset.py \
  --images data_preparation/HRTEM.tiff \
  --labels outputs/demo_labels/cu2o_mask.tiff \
  --output-dir data/custom
```

This writes `data_tem.npy` and `label.npy` with shape `(frames, 512, 512, 1)`. Image and label stacks should have the same frame count. Labels are interpreted as bright positive pixels by default; use `--positive dark` only for legacy masks where atom positions are black on a bright background.

## Data Augmentation

After inspecting and manually correcting the generated labels, augment the prepared arrays:

```bash
python data_preparation/dataaugmentation.py \
  --data-dir data/custom \
  --output-dir data/custom_augmented \
  --rotation-angles 0:360:10 \
  --magnifications 90,95,105,110 \
  --blur-sigmas 1,2
```

The default settings create rotations from 0 to 350 degrees in 10-degree increments, magnification variants at 90%, 95%, 105%, and 110%, and Gaussian-blur variants with sigma values of 1 and 2. White padding connected to the image border is filled with the average non-padding image intensity before augmentation, preventing white-border artifacts after rotation. Image and label transforms are paired, and labels use nearest-neighbor interpolation to remain binary.

The legacy notebooks are kept as provenance for the original manual workflow. Prefer `workflow.ipynb` and the scripts above for public use and reviewer inspection.
