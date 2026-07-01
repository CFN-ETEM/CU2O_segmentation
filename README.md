# Cu2O U-Net Segmentation Workflow

This repository provides a workflow for segmenting Cu2O atomic columns from in situ HRTEM images using a U-Net model. The included data are a small demonstration set, so users can test the workflow without access to the full experimental dataset. The same scripts and notebook can be adapted to aligned HRTEM/IFFT image stacks from other experiments.

## Repository Layout

- `workflow.ipynb`: visual end-to-end notebook for label generation, label inspection, data augmentation, U-Net training, inference, and evaluation.
- `data_preparation/`: label-generation and dataset-building tools. The included `HRTEM.tiff` and `IFFT.tiff` are a minimal demo pair.
- `data/`: prepared example arrays used for pretrained-model inference.
- `pretrained/pretrained_cu2o.hdf5`: pretrained U-Net model for demonstration inference.
- `unetmodule.py`: TensorFlow/Keras U-Net architecture and checkpoint callback.
- `train_unet.py`: command-line training entry point.
- `predict.py`: command-line inference entry point.
- `analysis/`: centroid extraction, matching metrics, and an evaluation CLI.
- `main.ipynb`: legacy notebook retained for provenance.

## Installation

Create an isolated conda environment:

```bash
conda env create -f environment.yml
conda activate cu2o-seg
```

Or install with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

When using a GPU on a shared workstation, either expose a CUDA device:

```bash
export CUDA_VISIBLE_DEVICES=1
```

or pass `--gpu 1` to the training and prediction scripts.

## Notebook Workflow

For a visual, step-by-step workflow, open:

```bash
jupyter notebook workflow.ipynb
```

The notebook is organized so users can adapt the workflow by changing the input paths and parameters in the first code cell:

```python
HRTEM_PATH = Path("data_preparation/HRTEM.tiff")
IFFT_PATH = Path("data_preparation/IFFT.tiff")
```

The notebook shows the input image pair, generates initial Cu2O labels, displays the generated mask and overlay for inspection, builds U-Net-ready arrays, trains a model, runs inference, and evaluates predicted masks with centroid matching. For user datasets, inspect and manually correct the generated labels before building the final training arrays.
The training arrays are augmented by default using 10-degree rotations, magnification variants, and Gaussian-blur variants before model training.

## Command-Line Workflow

Generate an initial Cu2O atom mask from the included HRTEM/IFFT demo pair:

```bash
python data_preparation/generate_cu2o_labels.py \
  --hrtem data_preparation/HRTEM.tiff \
  --ifft data_preparation/IFFT.tiff \
  --output-dir outputs/demo_labels
```

This writes:

- `outputs/demo_labels/cu2o_mask.tiff`
- `outputs/demo_labels/cu2o_overlay.tiff`
- `outputs/demo_labels/cu2o_points.csv`

After inspecting and, if needed, manually correcting the label mask, convert the image and mask to U-Net arrays:

```bash
python data_preparation/build_unet_dataset.py \
  --images data_preparation/HRTEM.tiff \
  --labels outputs/demo_labels/cu2o_mask.tiff \
  --output-dir data/custom
```

This writes `data_tem.npy` and `label.npy` with shape `(frames, 512, 512, 1)`.

Augment the prepared training arrays:

```bash
python data_preparation/dataaugmentation.py \
  --data-dir data/custom \
  --output-dir data/custom_augmented \
  --rotation-angles 0:360:10 \
  --magnifications 90,95,105,110 \
  --blur-sigmas 1,2
```

This writes augmented `data_tem.npy` and `label.npy` arrays to `data/custom_augmented/`. The default settings generate rotations every 10 degrees from 0 to 350 degrees, magnification variants at 90%, 95%, 105%, and 110%, and Gaussian-blur variants with sigma values of 1 and 2. Before augmentation, white padding connected to the image border is filled with the average non-padding image intensity so rotations do not create white-border artifacts. Geometric transforms are applied to images and labels together, and labels are transformed with nearest-neighbor interpolation to preserve binary masks.

Run inference with the pretrained model:

```bash
python predict.py \
  --model pretrained/pretrained_cu2o.hdf5 \
  --input data/data_tem.npy \
  --output-dir results/demo_predictions \
  --gpu 1
```

Evaluate predictions against labels:

```bash
python analysis/evaluate_predictions.py \
  --labels data/label.npy \
  --predictions results/demo_predictions/prediction_mask.npy \
  --output results/demo_predictions/metrics.csv
```

Train a new model from prepared arrays:

```bash
python train_unet.py \
  --data-dir data/custom_augmented \
  --output-dir results/custom_training \
  --epochs 300 \
  --batch-size 4 \
  --gpu 1
```

Checkpoints are saved as `model_<epoch>.hdf5`, the training log is saved as `log.csv`, and the final model is saved as `model_final.hdf5`.

## Using a Custom Dataset

Prepare paired files for each experiment:

1. `HRTEM.tiff`: raw HRTEM image or time-series stack.
2. `IFFT.tiff`: inverse-FFT-filtered image or stack aligned to the HRTEM frames.

Run `generate_cu2o_labels.py` on the pair. The script detects atom candidates in the IFFT image, filters them by HRTEM intensity and nearest-neighbor spacing, then writes a binary mask, overlay, and coordinate table. Tune `--min-intensity`, `--max-intensity`, `--nn-min`, `--nn-max`, and `--second-nn-max` for different contrast, magnification, or lattice spacing. Use `--crop-height` and `--crop-width` when only the interface region should be labeled.

Before training, review the generated overlay and mask and manually correct the labels where needed. Then use the curated mask with `build_unet_dataset.py`, augment the prepared arrays with `dataaugmentation.py`, and train on the augmented output directory.

## Notes

Generated folders such as `outputs/`, `results/`, `data/custom/`, and `data/custom_augmented/` are ignored by git. The older notebooks in `data_preparation/` are retained to document the original label-generation process, but `workflow.ipynb` and the command-line scripts are the recommended public workflow.
