# Training Scripts

This directory contains model training code.

## Scripts

- `train.py` - Main training script for the segmentation model
- `outputs/` - Training outputs (metrics, plots, checkpoints)

## Usage

To train the model:

```bash
python training/train.py
```

Or from the root:

```bash
python -m training.train
```

This will:
1. Load images from `Dataset-Segmentation/`
2. Create 256×256 patches
3. Train a U-Net model
4. Save model to `models/semantic_segmentation_model.h5`
5. Generate training outputs in `training/outputs/`

## Configuration

Training parameters can be configured in the script or via environment variables (future enhancement).
