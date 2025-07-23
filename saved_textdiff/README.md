# Saved Models Directory

This directory will contain your trained models and outputs after training.

## Structure After Training

```
saved_textdiff/
├── monuseg_2/
│   ├── model_best.pth          # Best performing model (copy)
│   └── model_XX.pth            # Best model at specific epoch (e.g., model_05.pth)
├── MosMedDataPlus/
│   ├── model_best.pth
│   └── model_XX.pth
└── qata_cov19_v2_2/
    ├── model_best.pth
    └── model_XX.pth
```

**Note**: `XX` represents the epoch number where the best performance was achieved.

## Usage

These models will be created during training and can be used for:
- Evaluation and testing
- Inference on new images
- Transfer learning to new datasets

The `model_best.pth` files are the recommended checkpoints for inference.
