# Checkpoints Directory

This directory should contain pre-trained model checkpoints.

## Download Instructions

Download the pre-trained DDPM checkpoint from our Google Drive:
- **Google Drive**: [Download Checkpoint](https://drive.google.com/drive/folders/1SjzYE_dD5IimiiBYIgd8AO-85F9u3LEj?usp=sharing)

Place the downloaded `256x256_diffusion_uncond.pt` file here:

```
checkpoints/
└── ddpm/
    └── 256x256_diffusion_uncond.pt
```

## File Details

- **256x256_diffusion_uncond.pt**: Pre-trained DDPM model for 256x256 images
- **Size**: ~1.2GB
- **Required for**: Training and inference

Without this checkpoint, you won't be able to train or run inference with FastTextDiff.
