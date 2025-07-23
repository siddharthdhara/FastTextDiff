# FastTextDiff: A Fast and Efficient Modern BERT based Text-Conditioned Diffusion Model for Medical Image Segmentation

## Authors
- **Venkata Siddharth Dhara**, IIIT Hyderabad
- **Pawan Kumar**, IIIT Hyderabad

## Overview

FastTextDiff is a novel approach that combines modern BERT language model with diffusion-based techniques for medical image segmentation. This repository contains the implementation of our text-conditioned diffusion model that leverages clinical text descriptions to improve medical image segmentation accuracy.

## Features

- **Modern BERT Integration**: Utilizes ClinicalModernBERT for text encoding
- **Diffusion-based Segmentation**: Employs DDPM (Denoising Diffusion Probabilistic Models) for image generation and segmentation
- **Multi-dataset Support**: Supports multiple medical imaging datasets
- **Efficient Training**: Fast training with mixed precision and optimized memory usage
- **Comprehensive Evaluation**: Built-in visualization and evaluation tools

## Repository Structure

```
FastTextDiff/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── LICENSE                            # License file
├── setup.py                           # Package installation setup
├── .gitignore                         # Git ignore rules
├── train.py                           # Main training script
├── train_advanced.py                  # Advanced training script
├── train_old.py                       # Legacy training script
├── eval.py                            # Evaluation and visualization script
├── utils.py                           # Utility functions
├── finetune_modernbert.py             # ModernBERT fine-tuning script (MLM)
├── finetune_modernbert_extended.py    # Extended BERT fine-tuning
├── modernbert_kd_mlm.py               # Knowledge distillation for ModernBERT
├── test_model_v2.py                   # Model testing utilities
├── checkpoints/                       # Pre-trained model checkpoints
│   └── ddpm/
│       └── 256x256_diffusion_uncond.pt
├── datasets/                          # Dataset storage
│   ├── monuseg_2/                     # MoNuSeg dataset
│   │   ├── Train_Folder/
│   │   │   ├── img/                   # Training images
│   │   │   ├── labelcol/              # Training labels
│   │   │   └── Train_text.xlsx        # Training clinical descriptions
│   │   ├── Val_Folder/
│   │   │   ├── img/                   # Validation images
│   │   │   ├── labelcol/              # Validation labels
│   │   │   └── Val_text.xlsx          # Validation clinical descriptions
│   │   └── Test_Folder/
│   │       ├── img/                   # Test images
│   │       ├── labelcol/              # Test labels
│   │       └── Test_text.xlsx         # Test clinical descriptions
│   ├── MosMedDataPlus/                # MosMed+ dataset
│   │   ├── Train_Folder/
│   │   │   ├── img/
│   │   │   ├── labelcol/
│   │   │   └── Train_text.xlsx
│   │   ├── Val_Folder/
│   │   │   ├── img/
│   │   │   ├── labelcol/
│   │   │   └── Val_text.xlsx
│   │   └── Test_Folder/
│   │       ├── img/
│   │       ├── labelcol/
│   │       └── Test_text.xlsx
│   └── qata_cov19_v2_2/               # QATA COVID-19 dataset
│       ├── Train_Folder/
│       │   ├── img/
│       │   ├── labelcol/
│       │   └── Train_text.xlsx
│       ├── Val_Folder/
│       │   ├── img/
│       │   ├── labelcol/
│       │   └── Val_text.xlsx
│       └── Test_Folder/
│           ├── img/
│           ├── labelcol/
│           └── Test_text.xlsx
├── experiments/                       # Experiment configurations
│   ├── monuseg_2/
│   ├── MosMedDataPlus/
│   └── qata_cov19_v2_2/
├── guided_diffusion/                  # Diffusion model implementation
├── src/                               # Source code modules
│   ├── data_util.py                   # Data utilities
│   ├── feature_extractors.py          # Feature extraction modules
│   ├── pixel_classifier.py            # Pixel-level classifier
│   ├── transformer.py                 # Transformer implementations
│   ├── load_dataset2.py               # Dataset loading utilities
│   └── utils.py                       # Additional utilities
├── scripts/                           # Shell scripts
│   ├── train.sh                       # Training script
│   ├── run_finetune.sh                # Fine-tuning script
│   └── run_finetune_kd_mlm.sh         # KD fine-tuning script
├── docs/                              # Documentation
│   └── INSTALLATION.md                # Detailed installation guide
├── saved_textdiff/                    # Saved model outputs
└── wandb/                            # Weights & Biases logs
```

## Datasets

The repository supports three medical imaging datasets:

1. **MoNuSeg**: Multi-organ nuclei segmentation dataset
2. **MosMedData+**: COVID-19 CT scan dataset
3. **QATA COVID-19 v2.2**: COVID-19 chest X-ray dataset

Each dataset folder contains:
- `Train_Folder/`: Training images and labels
- `Val_Folder/`: Validation images and labels  
- `Test_Folder/`: Test images and labels
- Text files with clinical descriptions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/siddharthdhara/FastTextDiff.git
cd FastTextDiff
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install transformers from Hugging Face Git Repository:
```bash
pip install git+https://github.com/huggingface/transformers.git
```

4. Install additional requirements if needed:
```bash
pip install flash-attn pandas pillow scikit-learn matplotlib wandb
```

5. Install MPI dependencies (Linux/Ubuntu):
```bash
sudo apt install -y mpich
pip install mpi4py
```

**Note**: If `mpich` installation fails, this is a common fix for MPI-related dependencies that may be required for distributed training.

## Usage

### Training

1. **Basic Training**:
```bash
python train.py --exp experiments/monuseg_2/ddpm.json
```

2. **Advanced Training Script**:
```bash
python train_advanced.py --exp experiments/monuseg_2/ddpm.json
```

3. **Using Shell Scripts**:
```bash
# For training
bash scripts/train.sh

# For fine-tuning
bash scripts/run_finetune.sh

# For knowledge distillation fine-tuning
bash scripts/run_finetune_kd_mlm.sh
```

### Evaluation and Visualization

Run evaluation on test data with visualization:
```bash
python eval.py --exp experiments/monuseg_2/ddpm.json --checkpoint saved_textdiff/monuseg_2/model_best.pth --img_path datasets/monuseg_2/Test_Folder/img/sample.tif --label_path datasets/monuseg_2/Test_Folder/labelcol/sample.png --text_file datasets/monuseg_2/Test_Folder/Test_text.xlsx
```

### Fine-tuning ModernBERT

Fine-tune the ModernBERT model for clinical text:
```bash
# Standard fine-tuning
python finetune_modernbert.py

# Extended fine-tuning
python finetune_modernbert_extended.py

# Knowledge distillation
python modernbert_kd_mlm.py
```

## Configuration

Each experiment uses a JSON configuration file located in the `experiments/` directory. Key parameters include:

- `image_size`: Input image dimensions (256x256)
- `batch_size`: Training batch size
- `max_training`: Maximum training epochs
- `dim`: Feature dimensions for diffusion model
- `steps`: Diffusion timesteps
- `number_class`: Number of segmentation classes

## Model Architecture

The FastTextDiff model consists of:

1. **Text Encoder**: ClinicalModernBERT for processing clinical descriptions
2. **Feature Extractor**: Multi-scale feature extraction from diffusion model
3. **Pixel Classifier**: Final segmentation head with text conditioning
4. **Diffusion Backbone**: Pre-trained DDPM for robust feature representation

## Results

The model achieves competitive performance on medical image segmentation tasks by leveraging both visual and textual information. Detailed results and metrics are available in our paper.

## Pre-trained Models

### Download Checkpoints
Pre-trained checkpoints are available for download:

- **Diffusion backbone**: [Download from Google Drive](https://drive.google.com/drive/folders/1SjzYE_dD5IimiiBYIgd8AO-85F9u3LEj?usp=sharing)
  - Download `256x256_diffusion_uncond.pt` and place in: `checkpoints/ddpm/256x256_diffusion_uncond.pt`
- **Fine-tuned models**: Available after training in `saved_textdiff/`

### Quick Setup
```bash
# Create necessary directories
mkdir -p checkpoints/ddpm
mkdir -p datasets

# Download from Google Drive (link above) and extract:
# 1. Place 256x256_diffusion_uncond.pt in checkpoints/ddpm/
# 2. Extract dataset folders to datasets/
# 3. Your structure should match the Repository Structure section

# Verify setup
ls checkpoints/ddpm/  # Should contain 256x256_diffusion_uncond.pt
ls datasets/          # Should contain monuseg_2, MosMedDataPlus, qata_cov19_v2_2
```

## Datasets

### Dataset Sources

**Option 1: Pre-processed Datasets (Recommended)**
Download our pre-processed datasets and checkpoint:
- **All Datasets + Checkpoint**: [Google Drive Folder](https://drive.google.com/drive/folders/1SjzYE_dD5IimiiBYIgd8AO-85F9u3LEj?usp=sharing)
  - Contains: MoNuSeg, MosMedData+, QATA COVID-19 datasets + DDPM checkpoint
  - Ready to use with proper folder structure and clinical descriptions

**Option 2: Original Dataset Sources**
Download the original datasets from their official sources:
- **MoNuSeg**: [Multi-Organ Nuclei Segmentation Challenge](https://monuseg.grand-challenge.org/Data/)
- **MosMedData+**: [COVID-19 CT Scan Dataset](https://medicalsegmentation.com/covid19/)
- **QATA COVID-19**: [COVID-19 Chest X-ray Dataset](https://www.kaggle.com/datasets/aysendegerli/qatacov19-dataset)

### Dataset Preprocessing
**If using Option 1 (Pre-processed)**: Skip this section - datasets are ready to use!

**If using Option 2 (Original sources)**: After downloading the original datasets, you'll need to:

1. **Organize** the data according to our [Repository Structure](#repository-structure)
2. **Create clinical descriptions** in Excel format (`Train_text.xlsx`, `Val_text.xlsx`, `Test_text.xlsx`)
3. **Ensure proper naming** for image and label files

### Expected Structure
Each dataset should be organized as:
```bash
datasets/your_dataset/
├── Train_Folder/
│   ├── img/                    # Training images (.tif, .png, .jpg)
│   ├── labelcol/              # Training labels/masks (.png)
│   └── Train_text.xlsx        # Clinical descriptions
├── Val_Folder/
│   ├── img/                    # Validation images
│   ├── labelcol/              # Validation labels/masks
│   └── Val_text.xlsx          # Clinical descriptions
└── Test_Folder/
    ├── img/                    # Test images
    ├── labelcol/              # Test labels/masks
    └── Test_text.xlsx         # Clinical descriptions
```

### Clinical Text Format
The Excel files should contain columns for:
- Image filename
- Clinical description/text
- Any additional metadata

**Note**: If you need help with dataset preprocessing or clinical text generation, please refer to our [Installation Guide](docs/INSTALLATION.md) for detailed instructions.

## Documentation

For detailed installation instructions, troubleshooting, and advanced usage, see:
- [Installation Guide](docs/INSTALLATION.md)

## Installation via pip

You can install this package directly:
```bash
pip install -e .
```

This will install the package and make command-line tools available:
```bash
fasttextdiff-train --exp experiments/monuseg_2/ddpm.json
fasttextdiff-eval --exp experiments/monuseg_2/ddpm.json --checkpoint model.pth
```


