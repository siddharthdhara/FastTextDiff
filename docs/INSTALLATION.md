# FastTextDiff Documentation

## Quick Start Guide

### System Requirements
- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- 16GB+ RAM
- 50GB+ free disk space

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/siddharthdhara/FastTextDiff.git
   cd FastTextDiff
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python -m venv fasttextdiff_env
   # On Windows:
   fasttextdiff_env\Scripts\activate
   # On Linux/Mac:
   source fasttextdiff_env/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install git+https://github.com/huggingface/transformers.git
   ```

4. **Install MPI (Linux/Ubuntu only)**:
   ```bash
   sudo apt install -y mpich
   pip install mpi4py
   ```

### Dataset Structure

Each dataset should follow this structure:
```
datasets/your_dataset/
├── Train_Folder/
│   ├── img/          # Training images
│   ├── labelcol/     # Training labels
│   └── Train_text.xlsx  # Clinical descriptions
├── Val_Folder/
│   ├── img/          # Validation images
│   ├── labelcol/     # Validation labels
│   └── Val_text.xlsx   # Clinical descriptions
└── Test_Folder/
    ├── img/          # Test images
    ├── labelcol/     # Test labels
    └── Test_text.xlsx  # Clinical descriptions
```

### Training

1. **Basic Training**:
   ```bash
   python train.py --exp experiments/monuseg_2/ddpm.json
   ```

2. **Advanced Training**:
   ```bash
   python train_advanced.py --exp experiments/monuseg_2/ddpm.json
   ```

3. **Using Shell Scripts**:
   ```bash
   bash scripts/train.sh
   bash scripts/run_finetune.sh
   ```

### Fine-tuning ModernBERT

1. **Standard Fine-tuning**:
   ```bash
   python finetune_modernbert.py
   ```

2. **Extended Fine-tuning**:
   ```bash
   python finetune_modernbert_extended.py
   ```

3. **Knowledge Distillation**:
   ```bash
   python modernbert_kd_mlm.py
   ```

### Evaluation

```bash
python eval.py --exp experiments/monuseg_2/ddpm.json \
               --checkpoint saved_textdiff/monuseg_2/model_best.pth \
               --img_path datasets/monuseg_2/Test_Folder/img/sample.tif \
               --label_path datasets/monuseg_2/Test_Folder/labelcol/sample.png \
               --text_file datasets/monuseg_2/Test_Folder/Test_text.xlsx
```

## Configuration Parameters

### Key Parameters in experiment JSON files:

- `image_size`: Input image dimensions (default: 256)
- `batch_size`: Training batch size (adjust based on GPU memory)
- `max_training`: Maximum training epochs
- `dim`: Feature dimensions for diffusion model
- `steps`: Diffusion timesteps
- `number_class`: Number of segmentation classes

### Example Configuration:
```json
{
  "exp_dir": "saved_textdiff/your_dataset",
  "model_type": "ddpm",
  "category": "your_dataset",
  "number_class": 2,
  "image_size": 256,
  "batch_size": 1,
  "max_training": 50
}
```

## Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**:
   - Reduce batch_size in config
   - Use smaller image_size
   - Enable gradient checkpointing

2. **ModernBERT Loading Issues**:
   - Ensure transformers is installed from git
   - Check internet connection for model download

3. **MPI Installation Fails**:
   - Install mpich first: `sudo apt install -y mpich`
   - Then install mpi4py: `pip install mpi4py`

4. **Dataset Loading Errors**:
   - Check dataset structure matches expected format
   - Verify file paths in configuration
   - Ensure Excel files have correct column names

### Performance Tips:

1. **GPU Optimization**:
   - Use mixed precision training
   - Enable CUDA optimization flags
   - Monitor GPU memory usage

2. **Training Speed**:
   - Use multiple GPUs if available
   - Optimize data loading with multiple workers
   - Use gradient accumulation for large batch sizes

## Model Architecture Details

### Components:

1. **Text Encoder**: ClinicalModernBERT
   - Processes clinical descriptions
   - Generates text embeddings
   - Fine-tuned on medical text

2. **Diffusion Backbone**: DDPM
   - Pre-trained on large image datasets
   - Provides robust feature representations
   - Multi-scale feature extraction

3. **Pixel Classifier**: 
   - Text-conditioned segmentation head
   - Combines visual and textual features
   - Outputs final segmentation masks

### Training Process:

1. **Phase 1**: Fine-tune ModernBERT on clinical text
2. **Phase 2**: Train pixel classifier with frozen diffusion backbone
3. **Phase 3**: End-to-end fine-tuning (optional)

## File Descriptions

### Main Scripts:
- `train.py` - Main training script
- `train_advanced.py` - Advanced training with additional features
- `eval.py` - Evaluation and visualization
- `utils.py` - Utility functions

### Fine-tuning Scripts:
- `finetune_modernbert.py` - Standard BERT fine-tuning
- `finetune_modernbert_extended.py` - Extended fine-tuning
- `modernbert_kd_mlm.py` - Knowledge distillation

### Testing:
- `test_model_v2.py` - Model testing utilities

### Configuration:
- `experiments/` - Experiment configurations
- `checkpoints/` - Pre-trained model checkpoints
