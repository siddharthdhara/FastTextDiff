#!/bin/bash
#SBATCH --job-name=ModernBERT-finetune
#SBATCH --partition=ihub
#SBATCH --account=programs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20  
#SBATCH --time=24:00:00
#SBATCH --output=../logs/%x_%j.out
#SBATCH --error=../logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=venkatasiddharth.d@research.iiit.ac.in

# Load the necessary modules
module load u18/python/3.8.3
module load u18/cudnn/8.4.0-cuda-11.6
module load u18/cuda/11.6

# Initialize Conda for the shell
conda init bash
source ~/.bashrc

# Activate your environment
conda activate textdiff

# Set WandB API key securely
export WANDB_API_KEY="f3ca18f61aaa0db8a3bb23d72ded0f052a98bbb9"

# Hugging Face login 
huggingface-cli login --token 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 

# Navigate to the script directory
cd ~/ModernBERT-MIMICIII/scripts


# Run the fine-tuning script
python finetune_modernbert.py

