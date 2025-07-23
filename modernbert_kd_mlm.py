import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForMaskedLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
import wandb
from huggingface_hub import HfApi

# Initialize Weights & Biases (wandb)
wandb.init(project="ModernClinicalBERT-KD", name="KD-training-run")

# Load tokenizer
teacher_model_name = "emilyalsentzer/Bio_ClinicalBERT"
student_model_name = "answerdotai/ModernBERT-base"

tokenizer = AutoTokenizer.from_pretrained(student_model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
teacher_model = AutoModelForMaskedLM.from_pretrained(teacher_model_name).to(device).eval()
student_model = AutoModelForMaskedLM.from_pretrained(student_model_name).to(device).train()

# Hyperparameters
lambda_kd = 0.5  # Weight for Knowledge Distillation loss
mlm_prob = 0.15  # Probability of masking tokens
batch_size = 8
lr = 5e-5
temperature = 2.0  # Temperature for soft labels
num_epochs = 10

# Log hyperparameters to wandb
wandb.config.update({
    "lambda_kd": lambda_kd,
    "mlm_prob": mlm_prob,
    "batch_size": batch_size,
    "learning_rate": lr,
    "temperature": temperature,
    "num_epochs": num_epochs
})


# Ensure teacher uses the same tokenizer
teacher_model.resize_token_embeddings(len(tokenizer))


# Resize student model embeddings to match the tokenizer vocab size
student_model.resize_token_embeddings(len(tokenizer))

# Load dataset (Medilora/mimic_iii_diagnosis_anonymous)
# dataset = load_dataset("Medilora/mimic_iii_diagnosis_anonymous", split="train")
dataset = load_dataset("mjkmain/mimic-100k", split="train")
texts = dataset["text"]

# Custom dataset class
class MaskedTextDataset(Dataset):
    def __init__(self, texts, tokenizer, mlm_prob=0.15, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.mlm_prob = mlm_prob
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Create MLM labels
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_prob)
        mask_token_indices = torch.bernoulli(probability_matrix).bool() & (input_ids != tokenizer.pad_token_id) & (input_ids != tokenizer.cls_token_id) & (input_ids != tokenizer.sep_token_id)
        input_ids[mask_token_indices] = tokenizer.mask_token_id
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

dataset = MaskedTextDataset(texts, tokenizer, mlm_prob=mlm_prob)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Optimizer
optimizer = optim.AdamW(student_model.parameters(), lr=lr)



# Training loop with tqdm
for epoch in range(num_epochs):
    epoch_loss = 0.0  # Track total loss per epoch
    num_batches = 0

    with tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as tepoch:
        for batch in tepoch:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Teacher predictions (no gradient calculation)
            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids, attention_mask=attention_mask).logits

            # Student predictions
            student_outputs = student_model(input_ids, attention_mask=attention_mask).logits

            # Compute MLM loss
            loss_mlm = F.cross_entropy(
                student_outputs.view(-1, student_outputs.size(-1)),
                labels.view(-1),
                ignore_index=tokenizer.pad_token_id
            )

            # Compute Knowledge Distillation loss
            loss_kd = F.kl_div(
                F.log_softmax(student_outputs / temperature, dim=-1),
                F.softmax(teacher_outputs / temperature, dim=-1),
                reduction='batchmean'
            )

            # Final loss
            loss = lambda_kd * loss_kd + (1 - lambda_kd) * loss_mlm

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update tracking variables
            epoch_loss += loss.item()
            num_batches += 1
            
            # Log batch loss to wandb
            wandb.log({"batch_loss": loss.item()})

            # Update tqdm progress bar with batch loss
            tepoch.set_postfix(loss=loss.item())

    avg_epoch_loss = epoch_loss / num_batches  # Compute average loss
    print(f"Epoch {epoch+1}/{num_epochs} | Average Loss: {avg_epoch_loss:.4f}")

    # Log epoch loss to wandb
    wandb.log({"epoch_loss": avg_epoch_loss})

# Save model and tokenizer
save_path = "siddharthdhara17/ModernClinicalBERT-4"
student_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print("Training complete!")

# Log model to wandb
wandb.save(save_path)


# Define your Hugging Face repo
repo_id = "siddharthdhara17/ModernClinicalBERT-4"

# Push model and tokenizer to the hub
student_model.push_to_hub(repo_id)
tokenizer.push_to_hub(repo_id)

print(f"Model pushed to Hugging Face")
