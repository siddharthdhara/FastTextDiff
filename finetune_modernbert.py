from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling # type: ignore
from datasets import load_dataset, DatasetDict # type: ignore
import torch # type: ignore
import os
import wandb # type: ignore
from transformers import EarlyStoppingCallback # type: ignore
import time
from transformers import TrainerCallback # type: ignore


# Configuration
model_id = "answerdotai/ModernBERT-base"
#dataset_name = "Medilora/mimic_iii_diagnosis_anonymous"
dataset_name = "mjkmain/mimic-100k"
output_dir = output_dir = "/home2/venkatasiddharth.d/ModernBERT-MIMICIII/outputs"
os.makedirs(output_dir, exist_ok=True)
push_to_hub = True
# hf_repo_name = "ModernBERT-MIMICIII-mlm"
# hf_repo_name =  "ModernClinicalBERT-2" 
#hf_repo_name =  "ModernClinicalBERT-3"
hf_repo_name =  "ClinicalModernBERT"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize wandb
wandb.init(project="ModernBERT-MIMICIV", name="finetuning_withflash_attn", config={"epochs": 4, "batch_size": 8})

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id).to(device)

# Enable Flash Attention-2
if hasattr(model.config, "use_flash_attention_2"):
    model.config.use_flash_attention_2 = True
else:
    print("Flash Attention-2 is not available for this model.")


# Push tokenizer to Hugging Face Hub
if push_to_hub:
    tokenizer.push_to_hub(hf_repo_name, commit_message="Uploading tokenizer")


# Test prediction function
def predict(text, top_k=5):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model(**inputs)
    masked_index = inputs["input_ids"][0].tolist().index(tokenizer.mask_token_id)
    top_k_logits = torch.topk(outputs.logits[0, masked_index], top_k)
    top_k_tokens = tokenizer.batch_decode(top_k_logits.indices)
    top_k_probabilities = torch.softmax(top_k_logits.values, dim=0)
    predictions = list(zip(top_k_tokens, top_k_probabilities))
    print(text)
    for token, prob in predictions:
        print(f"  {token.strip()}: {prob:.4f}")
    print("")

predict("patient admitted with [MASK] failure")
predict("diabetes [MASK] neuropathy")

# Load dataset
dataset = load_dataset(dataset_name)
if "validation" not in dataset.keys():
    split_dataset = dataset["train"].train_test_split(test_size=0.1)
    dataset = DatasetDict({
        "train": split_dataset["train"],
        "validation": split_dataset["test"]
    })

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length", # Uses longest sequence in a batch (saves memory)
        truncation=True,
        max_length=512,
        return_special_tokens_mask=True
    )

tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# Data collator for MLM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# Training arguments with Hugging Face Hub support
training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",  # Changed to run based on epochs
    logging_strategy="epoch",  # Logging based on epochs
    num_train_epochs=4,  # Set to 4 epochs
    max_grad_norm=1.0,
    warmup_ratio=0.1,
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    prediction_loss_only=True,
    save_strategy="epoch",  # Save based on epochs
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=3,
    push_to_hub=True,
    hub_model_id="siddharthdhara17/ClinicalModernBERT",
    hub_strategy="every_save",
    report_to="wandb", # Enabling wandb logging
    resume_from_checkpoint=True
)

# Start time for total training
start_time = time.time()

# Custom callback to log epoch training time
class TimeLoggingCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        epoch_time = state.log_history[-1]['epoch_time']
        print(f"Epoch {state.epoch} completed in {epoch_time:.2f} seconds")
        wandb.log({"epoch": state.epoch, "epoch_time (seconds)": epoch_time})

# Custom Trainer class to log training time
class TimeLoggingTrainer(Trainer):
    def train(self, resume_from_checkpoint=None):
        epoch_times = []
        for epoch in range(int(self.args.num_train_epochs)):
            epoch_start = time.time()
            print(f"Starting epoch {epoch + 1}/{int(self.args.num_train_epochs)}")
            
            # Perform training for one epoch
            train_output = super().train(resume_from_checkpoint=resume_from_checkpoint)
            
            epoch_end = time.time()
            epoch_time = epoch_end - epoch_start
            epoch_times.append(epoch_time)
            
            # Add time to log history for the current epoch
            self.state.log_history.append({"epoch_time": epoch_time, "epoch": epoch + 1})
            
            wandb.log({"epoch": epoch + 1, "epoch_time (seconds)": epoch_time})
        return epoch_times

    
# Use the modified trainer class
trainer = TimeLoggingTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    callbacks=[TimeLoggingCallback()]
)

# Trainer setup
#trainer = Trainer(
    #model=model,
    #args=training_args,
    #train_dataset=tokenized_datasets["train"],
    #eval_dataset=tokenized_datasets["validation"],
    #data_collator=data_collator
    #callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Stop if no improvement for 3 eval steps
#)

# Train model
# trainer.train(resume_from_checkpoint="./outputs/checkpoint-60000")
# trainer.train(resume_from_checkpoint="/home2/venkatasiddharth.d/outputs/checkpoint-60000")

# Train model and save checkpoints
#trainer.train()

# Train the model while logging time
epoch_times = trainer.train()

# Compute total training time
total_training_time = time.time() - start_time
print(f"Total training time: {total_training_time:.2f} seconds")

# Log total time to wandb
wandb.log({"total_training_time (seconds)": total_training_time})

# Explicitly save tokenizer and latest model checkpoint
trainer.save_model(output_dir)  # Save model
tokenizer.save_pretrained(output_dir)  # Save tokenizer

# Push the final model, tokenizer, and all saved checkpoints
if push_to_hub:
    trainer.push_to_hub(commit_message="Final trained model with tokenizer")

    # Manually push intermediate checkpoints
    for checkpoint in os.listdir(output_dir):
        checkpoint_path = os.path.join(output_dir, checkpoint)
        if checkpoint.startswith("checkpoint-") and os.path.isdir(checkpoint_path):
            print(f"Pushing {checkpoint} to HF Hub...")
            trainer.push_to_hub(commit_message=f"Pushing {checkpoint}")


# Test the fine-tuned model
predict("patient admitted with [MASK] failure")
predict("diabetes [MASK] neuropathy")
predict("chest [MASK] is normal")

wandb.finish()

