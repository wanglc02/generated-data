import os
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import random

# Set output directory paths
train_models_output_dir = './sst2_finetune'
pretrained_models_dir = 'path/to/pretrained/models'  # Replace with the actual pretrained models directory
full_reports_path = './results/full_reports_sst2.txt'
eval_reports_path = './results/eval_reports_sst2.txt'
tokenizer_path = 'path/to/tokenizer'  # Replace with the actual tokenizer directory

# Create output directories if they do not exist
os.makedirs(os.path.dirname(full_reports_path), exist_ok=True)
os.makedirs(os.path.dirname(eval_reports_path), exist_ok=True)
os.makedirs(train_models_output_dir, exist_ok=True)

# Set random seed
def set_seed(seed_value=42):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # If using multiple GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

def fine_tune_model(model, tokenizer, train_dataset, eval_dataset, generation):
    def tokenize_function(example):
        return tokenizer(example['sentence'], padding="max_length", truncation=True, max_length=128)
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    output_dir = f'{train_models_output_dir}/{generation}'
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=1,
        weight_decay=0.01,
        save_total_limit=5,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)

def evaluate_model(model, tokenizer, eval_dataset):
    def tokenize_function(example):
        return tokenizer(example['sentence'], padding="max_length", truncation=True, max_length=128)

    tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    tokenized_eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    eval_dataloader = DataLoader(tokenized_eval_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    predictions = []
    labels = []

    with torch.no_grad():
        for batch in eval_dataloader:
            inputs = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["label"].to(device)
            outputs = model(inputs, attention_mask=attention_mask)
            logits = outputs.logits
            batch_predictions = torch.argmax(logits, dim=1).cpu().numpy()
            batch_labels = labels_batch.cpu().numpy()
            predictions.extend(batch_predictions)
            labels.extend(batch_labels)

    accuracy = accuracy_score(labels, predictions)
    report = classification_report(labels, predictions, zero_division=0)
    return accuracy, report

# Load SST-2 dataset
sst2_dataset = load_dataset("glue", "sst2")
sst2_train_dataset = sst2_dataset['train']
sst2_eval_dataset = sst2_dataset['validation']

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer.pad_token = tokenizer.eos_token

NUM_GENERATIONS = 40
# Fine-tune and test on SST-2
for generation in range(1, NUM_GENERATIONS + 1):
    parent_directory = f"{pretrained_models_dir}/gene{generation}"
    model = AutoModelForSequenceClassification.from_pretrained(parent_directory)
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to("cuda")
    fine_tune_model(model, tokenizer, sst2_train_dataset, sst2_eval_dataset, generation)

    accuracy, report = evaluate_model(model, tokenizer, sst2_eval_dataset)
    with open(full_reports_path, "a") as file:
        file.write(f"Generation: {generation}, Dataset: SST-2, Accuracy: {accuracy}\n")
        file.write("Classification Report:\n")
        file.write(report + "\n\n")

for generation in range(1, NUM_GENERATIONS + 1):
    parent_directory = f"{train_models_output_dir}/{generation}"
    model = AutoModelForSequenceClassification.from_pretrained(parent_directory)
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to("cuda")
    accuracy, report = evaluate_model(model, tokenizer, sst2_eval_dataset)
    with open(eval_reports_path, "a") as file:
        file.write(f"Generation: {generation}, Dataset: SST-2, Accuracy: {accuracy}\n")
        file.write("Classification Report:\n")
        file.write(report + "\n\n")
