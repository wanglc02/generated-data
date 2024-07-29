import os
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import random

# 配置文件路径和参数
config = {
    "train_models_output_dir": './cola_finetune',
    "pretrained_models_dir": './pretrained_models',
    "full_reports_path": './results/full_reports.txt',
    "eval_reports_path": './results/eval_reports.txt',
    "tokenizer_path": './tokenizer',
    "num_generations": 40,
    "max_length": 128,
    "train_batch_size": 8,
    "eval_batch_size": 32,
    "learning_rate": 2e-5,
    "num_train_epochs": 1,
    "weight_decay": 0.01
}

# 创建输出目录（如果不存在）
os.makedirs(os.path.dirname(config["full_reports_path"]), exist_ok=True)
os.makedirs(os.path.dirname(config["eval_reports_path"]), exist_ok=True)
os.makedirs(config["train_models_output_dir"], exist_ok=True)

# 设置随机种子
def set_seed(seed_value=42):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # 如果使用多个 GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

def fine_tune_model(model, tokenizer, train_dataset, eval_dataset, generation):
    def tokenize_function(example):
        return tokenizer(example['sentence'], padding="max_length", truncation=True, max_length=config["max_length"])
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    output_dir = f'{config["train_models_output_dir"]}/{generation}'
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["train_batch_size"],
        per_device_eval_batch_size=config["eval_batch_size"],
        num_train_epochs=config["num_train_epochs"],
        weight_decay=config["weight_decay"],
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

def evaluate_sentiment_classification_model(model, tokenizer, eval_dataset):
    def tokenize_function(example):
        return tokenizer(example['sentence'], padding="max_length", truncation=True, max_length=config["max_length"])

    tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    tokenized_eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    eval_dataloader = DataLoader(tokenized_eval_dataset, batch_size=config["eval_batch_size"], shuffle=False)

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

# 加载数据集
dataset = load_dataset("glue", "cola")
train_dataset = dataset['train']
eval_dataset = dataset['validation']

tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"])
tokenizer.pad_token = tokenizer.eos_token  # 确保定义padding token

# Fine-tune and test
for generation in range(1, config["num_generations"] + 1):
    parent_directory = f"{config["pretrained_models_dir"]}/gene{generation}"
    model = AutoModelForSequenceClassification.from_pretrained(parent_directory, num_labels=2)
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to("cuda")
    fine_tune_model(model, tokenizer, train_dataset, eval_dataset, generation)

    accuracy, report = evaluate_sentiment_classification_model(model, tokenizer, eval_dataset)
    with open(config["full_reports_path"], "a") as file:
        file.write(f"Generation: {generation}, Accuracy: {accuracy}\n")
        file.write("Classification Report:\n")
        file.write(report + "\n\n")

# Test
for generation in range(1, config["num_generations"] + 1):
    parent_directory = f"{config["train_models_output_dir"]}/{generation}"
    model = AutoModelForSequenceClassification.from_pretrained(parent_directory, num_labels=2)
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to("cuda")
    accuracy, report = evaluate_sentiment_classification_model(model, tokenizer, eval_dataset)
    with open(config["eval_reports_path"], "a") as file:
        file.write(f"Generation: {generation}, Accuracy: {accuracy}\n")
        file.write("Classification Report:\n")
        file.write(report + "\n\n")
