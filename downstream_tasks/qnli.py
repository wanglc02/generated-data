import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import random

# 设置输出目录路径
train_models_output_dir = './qnli_finetune'
pretrained_models_dir = './pretrained_models'
full_reports_path = './results/full_reports.txt'
eval_reports_path = './results/eval_reports.txt'
tokenizer_path = './tokenizer'

# 创建输出目录（如果不存在）
os.makedirs(os.path.dirname(full_reports_path), exist_ok=True)
os.makedirs(os.path.dirname(eval_reports_path), exist_ok=True)
os.makedirs(train_models_output_dir, exist_ok=True)

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
        return tokenizer(example['question'], example['sentence'], padding="max_length", truncation=True, max_length=128)
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    output_dir = f'{train_models_output_dir}/{generation}'
    os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在

    training_args = TrainingArguments(
        output_dir=output_dir,  # 输出目录
        evaluation_strategy="epoch",  # 评估策略
        learning_rate=2e-5,  # 学习率
        per_device_train_batch_size=8,  # 训练批次大小
        per_device_eval_batch_size=8,  # 评估批次大小
        num_train_epochs=1,  # 训练轮数
        weight_decay=0.01,  # 权重衰减
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
    # 保存模型权重
    trainer.save_model(output_dir)

def evaluate_model(model, tokenizer, eval_dataset):
    def tokenize_function(example):
        return tokenizer(example['question'], example['sentence'], padding="max_length", truncation=True, max_length=128)

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

def extract_accuracy_from_report(text_file_path):
    # 打开文本文件
    with open(text_file_path, 'r') as file:
        lines = file.readlines()

    # 创建列表以保存提取的数据
    generations = []
    accuracies = []

    # 循环遍历文件中的每一行并提取数据
    for line in lines:
        if line.strip().startswith('Generation'):
            # 提取世代号和准确率
            parts = line.split(',')
            generation = int(parts[0].split(':')[1].strip())
            accuracy = float(parts[1].split(':')[1].strip())
            generations.append(generation)
            accuracies.append(accuracy)

    # 创建一个DataFrame来保存提取的数据
    df = pd.DataFrame({
        'Generation': generations,
        'Accuracy': accuracies
    })

    return df

# 加载数据集
dataset = load_dataset("glue", "qnli")
train_dataset = dataset['train']
eval_dataset = dataset['validation']

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer.pad_token = tokenizer.eos_token  # 确保定义padding token

NUM_GENERATIONS = 40
# Fine-tune and test
for generation in range(1, NUM_GENERATIONS + 1):
    parent_directory = f"{pretrained_models_dir}/gene{generation}"
    model = AutoModelForSequenceClassification.from_pretrained(parent_directory, num_labels=2)
    model.config.pad_token_id = tokenizer.pad_token_id  # 设置模型的padding token
    model = model.to("cuda")
    fine_tune_model(model, tokenizer, train_dataset, eval_dataset, generation)

    accuracy, report = evaluate_model(model, tokenizer, eval_dataset)
    with open(full_reports_path, "a") as file:  # 使用追加模式'a'
        file.write(f"Generation: {generation}, Accuracy: {accuracy}\n")
        file.write("Classification Report:\n")
        file.write(report + "\n\n")

# Test
for generation in range(1, NUM_GENERATIONS + 1):
    parent_directory = f"{train_models_output_dir}/{generation}"
    model = AutoModelForSequenceClassification.from_pretrained(parent_directory, num_labels=2)
    model.config.pad_token_id = tokenizer.pad_token_id  # 设置模型的padding token
    model = model.to("cuda")
    accuracy, report = evaluate_model(model, tokenizer, eval_dataset)
    with open(eval_reports_path, "a") as file:  # 使用追加模式'a'
        file.write(f"Generation: {generation}, Accuracy: {accuracy}\n")
        file.write("Classification Report:\n")
        file.write(report + "\n\n")
