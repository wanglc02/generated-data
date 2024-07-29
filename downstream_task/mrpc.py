import os
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import random

# 设置输出目录路径
train_models_output_dir = './mrpc_finetune'
pretrained_models_dir = './pretrained_models'
full_reports_path = './results/full_reports_mrpc.txt'
eval_reports_path = './results/eval_reports_mrpc.txt'
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
        return tokenizer(example['sentence1'], example['sentence2'], padding="max_length", truncation=True, max_length=128)
    
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
        per_device_train_batch_size=64,  # 训练批次大小
        per_device_eval_batch_size=64,  # 评估批次大小
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
        return tokenizer(example['sentence1'], example['sentence2'], padding="max_length", truncation=True, max_length=128)
    
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

# 加载MRPC数据集
mrpc_dataset = load_dataset("glue", "mrpc")
mrpc_train_dataset = mrpc_dataset['train']
mrpc_eval_dataset = mrpc_dataset['validation']

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer.pad_token = tokenizer.eos_token  # 确保定义padding token

NUM_GENERATIONS = 40
# Fine-tune and test on MRPC
for generation in range(1, NUM_GENERATIONS + 1):
    parent_directory = f"{pretrained_models_dir}/gene{generation}"
    model = AutoModelForSequenceClassification.from_pretrained(parent_directory)
    model.config.pad_token_id = tokenizer.pad_token_id  # 设置模型的padding token
    model = model.to("cuda")
    fine_tune_model(model, tokenizer, mrpc_train_dataset, mrpc_eval_dataset, generation)

    accuracy, report = evaluate_model(model, tokenizer, mrpc_eval_dataset)
    with open(full_reports_path, "a") as file:  # 使用追加模式'a'
        file.write(f"Generation: {generation}, Accuracy: {accuracy}\n")
        file.write("Classification Report:\n")
        file.write(report + "\n\n")
