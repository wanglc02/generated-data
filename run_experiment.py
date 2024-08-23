import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import argparse
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoConfig
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 导入所有预处理函数
from preprocessing import (
    preprocess_qqp, preprocess_cola, preprocess_imdb,
    preprocess_mrpc, preprocess_qnli, preprocess_rte,
    preprocess_sst2, preprocess_yelp
)

def set_seed(seed_value=42):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # 如果使用多个 GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate models on various datasets")
    parser.add_argument("--dataset", type=str, required=True, choices=["cola", "sst2", "qqp", "mrpc", "qnli", "rte", "imdb", "yelp"], help="Dataset to use for fine-tuning and evaluation")
    parser.add_argument("--fine_tune_dir", type=str, required=True, help="Directory to save fine-tuned models")
    parser.add_argument("--eval_results_dir", type=str, required=True, help="Directory to save evaluation results")
    parser.add_argument("--pretrained_models_dir", type=str, required=True, help="Directory containing pretrained models")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer")
    parser.add_argument("--num_generations", type=int, default=40, help="Number of generations to fine-tune")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=float, default=1.0, help="Number of training epochs, can be an integer or a float")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--do_train", action='store_true', help="Whether to fine-tune the model")
    parser.add_argument("--do_eval", action='store_true', help="Whether to evaluate the model")
    return parser.parse_args()

def load_data(dataset_name):
    if dataset_name in ["cola", "sst2", "qqp", "mrpc", "qnli", "rte"]:
        dataset = load_dataset("glue", dataset_name)
        train_dataset = dataset['train']
        eval_dataset = dataset['validation']
    elif dataset_name == "imdb":
        dataset = load_dataset("imdb")
        train_dataset = dataset['train']
        eval_dataset = dataset['test'].shuffle(seed=42).select(range(min(len(dataset['test']), 5000)))
    elif dataset_name == "yelp":
        dataset = load_dataset("yelp_review_full")
        train_dataset = dataset['train']
        eval_dataset = dataset['test']
    return train_dataset, eval_dataset

def fine_tune_model(model, tokenizer, train_dataset, eval_dataset, args, generation):
    output_dir = os.path.join(args.fine_tune_dir, f'generation_{generation}')
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_train_epochs,  # 支持浮点数的训练 epoch 数
        save_strategy="no",  # 不自动保存检查点,
        learning_rate=args.learning_rate,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)  # 保存模型权重

def evaluate_model(model, tokenizer, eval_dataset, args, generation):
    preprocess_functions = {
        "qqp": lambda example: tokenizer(example['question1'], example['question2'], padding="max_length", truncation=True, max_length=args.max_length),
        "cola": lambda example: tokenizer(example['sentence'], padding="max_length", truncation=True, max_length=args.max_length),
        "imdb": lambda example: tokenizer(example['text'], padding="max_length", truncation=True, max_length=args.max_length),
        "mrpc": lambda example: tokenizer(example['sentence1'], example['sentence2'], padding="max_length", truncation=True, max_length=args.max_length),
        "qnli": lambda example: tokenizer(example['question'], example['sentence'], padding="max_length", truncation=True, max_length=args.max_length),
        "rte": lambda example: tokenizer(example['sentence1'], example['sentence2'], padding="max_length", truncation=True, max_length=args.max_length),
        "sst2": lambda example: tokenizer(example['sentence'], padding="max_length", truncation=True, max_length=args.max_length),
        "yelp": lambda example: tokenizer(example['text'], padding="max_length", truncation=True, max_length=args.max_length),
    }

    tokenize_function = preprocess_functions.get(args.dataset)
    if tokenize_function is None:
        raise ValueError(f"No tokenize function found for dataset {args.dataset}")

    tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    tokenized_eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    eval_dataloader = DataLoader(tokenized_eval_dataset, batch_size=args.eval_batch_size, shuffle=False)

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
    report = classification_report(labels, predictions, output_dict=True, zero_division=0)

    report_df = pd.DataFrame(report).transpose().stack().to_frame().T
    report_df.columns = [f"{col[1]}_{col[0]}" for col in report_df.columns]
    
    report_df.insert(0, 'Generation', generation)
    report_df.insert(1, 'Accuracy', accuracy)

    return report_df

def find_min_max_median(df, metric, is_percentage=True):
    metric_values = df[metric]
    if is_percentage:
        metric_values *= 100  # Convert metric to percentage if needed

    min_val = metric_values.min()
    max_val = metric_values.max()
    median_val = (min_val + max_val) / 2

    return min_val, max_val, median_val

def plot_metric_vs_generation(all_results_df, dataset_name, output_dir, metric, is_percentage=True):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get generations excluding baseline
    generations = all_results_df[all_results_df['Generation'] != 'baseline']['Generation'].astype(int)
    metric_values = all_results_df[all_results_df['Generation'] != 'baseline'][metric]

    baseline_metric = all_results_df[all_results_df['Generation'] == 'baseline'][metric].values[0]
    
    if is_percentage:
        metric_values *= 100
        baseline_metric *= 100

    # Plot the metric values across generations
    ax.plot(generations, metric_values, linestyle='-', marker='o', linewidth=4, label=f'{metric.replace("_", " ").title()}')

    # Add a horizontal line for the baseline metric
    ax.axhline(y=baseline_metric, color='black', linestyle='--', linewidth=2, label=f'Baseline: {baseline_metric:.4f}')

    # Set y-axis limits based on metric values
    min_val, max_val, median_val = find_min_max_median(all_results_df, metric, is_percentage)
    y_lower_limit = median_val - 15 if is_percentage else median_val - 0.15
    y_upper_limit = median_val + 15 if is_percentage else median_val + 0.15
    ax.set_ylim(y_lower_limit, y_upper_limit)

    # Set labels, title, and ticks
    ax.set_xlabel('Generation', fontsize=24)
    ylabel = f'{metric.replace("_", " ").title()} (%)' if is_percentage else r'$F_1$ score'
    ax.set_ylabel(ylabel, fontsize=24)
    ax.set_title(f'{dataset_name.upper()} - {metric.replace("_", " ").title()} vs Generation', fontsize=26, weight='bold')
    ax.tick_params(axis='both', which='major', labelsize=18)

    # Format y-axis labels
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x)}' if is_percentage else f'{x:.2f}'))

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Remove background grid
    ax.grid(False)

    # Add legend
    ax.legend(fontsize=18)
    
    # Save the plot as SVG
    plot_path = os.path.join(output_dir, f"{dataset_name}_{metric.lower()}_vs_generation.svg")
    plt.savefig(plot_path, format='svg')
    plt.close()

def main():
    args = parse_args()
    
    set_seed()

    # 创建目标文件夹
    os.makedirs(args.fine_tune_dir, exist_ok=True)
    os.makedirs(args.eval_results_dir, exist_ok=True)

    # 加载数据集
    train_dataset, eval_dataset = load_data(args.dataset)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token

    preprocess_functions = {
        "qqp": preprocess_qqp,
        "cola": preprocess_cola,
        "imdb": preprocess_imdb,
        "mrpc": preprocess_mrpc,
        "qnli": preprocess_qnli,
        "rte": preprocess_rte,
        "sst2": preprocess_sst2,
        "yelp": preprocess_yelp,
    }

    preprocess_fn = preprocess_functions.get(args.dataset)
    if preprocess_fn is None:
        raise ValueError(f"No preprocessing function found for dataset {args.dataset}")

    train_dataset = preprocess_fn(tokenizer, train_dataset, args.max_length)
    eval_dataset = preprocess_fn(tokenizer, eval_dataset, args.max_length)

    all_results_df = pd.DataFrame()

    if args.do_eval:
        config = AutoConfig.from_pretrained(args.tokenizer_path, num_labels=5 if args.dataset == "yelp" else 2)
        baseline_model = AutoModelForSequenceClassification.from_config(config)
        baseline_model.config.pad_token_id = tokenizer.pad_token_id

        fine_tune_model(baseline_model, tokenizer, train_dataset, eval_dataset, args, "baseline")

        baseline_results_df = evaluate_model(baseline_model, tokenizer, eval_dataset, args, "baseline")
        all_results_df = pd.concat([all_results_df, baseline_results_df], axis=0)

    for generation in range(1, args.num_generations + 1):
        if args.do_train:
            parent_directory = f"{args.pretrained_models_dir}/gene{generation}"
            model = AutoModelForSequenceClassification.from_pretrained(parent_directory, num_labels=5 if args.dataset == "yelp" else 2)
            model.config.pad_token_id = tokenizer.pad_token_id
            model = model.to("cuda")

            fine_tune_model(model, tokenizer, train_dataset, eval_dataset, args, generation)

        if args.do_eval:
            generation_results_df = evaluate_model(model, tokenizer, eval_dataset, args, generation)
            all_results_df = pd.concat([all_results_df, generation_results_df], axis=0)

    summary_excel_path = os.path.join(args.eval_results_dir, f"{args.dataset}_evaluation_results.xlsx")
    all_results_df.to_excel(summary_excel_path, index=False)
    
    # 绘制并保存图表
    plot_metric_vs_generation(all_results_df, args.dataset, args.eval_results_dir, 'Accuracy', is_percentage=True)
    plot_metric_vs_generation(all_results_df, args.dataset, args.eval_results_dir, 'f1-score_weighted avg', is_percentage=False)

if __name__ == "__main__":
    main()
