import os
import argparse
import importlib
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate models on various datasets")
    parser.add_argument("--dataset", type=str, required=True, choices=["cola", "sst2", "qqp", "mrpc", "qnli", "rte", "imdb", "yelp"], help="Dataset to use for fine-tuning and evaluation")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save fine-tuned models and results")
    parser.add_argument("--pretrained_models_dir", type=str, required=True, help="Directory containing pretrained models")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer")
    parser.add_argument("--num_generations", type=int, default=40, help="Number of generations to fine-tune")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    return parser.parse_args()

def main():
    args = parse_args()

    # 动态导入相应的数据集模块
    dataset_module = importlib.import_module(f"downstream_tasks.{args.dataset}")

    # 设置随机种子
    dataset_module.set_seed()

    # 加载数据集
    if args.dataset in ["cola", "sst2", "qqp", "mrpc", "qnli", "rte"]:
        dataset = load_dataset("glue", args.dataset)
        train_dataset = dataset['train']
        eval_dataset = dataset['validation']
    elif args.dataset == "imdb":
        dataset = load_dataset("imdb")
        train_dataset = dataset['train']
        eval_dataset = dataset['test'].shuffle(seed=42).select(range(min(len(dataset['test']), 5000)))  # 限制测试集大小
    elif args.dataset == "yelp":
        dataset = load_dataset("yelp_review_full")
        train_dataset = dataset['train']
        eval_dataset = dataset['test']

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token  # 确保定义padding token

    full_reports = []
    eval_reports = []

    # Fine-tune and test
    for generation in range(1, args.num_generations + 1):
        parent_directory = f"{args.pretrained_models_dir}/gene{generation}"
        model = AutoModelForSequenceClassification.from_pretrained(parent_directory, num_labels=5 if args.dataset == "yelp" else 2)
        model.config.pad_token_id = tokenizer.pad_token_id
        model = model.to("cuda")

        output_dir = os.path.join(args.output_dir, f"{args.dataset}_finetune", f"generation_{generation}")
        dataset_module.fine_tune_model(model, tokenizer, train_dataset, eval_dataset, output_dir, args.max_length, args.train_batch_size, args.eval_batch_size, args.learning_rate, args.num_train_epochs, args.weight_decay)

        accuracy, report = dataset_module.evaluate_model(model, tokenizer, eval_dataset, args.max_length, args.eval_batch_size)
        report['Generation'] = generation
        report['Accuracy'] = accuracy
        full_reports.append(report)

    # 将结果转换为 DataFrame 并保存为 Excel 文件
    full_reports_df = pd.json_normalize(full_reports, sep='_')
    full_reports_path = os.path.join(args.output_dir, f"full_reports_{args.dataset}.xlsx")
    full_reports_df.to_excel(full_reports_path, index=False)

if __name__ == "__main__":
    main()
