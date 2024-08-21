import os
import subprocess
from multiprocessing import Pool

def run_experiment(model_size, dataset, pretrained_models_dir, tokenizer_path, num_train_epochs, gpu_id):
    fine_tune_dir = f"./finetunes/{model_size}/{dataset}_finetune_{num_train_epochs}"
    eval_results_dir = f"./results-{model_size}/{dataset}_eval_results_{num_train_epochs}"

    # 确保输出目录存在
    os.makedirs(fine_tune_dir, exist_ok=True)
    os.makedirs(eval_results_dir, exist_ok=True)

    # 构建命令行
    cmd = [
        "python", "run_experiment.py",
        "--do_train",
        "--do_eval",
        "--dataset", dataset,
        "--fine_tune_dir", fine_tune_dir,
        "--eval_results_dir", eval_results_dir,
        "--pretrained_models_dir", pretrained_models_dir,
        "--tokenizer_path", tokenizer_path,
        "--num_generations", "40",
        "--num_train_epochs", str(num_train_epochs),
        "--max_length", "128",
        "--train_batch_size", "8",
        "--eval_batch_size", "32"
    ]

    # 设置 CUDA 设备
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 运行命令行
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Experiment failed: {dataset} with model size {model_size} on GPU {gpu_id}")

def main():
    model_sizes = ["1m", "33m"]
    new_tasks = {
        "yelp": 0.01,
        "qqp": 0.01,
        "qnli": 0.1,
        "sst2": 0.1
    }
    pretrained_model_dirs = {
        "1m": "/home/shixianjie/generated-data/models/TinyStories-1m",
        "33m": "/home/shixianjie/generated-data/models/TinyStories-33m"
    }
    tokenizer_paths = {
        "1m": "/home/shixianjie/generated-data/sourcemodels/TinyStories-1M",
        "33m": "/home/shixianjie/generated-data/sourcemodels/TinyStories-33M"
    }

    tasks = []

    # 分配 GPU
    gpu_id = 0

    for model_size in model_sizes:
        for dataset, epochs in new_tasks.items():
            pretrained_models_dir = pretrained_model_dirs[model_size]
            tokenizer_path = tokenizer_paths[model_size]
            
            tasks.append((model_size, dataset, pretrained_models_dir, tokenizer_path, epochs, gpu_id))
            
            # 在多个 GPU 之间循环分配任务
            gpu_id = (gpu_id + 1) % 2 + 4  # 从 GPU 4 开始

    # 使用多进程并行运行实验
    with Pool(processes=min(len(tasks), 8)) as pool:  # 使用最多 8 个进程并行执行
        pool.starmap(run_experiment, tasks)

if __name__ == "__main__":
    main()
