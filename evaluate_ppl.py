import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


os.environ['http_proxy'] = "http://127.0.0.1:7890"
os.environ['https_proxy'] = "http://127.0.0.1:7890"
os.environ['all_proxy'] = "socks5://127.0.0.1:7890"

HF_ENDPOINT="https://hf-mirror.com"
os.environ["HF_ENDPOINT"] = HF_ENDPOINT
# 设置设备
device = "cuda"

# 加载数据集
def load_data(tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    test = load_dataset("roneneldan/TinyStories", split="validation")
    
    # 对每个故事分别编码，不要使用join
    encodings = [tokenizer(story, return_tensors="pt") for story in test["text"]]
    return tokenizer, encodings

# 计算PPL
def calculate_ppl(model_path, encodings):
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    max_length = model.config.max_position_embeddings
    stride = 512  # 窗口步长，控制上下文长度

    nlls = []  # 用于保存每个token的NLL
    total_tokens = 0  # 计算总token数

    for encoding in tqdm(encodings):  # 遍历每个故事
        seq_len = encoding.input_ids.size(1)
        input_ids = encoding.input_ids.to(device)

        # 计算每个故事的NLL
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc

            # 截取当前窗口的input_ids和target_ids
            input_ids_window = input_ids[:, begin_loc:end_loc]
            target_ids = input_ids_window.clone()

            # 设置要忽略的部分，-100表示忽略
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids_window, labels=target_ids)
                neg_log_likelihood = outputs.loss

            # 将当前窗口的NLL添加到nlls列表
            nlls.append(neg_log_likelihood.item() * trg_len)
            total_tokens += trg_len

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

    # 计算整体的平均PPL
    mean_nll = sum(nlls) / total_tokens
    ppl = torch.exp(torch.tensor(mean_nll))
    return ppl.item()

# 遍历生成并计算PPL
def evaluate_models(model_dir, tokenizer_path, num_generations, model_size):
    tokenizer, encodings = load_data(tokenizer_path)
    results = []

    for generation in range(1, num_generations + 1):
        if model_size == "33m":
            parent_directory = f"{model_dir}/gene{generation}"
            checkpoints = [d for d in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, d)) and d.startswith("checkpoint")]
            latest_checkpoint = max(checkpoints, key=lambda x: int(re.search(r'\d+', x).group()))
            model_path = os.path.join(parent_directory, latest_checkpoint)
        else:
            model_path = f"{model_dir}/gene{generation}"

        if os.path.isdir(model_path):
            ppl = calculate_ppl(model_path, encodings)
            results.append({"Generation": generation, "PPL": ppl})
            print(f"Model: {model_path}, PPL: {ppl}")

    # 保存结果到Excel
    output_file = f"./ppl_results_{model_size}.xlsx"
    df = pd.DataFrame(results)
    df.to_excel(output_file, index=False)

# 画图
def plot_ppl():
    file_path_1m = "./ppl_results_1m.xlsx"
    file_path_33m = "./ppl_results_33m.xlsx"

    data_1m = pd.read_excel(file_path_1m)
    data_33m = pd.read_excel(file_path_33m)

    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制1M数据
    ax.plot(data_1m['Generation'], data_1m['PPL'], label='1M', color='tab:blue', linewidth=3)

    # 绘制33M数据
    ax.plot(data_33m['Generation'], data_33m['PPL'], label='33M', color='tab:green', linewidth=3)

    # 设置标签和标题
    ax.set_xlabel('Generation', fontsize=24)
    ax.set_ylabel('Perplexity', fontsize=24)
    ax.legend(fontsize=20)

    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)

    plt.tight_layout()
    plt.savefig('PPL_comparison.png')
    plt.show()

# 主函数
def main():
    evaluate_models("./models/TinyStories-1m", "./sourcemodels/TinyStories-1M", 40, "1m")
    evaluate_models("./models/TinyStories-33m", "./sourcemodels/TinyStories-33M", 40, "33m")
    plot_ppl()

if __name__ == "__main__":
    main()
