import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configuration
HF_ENDPOINT = "https://hf-mirror.com"
os.environ["HF_ENDPOINT"] = HF_ENDPOINT
device = "cuda"

# Directory settings
RESULT_DIR = "./result"
GRAPH_DIR = "./graph"
MODEL_DIR_1M = "./models/TinyStories-1m"
MODEL_DIR_33M = "./models/TinyStories-33m"
TOKENIZER_PATH_1M = "./sourcemodels/TinyStories-1M"
TOKENIZER_PATH_33M = "./sourcemodels/TinyStories-33M"
PLOT_SAVE_PATH = f"{GRAPH_DIR}/PPL_comparison.png"

# Evaluation settings
NUM_GENERATIONS = 40
STRIDE = 512  # Window stride to control context length

# Ensure directories exist
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)

# Load dataset
def load_data(tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    test = load_dataset("roneneldan/TinyStories", split="validation")
    
    # Encode each story separately
    encodings = [tokenizer(story, return_tensors="pt") for story in test["text"]]
    return tokenizer, encodings

# Calculate PPL
def calculate_ppl(model_path, encodings):
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    max_length = model.config.max_position_embeddings

    nlls = []  # To store the NLL of each token
    total_tokens = 0  # Count the total number of tokens

    for encoding in tqdm(encodings):  # Iterate over each story
        seq_len = encoding.input_ids.size(1)
        input_ids = encoding.input_ids.to(device)

        # Calculate NLL for each story
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, STRIDE):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc

            # Extract input_ids and target_ids for the current window
            input_ids_window = input_ids[:, begin_loc:end_loc]
            target_ids = input_ids_window.clone()

            # Set the ignored part to -100, meaning it will be ignored
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids_window, labels=target_ids)
                neg_log_likelihood = outputs.loss

            # Add the NLL of the current window to nlls list
            nlls.append(neg_log_likelihood.item() * trg_len)
            total_tokens += trg_len

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

    # Calculate the overall average PPL
    mean_nll = sum(nlls) / total_tokens
    ppl = torch.exp(torch.tensor(mean_nll))
    return ppl.item()

# Iterate to generate and calculate PPL
def evaluate_models(model_dir, tokenizer_path, num_generations, model_size):
    tokenizer, encodings = load_data(tokenizer_path)
    results = []

    for generation in range(1, num_generations + 1):
        model_path = f"{model_dir}/gene{generation}"

        if os.path.isdir(model_path):
            ppl = calculate_ppl(model_path, encodings)
            results.append({"Generation": generation, "PPL": ppl})
            print(f"Model: {model_path}, PPL: {ppl}")

    # Save results to Excel
    output_file = f"{RESULT_DIR}/ppl_results_{model_size}.xlsx"
    df = pd.DataFrame(results)
    df.to_excel(output_file, index=False)

# Plot results
def plot_ppl():
    file_path_1m = f"{RESULT_DIR}/ppl_results_1m.xlsx"
    file_path_33m = f"{RESULT_DIR}/ppl_results_33m.xlsx"

    data_1m = pd.read_excel(file_path_1m)
    data_33m = pd.read_excel(file_path_33m)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot 1M data
    ax.plot(data_1m['Generation'], data_1m['PPL'], label='1M', color='tab:blue', linewidth=3)

    # Plot 33M data
    ax.plot(data_33m['Generation'], data_33m['PPL'], label='33M', color='tab:green', linewidth=3)

    # Set labels and title
    ax.set_xlabel('Generation', fontsize=24)
    ax.set_ylabel('Perplexity', fontsize=24)
    ax.legend(fontsize=20)

    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH)
    plt.show()

# Main function
def main():
    evaluate_models(MODEL_DIR_1M, TOKENIZER_PATH_1M, NUM_GENERATIONS, "1m")
    evaluate_models(MODEL_DIR_33M, TOKENIZER_PATH_33M, NUM_GENERATIONS, "33m")
    plot_ppl()

if __name__ == "__main__":
    main()
