import os
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
import pandas as pd
import re
import argparse

# Set random seed
def set_seed(seed_value=42):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Call set_seed
set_seed(42)

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train 1m or 33m model.")
    parser.add_argument("--model_size", type=str, choices=["1m", "33m"], required=True, help="Model size to train: '1m' or '33m'.")
    parser.add_argument("--num_generations", type=int, default=40, help="Number of generations.")
    parser.add_argument("--num_stories", type=int, default=1000000, help="Number of stories to generate.")
    parser.add_argument("--batch_size", type=int, default=2000, help="Batch size for generation.")
    return parser.parse_args()

# Parse arguments
args = parse_args()

# Set paths and parameters based on model size
if args.model_size == "1m":
    tokenizer_path = "./sourcemodels/TinyStories-1M"
    model_dir = "train_models/TinyStories-1m"
    
elif args.model_size == "33m":
    tokenizer_path = "./sourcemodels/TinyStories-33M"
    model_dir = "train_models/TinyStories-33m"
    
batch_size = args.batch_size

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer.pad_token = tokenizer.eos_token

# Tokenize function
def tokenize_function(example):
    return tokenizer(example['text'], truncation=True)

# Load evaluation dataset
eval_dataset = load_dataset("roneneldan/TinyStories", split="validation")

# Tokenize evaluation dataset
tokenized_eval_datasets = eval_dataset.map(
    tokenize_function,
    batched=True,
    batch_size=1024,
    num_proc=16,
    remove_columns=['text']
)
tokenized_eval_datasets.set_format("torch")

print("Start training...")

# Main iteration process
for generation in range(1, args.num_generations + 1):
    # Load trained model
    if generation == 1:
        print("Loading source model")
        model = AutoModelForCausalLM.from_pretrained(tokenizer_path)
    else:
        checkpoint_dir = f"{model_dir}/gene{generation-1}"
        if args.model_size == "33m":
            checkpoints = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d)) and d.startswith("checkpoint")]
            latest_checkpoint = max(checkpoints, key=lambda x: int(re.search(r'\d+', x).group()))
            checkpoint = os.path.join(checkpoint_dir, latest_checkpoint)
        else:
            checkpoint = checkpoint_dir

        print(f"Loading checkpoint from {checkpoint}")
        model = AutoModelForCausalLM.from_pretrained(checkpoint)

    model.to("cuda")

    # Generate dataset
    if generation > 1:
        print(f"Start to generate dataset for generation: {generation}")
        print("Params: ", sum(p.numel() for p in model.parameters()))
        num_batches = args.num_stories // batch_size
        generated_tensors = []

        with torch.no_grad():
            for _ in range(num_batches):
                input_ids = torch.tensor([[0]] * batch_size).to("cuda")
                attention_mask = torch.tensor([[0]] * batch_size).to("cuda")
                generated_outputs = model.generate(
                    input_ids,
                    do_sample=True,
                    max_new_tokens=600,
                    top_k=0,
                    attention_mask=attention_mask,
                )
                processed_tensors = [tensor[1:] for tensor in generated_outputs]
                generated_tensors.extend(processed_tensors)

        # Save generated token tensors
        torch.save(generated_tensors, f'{model_dir}/generate/gene{generation-1}_generate_tensors.pt')

    # Load generated tensors
    tensor_save_path = f'{model_dir}/generate/gene{generation-1}_generate_tensors.pt'
    generated_tensors = torch.load(tensor_save_path)

    # Transfer tensors from GPU to CPU and convert to list
    tensor_list = [tensor.tolist() for tensor in generated_tensors]

    # Create DataFrame
    df = pd.DataFrame({'input_ids': tensor_list})

    # Create Dataset from DataFrame
    train_dataset = Dataset.from_pandas(df)

    # Randomly sample and decode examples, then write to file
    sample_indices = random.sample(range(len(generated_tensors)), 10)
    samples_file_path = f'{model_dir}/sample/gene{generation-1}_samples.txt'
    with open(samples_file_path, 'w') as samples_file:
        for idx in sample_indices:
            decoded_text = tokenizer.decode(generated_tensors[idx], skip_special_tokens=True)
            samples_file.write(100 * '-' + "\n" + decoded_text + "\n\n")

    # Clear CUDA cache
    torch.cuda.empty_cache()

    # Set model path and initialize new model
    model_path = f"{model_dir}/gene{generation}"
    config = AutoConfig.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_config(config)

    # Detect number of GPUs
    num_gpus = torch.cuda.device_count()

    # Desired total effective batch size
    total_effective_batch_size = 320

    # Per GPU batch size
    per_device_train_batch_size = 32
    per_device_eval_batch_size = 16

    # Calculate gradient accumulation steps
    gradient_accumulation_steps = max(total_effective_batch_size // (per_device_train_batch_size * num_gpus), 1)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=model_path, logging_dir='logs',
        overwrite_output_dir=True,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        evaluation_strategy="steps",
        fp16=True,
        save_strategy="no" 
        save_total_limit=5,
        learning_rate=5e-4,
        weight_decay=0.1,
        logging_steps=500 
        num_train_epochs=3,
        adam_beta1=0.9,
        adam_beta2=0.95,
        lr_scheduler_type="constant",
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    # Define and start training
    trainer = Trainer(
        model, training_args,
        train_dataset=train_dataset,
        eval_dataset=tokenized_eval_datasets,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.save_model(model_path)
