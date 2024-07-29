import os
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
import pandas as pd
import re

# Set random seed
def set_seed(seed_value=42):
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # If you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Call set_seed
set_seed(42)

# Set environment variables and parameters
NUM_GENERATIONS = 40
NUM_STORIES = 1000000
BATCH_SIZE = 2000

# Load tokenizer
tokenizer_path = "./sourcemodels/TinyStories-1M"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer.pad_token = tokenizer.eos_token

# Tokenize function
def tokenize_function(example):
    return tokenizer(example['text'], truncation=True)

# Load evaluation dataset from Hugging Face Hub
eval_dataset = load_dataset("roneneldan/TinyStories", split="validation")

# Tokenize evaluation dataset
tokenized_eval_datasets = eval_dataset.map(
    tokenize_function,
    batched=True,
    batch_size=1024,
    num_proc=16,  # Specify the number of processors to use
    remove_columns=['text']
)
tokenized_eval_datasets.set_format("torch")

print("Start training...")

# Main iteration process
for generation in range(2, NUM_GENERATIONS + 1):
    # Load trained model
    if generation == 1:
        print("Loading source model")
        model = AutoModelForCausalLM.from_pretrained("./sourcemodels/TinyStories-1M/")
        model.to("cuda")
    else:
        checkpoint = f"train_models/TinyStories-1m/gene{generation-1}"
        print(f"Loading checkpoint from {checkpoint}")
        
        model = AutoModelForCausalLM.from_pretrained(checkpoint)
        model.to("cuda")
    
    # Generate dataset
    if generation > 1:
        print(f"Start to generate dataset for generation: {generation}")
        print("Params: ", sum(p.numel() for p in model.parameters()))
        num_batches = NUM_STORIES // BATCH_SIZE
        generated_tensors = []

        with torch.no_grad():
            for _ in range(num_batches):
                input_ids = torch.tensor([[0]] * BATCH_SIZE).to("cuda")
                attention_mask = torch.tensor([[0]] * BATCH_SIZE).to("cuda")
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
        torch.save(generated_tensors, f'exper1m/generate/gene{generation-1}_generate_tensors.pt')
 
    # Load generated tensors
    tensor_save_path = f'exper1m/generate/gene{generation-1}_generate_tensors.pt'
    generated_tensors = torch.load(tensor_save_path)
    
    # Transfer tensors from GPU to CPU and convert to list
    tensor_list = [tensor.tolist() for tensor in generated_tensors]

    # Create DataFrame
    df = pd.DataFrame({'input_ids': tensor_list})

    # Create Dataset from DataFrame
    train_dataset = Dataset.from_pandas(df)
    
    # Randomly sample and decode examples, then write to file
    sample_indices = random.sample(range(len(generated_tensors)), 10)
    samples_file_path = f'exper1m/sample/gene{generation-1}_samples.txt'
    with open(samples_file_path, 'w') as samples_file:
        for idx in sample_indices:
            decoded_text = tokenizer.decode(generated_tensors[idx], skip_special_tokens=True)
            samples_file.write(100 * '-' + "\n" + decoded_text + "\n\n")
            
    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    # Set model path and initialize new model
    model_path = f"./train_models/TinyStories-1m/gene{generation}"
    config = AutoConfig.from_pretrained("./sourcemodels/TinyStories-1M/")
    model = AutoModelForCausalLM.from_config(config)

    # Check available GPU count
    num_gpus = torch.cuda.device_count()

    # Desired total effective batch size (total amount of data to be processed)
    total_effective_batch_size = 320

    # Per GPU batch size
    per_device_train_batch_size = 32
    per_device_eval_batch_size = 16

    # Calculate gradient_accumulation_steps
    gradient_accumulation_steps = max(total_effective_batch_size // (per_device_train_batch_size * num_gpus), 1)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=model_path, logging_dir='./logs',
        overwrite_output_dir=True,   
        per_device_train_batch_size=per_device_train_batch_size, 
        per_device_eval_batch_size=per_device_eval_batch_size,
        evaluation_strategy="steps", 
        eval_steps=600,
        fp16=True, 
        save_strategy="no",
        save_total_limit=1,
        learning_rate=5e-4,
        weight_decay=0.1,
        logging_steps=100, 
        num_train_epochs=3,
        adam_beta1=0.9, 
        adam_beta2 = 0.95, 
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
