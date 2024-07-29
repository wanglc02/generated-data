import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
import random
import pandas as pd
import re

# Set environment variables and parameters
NUM_GENERATIONS = 40
NUM_STORIES = 1000000
BATCH_SIZE = 200

# Load the tokenizer
tokenizer_path = "./sourcemodels/TinyStories-33M"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer.pad_token = tokenizer.eos_token

# Tokenize function
def tokenize_function(example):
    return tokenizer(example['text'], truncation=True)

# Load evaluation dataset from Hugging Face Hub
eval_dataset = load_dataset("roneneldan/TinyStories", split="validation")

# Tokenize the evaluation dataset
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
for generation in range(1, NUM_GENERATIONS + 1):
    # Load the trained model
    parent_directory = f"train_models/TinyStories-33m/gene{generation-1}"
    checkpoints = [d for d in os.listdir(parent_directory) if \
                os.path.isdir(os.path.join(parent_directory, d)) and d.startswith("checkpoint")]
    # Find the latest checkpoint
    latest_checkpoint = max(checkpoints, key=lambda x: int(re.search(r'\d+', x).group()))
    checkpoint = os.path.join(parent_directory, latest_checkpoint)

    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    model.to("cuda")

    print("generation: " + str(generation))
    print("params: ", sum(p.numel() for p in model.parameters()))
    
    # Generate dataset if generation is greater than 1
    if generation > 1:
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

        # Save the generated token tensors
        torch.save(generated_tensors, f'path/to/generated_tensors/gene{generation-1}_generate_tensors.pt')
 
    # Load the generated tensors
    tensor_save_path = f'path/to/generated_tensors/gene{generation-1}_generate_tensors.pt'
    generated_tensors = torch.load(tensor_save_path)
    
    # Transfer tensors from GPU to CPU and convert to list
    tensor_list = [tensor.tolist() for tensor in generated_tensors]

    # Create DataFrame
    df = pd.DataFrame({'input_ids': tensor_list})

    # Create Dataset from DataFrame
    train_dataset = Dataset.from_pandas(df)
    
    # Randomly sample and decode examples, then write to file
    sample_indices = random.sample(range(len(generated_tensors)), 10)
    samples_file_path = f'path/to/samples/gene{generation-1}_samples.txt'
    with open(samples_file_path, 'w') as samples_file:
        for idx in sample_indices:
            decoded_text = tokenizer.decode(generated_tensors[idx], skip_special_tokens=True)
            samples_file.write(100 * '-' + "\n" + decoded_text + "\n\n")
            
    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    # Set model path and initialize new model
    model_path = f"train_models/TinyStories-33m/gene{generation}"
    config = AutoConfig.from_pretrained("path/to/config")
    model = AutoModelForCausalLM.from_config(config)

    # Detect the number of available GPUs
    num_gpus = torch.cuda.device_count()

    # Desired total effective batch size
    total_effective_batch_size = 320

    # Per-GPU batch size
    per_device_train_batch_size = 32
    per_device_eval_batch_size = 16

    # Calculate gradient accumulation steps
    gradient_accumulation_steps = max(total_effective_batch_size // (per_device_train_batch_size * num_gpus), 1)

    # Define training parameters
    training_args = TrainingArguments(
        output_dir=model_path, logging_dir='logs',
        overwrite_output_dir=True,   
        per_device_train_batch_size=per_device_train_batch_size, 
        per_device_eval_batch_size=per_device_eval_batch_size,
        evaluation_strategy="steps", 
        eval_steps=2000,
        fp16=True, 
        save_steps=375,
        save_total_limit=5, 
        learning_rate=5e-4,
        weight_decay=0.1,
        logging_steps=500, 
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
