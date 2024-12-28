#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import random
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed
)
from datasets import load_dataset, Dataset
import pandas as pd
import re

# 设置随机种子以确保结果可复现
set_seed(42)  # 使用 transformers 提供的 set_seed 函数


# In[2]:


# 加载 Truthful QA 数据集的 "generation" 配置
dataset = load_dataset("truthfulqa/truthful_qa", "generation")

qa_dataset = dataset['validation']
qa_dataset[0]


# In[3]:


# 定义一个函数来提取所需字段
def preprocess_function(example):
    example['text'] = f"Question: {example['question']}\nAnswer: {example['best_answer']}"
    return example

# 应用预处理函数，移除不需要的列
qa_dataset = dataset.map(preprocess_function,remove_columns=dataset['validation'].column_names)
qa_dataset = qa_dataset['validation']

qa_dataset = qa_dataset.train_test_split(test_size=0.1)
train_dataset = qa_dataset['train']
test_dataset = qa_dataset['test']


# In[4]:


# 配置模型路径和训练参数
source_model_path = "../sourcemodels/TinyStories-33M"  # 确保路径正确
model_dir = "../models/TinyStories-33m-qa"

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(source_model_path)
tokenizer.pad_token = tokenizer.eos_token

# 定义分词函数
def tokenize_function(example):
    return tokenizer(example['text'], truncation=True, padding='max_length', max_length=1024, return_tensors='pt')

# 分词数据集
tokenized_datasets = qa_dataset.map(
    tokenize_function,
    batched=True,
    batch_size=1024,
    num_proc=16,
)
tokenized_datasets.set_format("torch")

train_dataset = tokenized_datasets['train']
test_dataset = tokenized_datasets['test']


# In[5]:


print("开始训练...")

# 加载初始模型
print("加载源模型")
model = AutoModelForCausalLM.from_pretrained(source_model_path)
model.to("cuda")

# 定义训练参数
num_gpus = torch.cuda.device_count()
total_effective_batch_size = 32 # 目标总有效批量大小
per_device_train_batch_size = 16
per_device_eval_batch_size = 16
gradient_accumulation_steps = max(total_effective_batch_size // (per_device_train_batch_size * num_gpus), 1)

training_args = TrainingArguments(
    output_dir=model_dir,
    logging_dir='logs',
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    evaluation_strategy="steps",
    eval_steps=500,  # 根据需要调整
    save_strategy="steps",
    save_steps=500,
    save_total_limit=1,
    learning_rate=5e-4,
    load_best_model_at_end=True,
    weight_decay=0.1,
    logging_steps=500,
    num_train_epochs=100,
    adam_beta1=0.9,
    adam_beta2=0.95,
    lr_scheduler_type="constant",
    gradient_accumulation_steps=gradient_accumulation_steps,
    fp16=True,
    report_to="none"
)

# 定义数据整理器
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# 开始训练
trainer.train()

# 保存训练好的模型
trainer.save_model(model_dir)

print("训练完成并已保存模型。")


# In[ ]:


from transformers import pipeline
#加载
generation_model = AutoModelForCausalLM.from_pretrained(model_dir) 
generation_tokenizer = AutoTokenizer.from_pretrained(model_dir)

# 创建文本生成 Pipeline
generation_pipeline = pipeline(
    "text-generation",
    model=generation_model,
    tokenizer=generation_tokenizer,
    device=0 if torch.cuda.is_available() else -1  # 使用 GPU（如果可用）
)

generation_pipeline("Question: What is the capital of France?\nAnswer:")

