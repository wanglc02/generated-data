#!/bin/bash

# 设置默认的参数
DATASET="cola"
NUM_TRAIN_EPOCHS=0.01
FINE_TUNE_DIR="./finetunes/${DATASET}_finetune_${NUM_TRAIN_EPOCHS}"
EVAL_RESULTS_DIR="./results-33m/${DATASET}_eval_results_${NUM_TRAIN_EPOCHS}"
PRETRAINED_MODELS_DIR="/home/shixianjie/llm-evolution/train_models/TinyStories-1m"
TOKENIZER_PATH="/home/shixianjie/llm-evolution/sourcemodels/TinyStories-1M"
NUM_GENERATIONS=40
MAX_LENGTH=128
TRAIN_BATCH_SIZE=8
EVAL_BATCH_SIZE=32
LEARNING_RATE=5e-5
WEIGHT_DECAY=0
GPU_ID=1  # 修改为你想使用的 GPU ID

# 执行 Python 脚本
CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/run_experiment.py \
    --do_train \
    --do_eval \
    --dataset $DATASET \
    --fine_tune_dir $FINE_TUNE_DIR \
    --eval_results_dir $EVAL_RESULTS_DIR \
    --pretrained_models_dir $PRETRAINED_MODELS_DIR \
    --tokenizer_path $TOKENIZER_PATH \
    --num_generations $NUM_GENERATIONS \
    --max_length $MAX_LENGTH \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --weight_decay $WEIGHT_DECAY
