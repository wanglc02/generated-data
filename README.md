# Transformer Model Pretraining and Fine-Tuning

This repository provides scripts for pretraining, fine-tuning, and evaluating transformer models on datasets such as CoLA, SST-2, QQP, MRPC, QNLI, RTE, IMDB, and Yelp.

## Setup

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Pretraining

To pretrain a model, use the `train.py` script:

```bash
python train.py --model_size {1m|33m} --num_generations 40 --num_stories 1000000 --batch_size 2000
```

- `--model_size`: Choose "1m" or "33m".
- `--num_generations`, `--num_stories`, `--batch_size`: Configure the pretraining.

## Fine-Tuning and Evaluation

Use `run_experiment.py` to fine-tune and evaluate models:

```bash
python run_experiment.py --dataset <dataset_name> --fine_tune_dir <dir> --eval_results_dir <dir> --pretrained_models_dir <dir> --tokenizer_path <path> --num_generations 40
```

- `--dataset`: Dataset to use (`cola`, `sst2`, `qqp`, etc.).
- `--fine_tune_dir`, `--eval_results_dir`, `--pretrained_models_dir`, `--tokenizer_path`: Specify paths.
- `--num_generations`: Number of fine-tuning generations.

## Perplexity Evaluation and Comparison

To evaluate the perplexity (PPL) of models across generations and plot the results:

1. Run the evaluation for both 1m and 33m models by executing the following in your terminal:

   ```
   python evaluate_ppl.py

   ```
2. Plot the PPL comparison:
   The `evaluate_ppl.py` script will automatically generate a plot saved as `PPL_comparison.png`. This plot shows the PPL across generations for both model sizes, allowing you to visually compare their performance.

## Running All Downstream Tasks

To automate fine-tuning across all datasets:

```bash
python run_downstreamtasks.py
```

Or use the shell script for a single experiment:

```bash
bash run_fintune.sh
```
