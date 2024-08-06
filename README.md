# Fine-tuning and Evaluating Transformers on Various Datasets

This repository provides scripts to fine-tune and evaluate transformer models on various datasets including CoLA, SST-2, QQP, MRPC, QNLI, RTE, IMDB, and Yelp.

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- Datasets
- Scikit-learn
- Pandas

You can install the required packages using the following command:

```bash
pip install torch transformers datasets scikit-learn pandas
```

## Usage

To fine-tune and evaluate a model on a specific dataset, run the following command:

```bash
python run_experiment.py --dataset <dataset_name> --output_dir <output_directory> --pretrained_models_dir <pretrained_models_directory> --tokenizer_path <tokenizer_path> --num_generations <num_generations>
```

## Example

```bash
python run_experiment.py --dataset sst2 --output_dir ./results --pretrained_models_dir ./pretrained_models --tokenizer_path ./tokenizer --num_generations 40
```

## Parameters

* `--dataset`: The name of the dataset to use. Options are: `cola`, `sst2`, `qqp`, `mrpc`, `qnli`, `rte`, `imdb`, `yelp`.
* `--output_dir`: The directory to save fine-tuned models and results.
* `--pretrained_models_dir`: The directory containing pretrained models.
* `--tokenizer_path`: The path to the tokenizer.
* `--num_generations`: The number of generations to fine-tune.
* `--max_length`: The maximum sequence length. Default is 128.
* `--train_batch_size`: The training batch size. Default is 8.
* `--eval_batch_size`: The evaluation batch size. Default is 32.
* `--learning_rate`: The learning rate. Default is 2e-5.
* `--num_train_epochs`: The number of training epochs. Default is 1.
* `--weight_decay`: The weight decay. Default is 0.01.
