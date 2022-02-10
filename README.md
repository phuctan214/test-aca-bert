# Training BERT with Compute/Time (Academic) Budget

This repository contains scripts for pre-training and finetuning BERT-like models with limited time and compute budget.
The code is based on the work presented in the following paper:

Peter Izsak, Moshe Berchansky, Omer Levy, [How to Train BERT with an Academic Budget](https://aclanthology.org/2021.emnlp-main.831.pdf) (EMNLP 2021).

## Installation

The pre-training and finetuning scripts are based on [Deepspeed](https://github.com/microsoft/DeepSpeed) and HuggingFace [Transformers](https://github.com/huggingface/transformers) libraries.

### Preliminary Installation

We recommend creating a virtual environment with python 3.6+, PyTorch and [`apex`](https://github.com/NVIDIA/apex).

### Installation Requirements
```bash
pip install -r requirements.txt
```

We suggest running Deepspeed's utility `ds_report` and verify Deepspeed components can be compiled (JIT).

## Pretraining

Pretraining script: `run_pretraining.py`

For all possible pretraining arguments see: `python run_pretraining.py -h`

We highly suggest reviewing the various [training features](#time-based-training) we provide within the library.

##### Example for training with the best configuration presented in our paper (24-layers/1024H/time-based learning rate schedule/fp16):

```bash
deepspeed run_pretraining.py \
  --model_type bert-mlm --tokenizer_name bert-large-uncased \
  --hidden_act gelu \
  --hidden_size 1024 \
  --num_hidden_layers 24 \
  --num_attention_heads 16 \
  --intermediate_size 4096 \
  --hidden_dropout_prob 0.1 \
  --attention_probs_dropout_prob 0.1 \
  --encoder_ln_mode pre-ln \
  --lr 1e-3 \
  --train_batch_size 4096 \
  --train_micro_batch_size_per_gpu 32 \
  --lr_schedule time \
  --curve linear \
  --warmup_proportion 0.06 \
  --gradient_clipping 0.0 \
  --optimizer_type adamw \
  --weight_decay 0.01 \
  --adam_beta1 0.9 \
  --adam_beta2 0.98 \
  --adam_eps 1e-6 \
  --total_training_time 24.0 \
  --early_exit_time_marker 24.0 \
  --dataset_path <dataset path> \
  --output_dir /tmp/training-out \
  --print_steps 100 \
  --num_epochs_between_checkpoints 10000 \
  --job_name pretraining_experiment \
  --project_name budget-bert-pretraining \
  --validation_epochs 3 \
  --validation_epochs_begin 1 \
  --validation_epochs_end 1 \
  --validation_begin_proportion 0.05 \
  --validation_end_proportion 0.01 \
  --validation_micro_batch 16 \
  --deepspeed \
  --data_loader_type dist \
  --do_validation \
  --use_early_stopping \
  --early_stop_time 180 \
  --early_stop_eval_loss 6 \
  --seed 42 \
  --fp16
```

### Time-based Training

Pretraining can be limited to a time-based value by defining `--total_training_time=24.0` (24 hours for example).

### Checkpoints and Finetune Checkpoints

There are 2 types of checkpoints that can be enabled:

- Training checkpoint - saves model weights, optimizer state and training args. Defined by `--num_epochs_between_checkpoints`.
- Finetuning checkpoint - saves model weights and configuration to be used for finetuning later on. Defined by `--finetune_time_markers`.

`finetune_time_markers` can be assigned multiple points in the training time-budget by providing a list of time markers of the overall training progress. For example `--finetune_time_markers=0.5` will save a finetuning checkpoint when reaching 50% of training time budget. For multiple finetuning checkpoints, use commas without space `0.5,0.6,0.9`.

### Validation Scheduling

Enable validation while pre-training with `--do_validation`

Control the number of epochs between validation runs with `--validation_epochs=<num>`

To control the amount of validation runs in the beginning and end (running more that `validation_epochs`) use `validation_begin_proportion` and `validation_end_proportion` to specify the proportion of time and, `validation_epochs_begin` and `validation_epochs_end` to control the custom values accordingly. 

### Mixed Precision Training

Mixed precision is supported by adding `--fp16`. Use `--fp16_backend=ds` to use Deepspeed's mixed precision backend and `--fp16_backend=apex` for `apex` (`--fp16_opt` controls optimization level).

## Finetuning

Use `run_glue.py` to run finetuning for a saved checkpoint on GLUE tasks. 

The finetuning script is identical to the one provided by Huggingface with the addition of our model.

For all possible pretraining arguments see: `python run_glue.py -h`

##### Example for finetuning on MRPC:

```bash
python run_glue.py \
  --model_name_or_path <path to model> \
  --task_name MRPC \
  --max_seq_length 128 \
  --output_dir /tmp/finetuning \
  --overwrite_output_dir \
  --do_train --do_eval \
  --evaluation_strategy steps \
  --per_device_train_batch_size 32 --gradient_accumulation_steps 1 \
  --per_device_eval_batch_size 32 \
  --learning_rate 5e-5 \
  --weight_decay 0.01 \
  --eval_steps 50 --evaluation_strategy steps \
  --max_grad_norm 1.0 \
  --num_train_epochs 5 \
  --lr_scheduler_type polynomial \
  --warmup_steps 50
```
