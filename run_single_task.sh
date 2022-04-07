#!/bin/bash

set -x

python run_classifier.py \
    --dataset_name germeval18 \
    --dataset_config_name binary \
    --model_name_or_path bert-base-multilingual-cased \
    --tokenizer_name bert-base-multilingual-cased \
    --logging_steps 100 \
    --save_steps 10000 \
    --evaluation_strategy steps \
    --save_strategy steps \
    --num_train_epochs 1000 \
    --early_stopping_patience 2 \
    --save_total_limit 2 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 128 \
    --train_adapter \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --do_predict \
    --output_dir tmp/models/tmp \
    --logging_dir tmp/models/tmp/log
