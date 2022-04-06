#!/bin/bash

set -x

python run_classifier.py \
    --dataset_name src/data/germeval18.py \
    --dataset_config_name binary \
    --model_name_or_path bert-base-multilingual-cased \
    --tokenizer_name bert-base-multilingual-cased \
    --logging_steps 100 \
    --evaluation_strategy steps \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --save_steps 10000 \
    --save_total_limit 1 \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --do_predict \
    --output_dir tmp/models/tmp \
    --logging_dir tmp/models/tmp/log
