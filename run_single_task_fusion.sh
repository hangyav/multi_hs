#!/bin/bash

set -x

python run_classifier.py \
    --dataset_name germeval18 \
    --dataset_config_name binary \
    --model_name_or_path bert-base-multilingual-cased \
    --tokenizer_name bert-base-multilingual-cased \
    --logging_steps 100 \
    --save_steps 100 \
    --evaluation_strategy steps \
    --save_strategy steps \
    --num_train_epochs 1000 \
    --early_stopping_patience 2 \
    --save_total_limit 2 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 128 \
    --gradient_accumulation_steps 2 \
    --train_adapter \
    --train_sampling none \
    --freeze_model_core 1 \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --do_predict \
    --load_adapter tmp/models/tmp_single_adapter/germeval18-binary,tmp/models/tmp_single_adapter/germeval18-fine_grained \
    --fuse_adapters germeval18-binary,germeval18-fine_grained \
    --output_dir tmp/models/tmp_single_fusion \
    --logging_dir tmp/models/tmp_single_fusion/log
