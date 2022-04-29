#!/bin/bash

# this is esentially the same as run_single_task_fusion.sh but:
# * loading the adapters trained with multi task setup
# * and loading the updated bert model

set -x

python run_classifier.py \
    --dataset_name germeval18 \
    --dataset_config_name binary \
    --model_name_or_path tmp/models/tmp_multi_adapter \
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
    --do_visualize \
    --load_adapter tmp/models/tmp_multi_adapter/germeval18-binary tmp/models/tmp_multi_adapter/germeval18-fine_grained \
    --fuse_adapters germeval18-binary germeval18-fine_grained \
    --task_name germeval18-binary,germeval18-fine_grained \
    --output_dir tmp/models/tmp_multi_fusion \
    --logging_dir tmp/models/tmp_multi_fusion/log
