#!/bin/bash

set -e
set -x

mkdir -p tmp

# traiing on the external datasets
# see run_classifier.py for detailed parameters
python ./run_classifier.py \
  --run_name MULTI_HS \
  --report_to none \
  --model_name_or_path xlm-roberta-base \
  --tokenizer_name xlm-roberta-base \
  --model_type prompt \
  --logging_steps 100 \
  --save_steps 100 \
  --evaluation_strategy steps \
  --save_strategy steps \
  --num_train_epochs 1000 \
  --early_stopping_patience 10 \
  --save_total_limit 2 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-05 \
  --warmup_steps 100 \
  --train_sampling none \
  --data_selector_method_train stratify \
  --data_selector_method_eval stratify \
  --freeze_model_core 0  \
  --overwrite_output_dir \
  --seed 0 \
  --do_mlm 0 \
  --mlm_weight 1.0 \
  --do_train \
  --do_eval \
  --task_name ami18 ami18 ami18 hateval19 large_scale_abuse mlma srw16 \
  --dataset_name ami18 ami18 ami18 hateval19 large_scale_abuse mlma srw16 \
  --dataset_config_name en-misogyny en-misogyny_category en-target en-target fine_grained en-sentiment_rep fine_grained \
  --prompt_ids 1 0 0 0 0 0 0 \
  --max_train_samples -1 \
  --max_eval_samples -1 \
  --preprocess_steps no \
  --loss_functions none \
  --output_dir tmp/step1 \
  --logging_dir tmp/step1/log

# training and evaluating on the target task
python ./run_classifier.py \
  --run_name MULTI_HS \
  --report_to none \
  --model_name_or_path tmp/step1 \
  --tokenizer_name xlm-roberta-base \
  --model_type prompt \
  --logging_steps 2 \
  --save_steps 2 \
  --evaluation_strategy steps \
  --save_strategy steps \
  --num_train_epochs 100 \
  --early_stopping_patience 10 \
  --save_total_limit 2 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-05 \
  --warmup_steps 10 \
  --train_sampling none \
  --max_train_samples 4 \
  --max_eval_samples 16 \
  --data_selector_method_train per_label \
  --data_selector_method_eval stratify \
  --freeze_model_core 0  \
  --overwrite_output_dir \
  --seed 0 \
  --do_mlm 0 \
  --mlm_weight 1.0 \
  --do_train \
  --do_eval \
  --do_predict \
  --task_name hasoc19 \
  --dataset_name hasoc19 \
  --dataset_config_name en-fine_grained \
  --prompt_ids 0 \
  --preprocess_steps no \
  --loss_functions none \
  --output_dir tmp/step2 \
  --logging_dir tmp/step2/log
