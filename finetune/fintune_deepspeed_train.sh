#!/bin/bash

# 配置参数
NUM_GPUS=1  # 1 x 4090
DS_CONFIG="finetune_ds_config.json"

# 启动DeepSpeed分布式训练（根据空闲状态选择gpu）
export CUDA_VISIBLE_DEVICES=0
deepspeed \
  --num_gpus $NUM_GPUS \
  finetune_roberta_classification.py \
  --local_rank 0 \
  --train_file "../data/intention_train.txt" \
  --test_file "../data/intention_test.txt" \
  --model_name_or_path "../roberta_ecommerce_ckpt" \
  --output_dir "../roberta_intention_classification_ckpt" \
  --deepspeed $DS_CONFIG \
  --epochs 3 \
  --batch_size 32 \
  --max_length 128 \
  --learning_rate 2e-5 \
  --weight_decay 0.01 \
  --warmup_steps 280
