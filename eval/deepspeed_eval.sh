#!/bin/bash

# 配置参数
NUM_GPUS=1  # 1 x 4090
DS_CONFIG="ds_config_eval.json"

# 启动DeepSpeed分布式训练（根据空闲状态选择gpu，可用的是0～5）
export CUDA_VISIBLE_DEVICES=0
deepspeed \
  --num_gpus $NUM_GPUS \
  cpt_roberta_eval_distibuted.py \
  --deepspeed $DS_CONFIG
