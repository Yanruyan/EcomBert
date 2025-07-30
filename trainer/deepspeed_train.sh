#!/bin/bash

# 配置参数
NUM_GPUS=4  # 4 x 4090
DS_CONFIG="deepspeed_config.json"

# 启动DeepSpeed分布式训练（根据空闲状态选择gpu）
export CUDA_VISIBLE_DEVICES=0,1,2,3
deepspeed \
  --num_gpus $NUM_GPUS \
  pretrain_mlm_distributed.py \
  --deepspeed $DS_CONFIG
