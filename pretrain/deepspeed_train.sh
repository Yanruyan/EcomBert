#!/bin/bash

# NCCL优化参数（适用于4090多卡通信）
#export NCCL_IB_DISABLE=1
#export NCCL_P2P_DISABLE=0
#export NCCL_DEBUG=INFO
#export NCCL_SOCKET_IFNAME=eth0  # 如果有多网卡，可指定正确的网卡名

# 其他可选优化（根据实际网络和集群情况调整）
# export NCCL_NET_GDR_LEVEL=2

# 配置参数
NUM_GPUS=1  # 1 x 4090
DS_CONFIG="deepspeed_config.json"

# 启动DeepSpeed分布式训练（根据空闲状态选择gpu，可用的是0～5）
export CUDA_VISIBLE_DEVICES=0
deepspeed \
  --num_gpus $NUM_GPUS \
  pretrain_mlm_distributed.py \
  --deepspeed $DS_CONFIG
