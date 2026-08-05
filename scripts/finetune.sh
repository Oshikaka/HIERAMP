#!/bin/bash
# HierAmp stage 1: fine-tune a pre-trained VAR (d16) for 5 epochs with per-scale class tokens.
# The backbone and the class tokens are trained jointly (loss = VAR token CE + cls_lw * L_cls);
# only the class tokens and the shared class head are newly added.
#
# Download the pre-trained VAR first:
#   wget https://huggingface.co/FoundationVision/var/resolve/main/var_d16.pth
#
# Usage: bash scripts/finetune.sh /path/to/imagenet [num_gpus]

DATA_PATH=${1:?usage: bash scripts/finetune.sh /path/to/imagenet [num_gpus]}
NUM_GPUS=${2:-8}

torchrun --nproc_per_node="$NUM_GPUS" train.py \
  --data_path="$DATA_PATH" \
  --exp_name=hieramp_d16 \
  --depth=16 --bs=768 --ep=5 --fp16=1 --alng=1e-3 --wpe=0.1 \
  --cls_scales=0_1_2_3_4_5_6_7_8_9 --cls_lw=1.0 \
  --var_init_ckpt=var_d16.pth
