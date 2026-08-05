#!/bin/bash
# HierAmp stage 2: generate the distilled dataset with the paper configuration
# (additive logit bias beta=5, top-rho=50%, scales 1-9, layer 14).
#
# Usage: bash scripts/sample.sh checkpoints/hieramp_d16.pth output/

CKPT=${1:?usage: bash scripts/sample.sh <hieramp_ckpt> <out_root>}
OUT=${2:?usage: bash scripts/sample.sh <hieramp_ckpt> <out_root>}

# ImageNet-1K, IPC 1 / 10 / 50 / 100
for IPC in 1 10 50 100; do
  python sample.py --var_ckpt "$CKPT" --save_dir "$OUT/in1k_ipc${IPC}" --ipc "$IPC"
done

# Plain VAR baseline (no amplification), IPC 10
python sample.py --var_ckpt "$CKPT" --save_dir "$OUT/in1k_ipc10_baseline" --ipc 10 --no_amplify

# Stage-aware schedule example: different rho for coarse/mid/fine stages
# python sample.py --var_ckpt "$CKPT" --save_dir "$OUT/in1k_ipc10_stagewise" --ipc 10 \
#   --beta 5 --top_ratio 0.3 0.5 0.7
