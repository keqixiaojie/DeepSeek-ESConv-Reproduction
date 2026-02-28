#!/bin/bash
# 激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate EscLab

# 设置 CUDA 设备 (你有 5090，通常是 0 号卡)
export CUDA_VISIBLE_DEVICES=0

# 运行训练
echo "Starting Vanilla Training..."
python src/trainer.py --mode vanilla