#!/bin/bash
# 激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate EscLab

# 设置 CUDA 设备
export CUDA_VISIBLE_DEVICES=0

# 运行训练
echo "Starting Strategy Training..."
python src/trainer.py --mode strategy