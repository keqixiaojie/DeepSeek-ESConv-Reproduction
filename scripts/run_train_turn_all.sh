#!/bin/bash
# 激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate EscLab

# 设置 CUDA 设备
export CUDA_VISIBLE_DEVICES=0

# 运行训练
# echo "Starting Strategy Training(with turn)..."
# python src/trainer.py --mode strategy --use_turn

echo "Starting Vanilla Training(with turn)..."
python src/trainer.py --mode vanilla --use_turn


