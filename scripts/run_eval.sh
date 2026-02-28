#!/bin/bash
# 激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate EscLab

# 设置 CUDA
export CUDA_VISIBLE_DEVICES=0

echo "====================================================="
echo "STARTING EVALUATION PIPELINE"
echo "====================================================="

# 1. 评测 Vanilla 模式 (确保你已经跑过 run_train_vanilla.sh 并且有权重)
echo "Step 1: Evaluating Vanilla Mode..."
python src/run_eval.py --eval_mode vanilla

# 2. 评测 Strategy 模型 - Joint 模式 (受限解码)
echo "Step 2: Evaluating Joint Mode (Constrained Decoding)..."
python src/run_eval.py --eval_mode joint

# 3. 评测 Strategy 模型 - Random 模式 (随机策略)
echo "Step 3: Evaluating Random Mode..."
python src/run_eval.py --eval_mode random

# 4. 评测 Strategy 模型 - Oracle 模式 (强制正确策略)
echo "Step 4: Evaluating Oracle Mode..."
python src/run_eval.py --eval_mode oracle

echo "All evaluations finished! Check 'results/' folder."