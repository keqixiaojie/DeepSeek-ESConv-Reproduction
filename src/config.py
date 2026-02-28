import os
import torch

class Config:
    # ================= 路径配置 =================
    # 项目根目录 (假设当前脚本在 src 下运行，向上一级)
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 数据路径
    RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "ESConv.json")
    PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
    
    # 模型输出路径
    OUTPUT_DIR_VANILLA = os.path.join(PROJECT_ROOT, "models", "output_vanilla")
    OUTPUT_DIR_STRATEGY = os.path.join(PROJECT_ROOT, "models", "output_strategy")
    
    # 基础模型路径 (我们将使用 DeepSeek-R1-Distill-Qwen-7B)
    # 如果你手动下载了，请改为本地绝对路径
    # 【修改这里】基础模型路径
    # 使用 os.path.join 拼接，确保路径绝对正确
    MODEL_NAME_OR_PATH = os.path.join(
        PROJECT_ROOT, 
        "models", 
        "deepseek_base", 
        "deepseek-ai", 
        "DeepSeek-R1-Distill-Qwen-7B"
    )

    # ================= 策略配置 =================
    # 这是你要求的 8 个策略标签，必须严格对应
    STRATEGY_TOKENS = [
        "[Question]",
        "[Restatement or Paraphrasing]",
        "[Reflection of feelings]",
        "[Self-disclosure]",
        "[Affirmation and Reassurance]",
        "[Providing Suggestions]",
        "[Information]",
        "[Others]"
    ]

    # ================= 训练超参数 =================
    MAX_LENGTH = 1024        # 输入序列最大长度
    WINDOW_SIZE = 5          # 历史对话轮数 (你要求的5轮)
    TEST_SIZE = 0.1          # 测试集比例
    VAL_SIZE = 0.1           # 验证集比例
    SEED = 42                # 随机种子，保证每次切分一样

    # LoRA 参数
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    
    # 训练参数
    BATCH_SIZE = 1           # 5090显存很大，可以设大点，但为了稳妥先设4
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 3
    
    # 【新增】热身比例，前 10% 的步数用来热身
    WARMUP_RATIO = 0.1
    
    # 设备自动检测
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 创建目录的辅助函数
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

# 初始化时自动创建必要的目录
ensure_dir(Config.PROCESSED_DATA_DIR)
ensure_dir(Config.OUTPUT_DIR_VANILLA)
ensure_dir(Config.OUTPUT_DIR_STRATEGY)