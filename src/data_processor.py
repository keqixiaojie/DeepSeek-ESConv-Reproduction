import json
import os
import random
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
from config import Config

"""
README for Data Processor (Updated for Turn-Awareness):
--------------------------
逻辑流程:
1. load_data: 读取原始 json 文件。
2. split_data: 按照 8:1:1 (默认) 切分训练、验证、测试集。
3. process_dialogs: 核心函数。
   - 支持 mode='vanilla' / 'strategy'
   - 【新增】支持 use_turn_info=True/False
     - 如果为 True，会在 Input 开头加上 [Turn: N] 标签。
     - Turn N 是全局轮次，反映对话的绝对进度。
4. save_data: 保存为 json。
   - 普通版: train_vanilla.json
   - 增强版: train_vanilla_turn.json (带轮次信息)
"""

def load_data(file_path):
    print(f"正在读取原始数据: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_history(history):
    """
    将历史对话列表转换为字符串格式。
    """
    formatted_text = ""
    for turn in history:
        role = "User" if turn['speaker'] == 'seeker' else "Assistant"
        formatted_text += f"{role}: {turn['content']}\n"
    formatted_text += "Assistant: " 
    return formatted_text

def process_dialogs(data, mode='vanilla', use_turn_info=False):
    """
    处理对话数据，生成模型训练所需的 (input, output) 对。
    mode: 'vanilla' 或 'strategy'
    use_turn_info: 是否在 Input 中加入全局轮次信息 [Turn: N]
    """
    processed_samples = []
    
    for dialog_entry in data:
        dialog = dialog_entry['dialog']
        history = []
        
        # 【新增】全局轮次计数器 (Supporter 的第几次发言)
        global_supporter_turn_id = 0
        
        for turn in dialog:
            # 只预测 supporter (Assistant) 的回复
            if turn['speaker'] == 'supporter':
                global_supporter_turn_id += 1 # 轮次 +1
                
                # 1. 构建输入 (Input)
                # 滑动窗口截取历史 (Context 依然只看最近 N 句)
                context_turns = history[-Config.WINDOW_SIZE:] if len(history) > Config.WINDOW_SIZE else history
                base_input_text = format_history(context_turns)
                
                # 【核心修改】如果是增强模式，注入轮次感知
                if use_turn_info:
                    # 结果示例: "[Turn: 1] User: Hello..."
                    input_text = f"[Turn: {global_supporter_turn_id}] {base_input_text}"
                else:
                    input_text = base_input_text
                
                # 2. 构建输出 (Output)
                content = turn['content']
                strategy = turn["annotation"].get('strategy') 
                
                if mode == 'strategy':
                    if strategy:
                        strategy_token = f"[{strategy}]"
                        # 确保策略在我们的白名单里
                        if strategy_token in Config.STRATEGY_TOKENS:
                            output_text = f"{strategy_token} {content}"
                            processed_samples.append({
                                "input": input_text,
                                "output": output_text,
                                "strategy_label": strategy_token 
                            })
                else:
                    # Vanilla 模式
                    output_text = content
                    processed_samples.append({
                        "input": input_text,
                        "output": output_text
                    })
            
            # 将当前轮加入历史
            history.append(turn)
            
    return processed_samples

def save_data(data, filename):
    filepath = os.path.join(Config.PROCESSED_DATA_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"已保存: {filepath} (样本数: {len(data)})")

def calculate_strategy_distribution(train_samples):
    """ 计算策略分布 (Random 模式用) """
    strategies = [s['strategy_label'] for s in train_samples]
    counter = Counter(strategies)
    total = len(strategies)
    
    distribution = []
    for st in Config.STRATEGY_TOKENS:
        count = counter.get(st, 0)
        prob = count / total if total > 0 else 0
        distribution.append({"strategy": st, "probability": prob, "count": count})
    
    df = pd.DataFrame(distribution)
    save_path = os.path.join(Config.PROCESSED_DATA_DIR, "strategy_distribution.csv")
    df.to_csv(save_path, index=False)
    print(f"策略分布已保存至: {save_path}")

def main():
    # 1. 读取 & 切分
    raw_data = load_data(Config.RAW_DATA_PATH)
    
    train_data, temp_data = train_test_split(raw_data, test_size=(Config.VAL_SIZE + Config.TEST_SIZE), random_state=Config.SEED)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=Config.SEED)
    
    print(f"数据集切分完成: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # ---------------------------------------------------------
    # Group A: 原始数据 (Baseline)
    # ---------------------------------------------------------
    print("\n[Group A] 处理原始数据 (无轮次信息)...")
    save_data(process_dialogs(train_data, 'vanilla'), "train_vanilla.json")
    save_data(process_dialogs(val_data, 'vanilla'), "dev_vanilla.json")
    save_data(process_dialogs(test_data, 'vanilla'), "test_vanilla.json")
    
    train_strategy = process_dialogs(train_data, 'strategy')
    save_data(train_strategy, "train_strategy.json")
    save_data(process_dialogs(val_data, 'strategy'), "dev_strategy.json")
    save_data(process_dialogs(test_data, 'strategy'), "test_strategy.json")
    
    # 计算分布 (以原始数据为准即可，分布是一样的)
    calculate_strategy_distribution(train_strategy)

    # ---------------------------------------------------------
    # Group B: 增强数据 (Turn-Aware)
    # ---------------------------------------------------------
    print("\n[Group B] 处理增强数据 (Input带 [Turn: N])...")
    # Vanilla + Turn
    save_data(process_dialogs(train_data, 'vanilla', use_turn_info=True), "train_vanilla_turn.json")
    save_data(process_dialogs(val_data, 'vanilla', use_turn_info=True), "dev_vanilla_turn.json")
    save_data(process_dialogs(test_data, 'vanilla', use_turn_info=True), "test_vanilla_turn.json")
    
    # Strategy + Turn
    save_data(process_dialogs(train_data, 'strategy', use_turn_info=True), "train_strategy_turn.json")
    save_data(process_dialogs(val_data, 'strategy', use_turn_info=True), "dev_strategy_turn.json")
    save_data(process_dialogs(test_data, 'strategy', use_turn_info=True), "test_strategy_turn.json")
    
    print("\n所有数据预处理完成！")

if __name__ == "__main__":
    main()