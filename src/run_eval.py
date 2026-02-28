# # import os
# # import json
# # import csv
# # import torch
# # import argparse
# # from tqdm import tqdm
# # from evaluator import Evaluator
# # from config import Config

# # def load_test_data(mode):
# #     """
# #     加载测试集。
# #     Vanilla 模式加载 vanilla 测试集。
# #     Strategy 相关模式 (Joint/Oracle/Random) 加载 strategy 测试集。
# #     """
# #     if mode == 'vanilla':
# #         file_path = os.path.join(Config.PROCESSED_DATA_DIR, "test_vanilla.json")
# #     else:
# #         file_path = os.path.join(Config.PROCESSED_DATA_DIR, "test_strategy.json")
    
# #     print(f"正在加载测试数据: {file_path}")
# #     with open(file_path, 'r', encoding='utf-8') as f:
# #         data = json.load(f)
# #     return data

# # def extract_ground_truth_strategy(text):
# #     """
# #     从目标文本中提取策略 token。
# #     假设格式为: "[Strategy] Response content..."
# #     """
# #     end_idx = text.find(']')
# #     if end_idx != -1:
# #         return text[:end_idx+1]
# #     return "[Others]" # 兜底

# # def run_evaluation(args):
# #     # 1. 初始化评估器
# #     # 注意：如果是 vanilla 模式，checkpoint_path 应该是 vanilla 的 output_dir
# #     # 如果是 strategy 模式，checkpoint_path 应该是 strategy 的 output_dir
# #     if args.eval_mode == 'vanilla':
# #         model_path = Config.OUTPUT_DIR_VANILLA
# #         evaluator = Evaluator(mode='vanilla', checkpoint_path=model_path)
# #     else:
# #         model_path = Config.OUTPUT_DIR_STRATEGY
# #         evaluator = Evaluator(mode='strategy', checkpoint_path=model_path)
    
# #     # 2. 加载数据
# #     # Vanilla 模式评测 vanilla 数据
# #     # Random/Joint/Oracle 模式评测 strategy 数据
# #     data_mode = 'vanilla' if args.eval_mode == 'vanilla' else 'strategy'
# #     test_data = load_test_data(data_mode)
    
# #     # 3. 准备结果文件
# #     os.makedirs("results", exist_ok=True)
# #     output_csv = f"results/result_{args.eval_mode}.csv"
    
# #     headers = ["context", "target", "generated_response", "bleu-2", "rouge-l"]
# #     if args.eval_mode == 'joint':
# #         headers.extend(["target_strategy", "pred_strategy", "strategy_match"])
    
# #     results = []
    
# #     print(f"开始评测: 模式={args.eval_mode}, 数据量={len(test_data)}")
    
# #     # 4. 循环评测
# #     # 为了演示方便，这里可以用 [:50] 先跑个 demo，正式跑去掉切片
# #     for item in tqdm(test_data): 
# #         context = item['input']
# #         target_full = item['output']
        
# #         # 提取 Ground Truth 策略 (仅 Strategy 相关模式需要)
# #         gt_strategy = None
# #         target_content = target_full
        
# #         if args.eval_mode != 'vanilla':
# #             gt_strategy = extract_ground_truth_strategy(target_full)
# #             # 去掉策略词，只保留纯文本回复用于计算 content metrics
# #             # 假设 "[Strategy] " 后面有个空格
# #             target_content = target_full.replace(gt_strategy, "").strip()
        
# #         # 生成
# #         response, pred_strategy = evaluator.generate_response(
# #             context, 
# #             method=args.eval_mode, 
# #             ground_truth_strategy=gt_strategy
# #         )
        
# #         # 计算指标
# #         metrics = evaluator.calculate_metrics([target_content], response)
        
# #         row = {
# #             "context": context,
# #             "target": target_content,
# #             "generated_response": response,
# #             "bleu-2": round(metrics['bleu-2'], 4),
# #             "rouge-l": round(metrics['rouge-l'], 4)
# #         }
        
# #         # Joint 模式额外记录策略准确性
# #         if args.eval_mode == 'joint':
# #             row["target_strategy"] = gt_strategy
# #             row["pred_strategy"] = pred_strategy
# #             row["strategy_match"] = 1 if gt_strategy == pred_strategy else 0
            
# #         results.append(row)

# #     # 5. 保存结果
# #     print(f"正在保存结果到 {output_csv}...")
# #     with open(output_csv, 'w', encoding='utf-8', newline='') as f:
# #         writer = csv.DictWriter(f, fieldnames=headers)
# #         writer.writeheader()
# #         writer.writerows(results)
        
# #     # 6. 打印平均分
# #     avg_bleu = sum(r['bleu-2'] for r in results) / len(results)
# #     avg_rouge = sum(r['rouge-l'] for r in results) / len(results)
# #     print(f"\n========== 评测报告 ({args.eval_mode}) ==========")
# #     print(f"Avg BLEU-2: {avg_bleu:.4f}")
# #     print(f"Avg ROUGE-L: {avg_rouge:.4f}")
    
# #     if args.eval_mode == 'joint':
# #         acc = sum(r['strategy_match'] for r in results) / len(results)
# #         print(f"Strategy Accuracy: {acc:.4f}")

# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser()
# #     # 允许传入 vanilla, random, joint, oracle
# #     parser.add_argument("--eval_mode", type=str, required=True, 
# #                         choices=['vanilla', 'random', 'joint', 'oracle'])
# #     args = parser.parse_args()
    
# #     run_evaluation(args)
# import os
# import json
# import csv
# import argparse
# import numpy as np
# from tqdm import tqdm
# from evaluator import Evaluator
# from config import Config
# from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# def load_test_data(mode):
#     filename = "test_vanilla.json" if mode == 'vanilla' else "test_strategy.json"
#     file_path = os.path.join(Config.PROCESSED_DATA_DIR, filename)
#     print(f"Loading Data: {file_path}")
#     with open(file_path, 'r', encoding='utf-8') as f:
#         return json.load(f)

# def extract_ground_truth_strategy(text):
#     end_idx = text.find(']')
#     if end_idx != -1: return text[:end_idx+1]
#     return "[Others]"

# def run_evaluation(args):
#     # 1. 初始化
#     model_path = Config.OUTPUT_DIR_VANILLA if args.eval_mode == 'vanilla' else Config.OUTPUT_DIR_STRATEGY
#     evaluator = Evaluator(mode='strategy' if args.eval_mode != 'vanilla' else 'vanilla', checkpoint_path=model_path)
    
#     data_mode = 'vanilla' if args.eval_mode == 'vanilla' else 'strategy'
#     test_data = load_test_data(data_mode)
    
#     # 结果容器
#     results = []
#     all_metrics = {"bleu-2": [], "rouge-l": [], "extrema": [], "ppl": []}
    
#     # 策略统计容器 (仅 Joint 模式)
#     true_strategies = []
#     pred_strategies = []
    
#     print(f"========== Start Eval: {args.eval_mode} ==========")
    
#     for item in tqdm(test_data):
#         context = item['input']
#         target_full = item['output']
        
#         # 提取真实策略和纯文本
#         gt_strategy = None
#         target_content = target_full
        
#         if args.eval_mode != 'vanilla':
#             gt_strategy = extract_ground_truth_strategy(target_full)
#             # target_content = target_full.replace(gt_strategy, "").strip()
#             # 【修复方案】使用切片，严格保留原句的空格和格式
#             # 只有当 strategy 在开头时才切分
#             if target_full.startswith(gt_strategy):
#                 target_content = target_full[len(gt_strategy):] 
#             else:
#                 target_content = target_full.replace(gt_strategy, "") # 兜底
        
#         # === Step 1: 生成 (Generation) ===
#         # 这部分逻辑是对的，我们在 generate_response 里处理了拼接
#         response, pred_strat = evaluator.generate_response(
#             context, method=args.eval_mode, ground_truth_strategy=gt_strategy
#         )
        
#         # === Step 2: 计算生成指标 (PPL修复核心) ===
#         ppl = 0.0
#         if args.eval_mode != 'joint': # Joint 依然跳过
#             try:
#                 # 【关键修复】
#                 # 1. Vanilla: Context -> Response
#                 # 2. Oracle/Random: Context + [Strategy] -> Response
                
#                 ppl_context = context
                
#                 if args.eval_mode in ['oracle', 'random']:
#                     # Oracle 模式下，Context 必须包含策略词！
#                     # 注意：我们要把 [Strategy] 视为 Context 的一部分，mask 掉它的 loss
#                     # 我们手动把策略拼在 Context 后面
#                     ppl_context = context + gt_strategy + " "
#                     # 注意：这里可能需要加个空格，取决于 tokenizer 行为，
#                     # 但考虑到 add_special_tokens，直接拼通常没问题
                
#                 # 计算 PPL：输入是 (Context + Strategy)，预测目标是 (Target Content)
#                 ppl = evaluator.calculate_ppl(ppl_context, target_content)
                
#             except Exception as e:
#                 # print(f"PPL Error: {e}") # 调试用
#                 ppl = float('nan')
        
#         # Content Metrics
#         metrics = evaluator.calculate_metrics([target_content], response)
        
#         # 记录分数
#         all_metrics["bleu-2"].append(metrics['bleu-2'])
#         all_metrics["rouge-l"].append(metrics['rouge-l'])
#         all_metrics["extrema"].append(metrics['extrema'])
#         if args.eval_mode != 'joint':
#             all_metrics["ppl"].append(ppl)
            
#         # 记录策略 (Joint)
#         if args.eval_mode == 'joint':
#             true_strategies.append(gt_strategy)
#             # 简单的清洗，防止模型生成了奇怪的策略词
#             if pred_strat not in Config.STRATEGY_TOKENS:
#                 pred_strat = "[Others]"
#             pred_strategies.append(pred_strat)

#         # 保存 CSV 行
#         row = {
#             "context": context, "target": target_content, "generated": response,
#             "bleu-2": metrics['bleu-2'], "rouge-l": metrics['rouge-l'], "extrema": metrics['extrema'],
#             "ppl": ppl if args.eval_mode != 'joint' else "-"
#         }
#         if args.eval_mode == 'joint':
#             row["target_strategy"] = gt_strategy
#             row["pred_strategy"] = pred_strat
#         results.append(row)

#     # === Step 3: 聚合报告 ===
#     avg_metrics = {k: np.mean(v) for k, v in all_metrics.items() if v}
    
#     final_report = {
#         "mode": args.eval_mode,
#         "sample_size": len(results),
#         "generation_metrics": avg_metrics
#     }
    
#     # 策略指标计算
#     if args.eval_mode == 'joint':
#         acc = accuracy_score(true_strategies, pred_strategies)
#         f1 = f1_score(true_strategies, pred_strategies, average='macro', zero_division=0)
#         final_report["strategy_metrics"] = {
#             "accuracy": acc,
#             "macro_f1": f1
#         }
#         # 混淆矩阵 (转为 list 方便 JSON 序列化)
#         labels = Config.STRATEGY_TOKENS
#         cm = confusion_matrix(true_strategies, pred_strategies, labels=labels)
#         final_report["confusion_matrix"] = cm.tolist()
#         final_report["labels"] = labels
        
#     # === Step 4: 保存文件 ===
#     os.makedirs("results", exist_ok=True)
    
#     # 保存 CSV (详细 Case)
#     csv_path = f"results/details_{args.eval_mode}.csv"
#     keys = results[0].keys()
#     with open(csv_path, 'w', encoding='utf-8', newline='') as f:
#         writer = csv.DictWriter(f, fieldnames=keys)
#         writer.writeheader()
#         writer.writerows(results)
        
#     # 保存 JSON (宏观报告)
#     json_path = f"results/report_{args.eval_mode}.json"
#     with open(json_path, 'w', encoding='utf-8') as f:
#         json.dump(final_report, f, indent=4)
        
#     print(f"\n========== 评测完成 ==========")
#     print(json.dumps(final_report, indent=4))
#     print(f"详情已保存至: {csv_path}")
#     print(f"报告已保存至: {json_path}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--eval_mode", type=str, required=True, choices=['vanilla', 'random', 'joint', 'oracle'])
#     args = parser.parse_args()
#     run_evaluation(args)
import os
import json
import csv
import argparse
import numpy as np
from tqdm import tqdm
from evaluator import Evaluator
from config import Config
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def load_test_data(mode, use_turn=False):
    """
    加载测试数据
    mode: 'vanilla' | 'strategy'
    use_turn: 是否使用带轮次信息的测试集
    """
    # 动态确定文件名后缀
    suffix = "_turn" if use_turn else ""
    
    filename = f"test_vanilla{suffix}.json" if mode == 'vanilla' else f"test_strategy{suffix}.json"
    file_path = os.path.join(Config.PROCESSED_DATA_DIR, filename)
    
    print(f"Loading Data: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到测试文件: {file_path}。请检查是否已运行 data_processor.py 生成带 turn 的数据。")
        
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_ground_truth_strategy(text):
    end_idx = text.find(']')
    if end_idx != -1: return text[:end_idx+1]
    return "[Others]"

def run_evaluation(args):
    # 0. 准备后缀 (用于输出文件命名)
    suffix = "_turn" if args.use_turn else ""
    
    # 1. 确定模型路径
    base_dir = Config.OUTPUT_DIR_VANILLA if args.eval_mode == 'vanilla' else Config.OUTPUT_DIR_STRATEGY
    # 如果开启 turn 模式，模型路径也需要加后缀
    if args.use_turn:
        model_path = f"{base_dir}{suffix}"
    else:
        model_path = base_dir
        
    print(f"========== Start Eval: {args.eval_mode} (Turn-Aware: {args.use_turn}) ==========")
    print(f"Model Path: {model_path}")

    # 初始化评估器
    # 注意：传给 Evaluator 的 mode 依然是 'vanilla' 或 'strategy'，不需要带 turn
    # 因为 Evaluator 内部只关心要不要 resize embedding，turn 只是 input 的文本变化，不改变模型架构逻辑
    evaluator_mode = 'strategy' if args.eval_mode != 'vanilla' else 'vanilla'
    evaluator = Evaluator(mode=evaluator_mode, checkpoint_path=model_path)
    
    # 2. 加载数据
    data_mode = 'vanilla' if args.eval_mode == 'vanilla' else 'strategy'
    test_data = load_test_data(data_mode, use_turn=args.use_turn)
    
    # 结果容器
    results = []
    all_metrics = {"bleu-2": [], "rouge-l": [], "extrema": [], "ppl": []}
    
    # 策略统计容器 (仅 Joint 模式)
    true_strategies = []
    pred_strategies = []
    
    for item in tqdm(test_data):
        context = item['input']
        target_full = item['output']
        
        # 提取真实策略和纯文本
        gt_strategy = None
        target_content = target_full
        
        if args.eval_mode != 'vanilla':
            gt_strategy = extract_ground_truth_strategy(target_full)
            
            # 【保留你的修复方案】使用切片，严格保留原句的空格和格式
            # 只有当 strategy 在开头时才切分
            if target_full.startswith(gt_strategy):
                target_content = target_full[len(gt_strategy):] 
            else:
                target_content = target_full.replace(gt_strategy, "") # 兜底
        
        # === Step 1: 生成 (Generation) ===
        response, pred_strat = evaluator.generate_response(
            context, method=args.eval_mode, ground_truth_strategy=gt_strategy
        )
        
        # === Step 2: 计算生成指标 ===
        ppl = 0.0
        if args.eval_mode != 'joint': # Joint 依然跳过
            try:
                ppl_context = context
                
                if args.eval_mode in ['oracle', 'random']:
                    # Oracle 模式 PPL 构造逻辑
                    # 你的观察是对的：StrategyToken 和 Content 之间可能需要空格
                    # 假设 target_content 已经保留了前导空格（因为用了切片），
                    # 那么 context + gt_strategy 拼接后，接上 target_content 应该就顺畅了。
                    # 但为了保险，可以显式加个空格，或者依赖 target_content 自带的空格。
                    
                    # 方案：context + [Strategy]
                    # 注意：如果 tokenizer.add_special_tokens 没加空格，这里可能需要手动微调
                    # 你之前加了 + " " 效果好，那我们就保留 + " "，
                    # 除非 target_content 本身已经有空格了，那就可能双空格。
                    # 最稳妥的方式：
                    ppl_context = context + gt_strategy
                    if not target_content.startswith(" "):
                         ppl_context += " "
                
                # 计算 PPL
                ppl = evaluator.calculate_ppl(ppl_context, target_content)
                
            except Exception as e:
                # print(f"PPL Error: {e}") 
                ppl = float('nan')
        
        # Content Metrics
        metrics = evaluator.calculate_metrics([target_content], response)
        
        # 记录分数
        all_metrics["bleu-2"].append(metrics['bleu-2'])
        all_metrics["rouge-l"].append(metrics['rouge-l'])
        all_metrics["extrema"].append(metrics['extrema'])
        if args.eval_mode != 'joint':
            all_metrics["ppl"].append(ppl)
            
        # 记录策略 (Joint)
        if args.eval_mode == 'joint':
            true_strategies.append(gt_strategy)
            # 简单的清洗
            if pred_strat not in Config.STRATEGY_TOKENS:
                pred_strat = "[Others]"
            pred_strategies.append(pred_strat)

        # 保存 CSV 行
        row = {
            "context": context, "target": target_content, "generated": response,
            "bleu-2": metrics['bleu-2'], "rouge-l": metrics['rouge-l'], "extrema": metrics['extrema'],
            "ppl": ppl if args.eval_mode != 'joint' else "-"
        }
        if args.eval_mode == 'joint':
            row["target_strategy"] = gt_strategy
            row["pred_strategy"] = pred_strat
        results.append(row)

    # === Step 3: 聚合报告 ===
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items() if v}
    
    final_report = {
        "mode": args.eval_mode,
        "use_turn": args.use_turn, # 记录是否开启了 Turn 模式
        "sample_size": len(results),
        "generation_metrics": avg_metrics
    }
    
    # 策略指标计算
    if args.eval_mode == 'joint':
        acc = accuracy_score(true_strategies, pred_strategies)
        f1 = f1_score(true_strategies, pred_strategies, average='macro', zero_division=0)
        final_report["strategy_metrics"] = {
            "accuracy": acc,
            "macro_f1": f1
        }
        # 混淆矩阵
        labels = Config.STRATEGY_TOKENS
        cm = confusion_matrix(true_strategies, pred_strategies, labels=labels)
        final_report["confusion_matrix"] = cm.tolist()
        final_report["labels"] = labels
        
    # === Step 4: 保存文件 ===
    os.makedirs("results", exist_ok=True)
    
    # 这里的命名也加上 suffix，防止覆盖
    csv_path = f"results/details_{args.eval_mode}{suffix}.csv"
    json_path = f"results/report_{args.eval_mode}{suffix}.json"
    
    keys = results[0].keys()
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
        
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=4)
        
    print(f"\n========== 评测完成 ==========")
    print(json.dumps(final_report, indent=4))
    print(f"详情已保存至: {csv_path}")
    print(f"报告已保存至: {json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_mode", type=str, required=True, choices=['vanilla', 'random', 'joint', 'oracle'])
    # 【新增】开关参数，不加就是 False，加了就是 True
    parser.add_argument("--use_turn", action="store_true", help="是否使用带轮次信息的数据/模型进行评测")
    
    args = parser.parse_args()
    run_evaluation(args)