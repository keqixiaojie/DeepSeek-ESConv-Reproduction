# import os
# import torch
# import json
# import csv
# import numpy as np
# from tqdm import tqdm
# from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
# from peft import PeftModel
# from config import Config
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from rouge import Rouge 

# # 确保安装了 rouge: pip install rouge

# class StrategyConstraintLogitsProcessor(LogitsProcessor):
#     """
#     【Joint 模式核心组件】
#     这是一个“过滤器”。在模型生成文本的每一步，它都会被调用。
#     我们的目标是：如果这是生成的第一个 Token，强制它必须是 8 个策略词之一。
#     """
#     def __init__(self, tokenizer, strategy_ids):
#         self.tokenizer = tokenizer
#         self.strategy_ids = strategy_ids # 策略词对应的 Token ID 列表
#         self.first_token_generated = False # 状态标记：是否已经生成了第一个词

#     def __call__(self, input_ids, scores):
#         # input_ids: 当前已有的输入 [batch_size, seq_len]
#         # scores: 模型预测下一个词的概率分数 [batch_size, vocab_size]
        
#         # 逻辑：如果是第一次生成（还没生成过策略词），我们就干预
#         if not self.first_token_generated:
#             # 1. 创建一个全为负无穷的 mask (意思是所有词的概率都归零)
#             mask = torch.full_like(scores, float('-inf'))
            
#             # 2. 只把策略词的位置填回原来的分数
#             # 这样模型就只能看到这 8 个词的概率，其他词全是 -inf
#             mask[:, self.strategy_ids] = scores[:, self.strategy_ids]
            
#             # 3. 更新状态，下次就不拦截了
#             self.first_token_generated = True
#             return mask
#         else:
#             # 如果不是第一个词，直接放行，原样返回 scores
#             return scores

# class Evaluator:
#     def __init__(self, mode, checkpoint_path=None):
#         self.mode = mode 
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
#         # 1. 加载 Tokenizer
#         print(f"[{mode.upper()}] Loading Tokenizer from {Config.MODEL_NAME_OR_PATH}...")
#         self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME_OR_PATH, trust_remote_code=True)
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
            
#         # Strategy 模式下，必须确保分词器里有策略词
#         if mode == 'strategy':
#             self.tokenizer.add_special_tokens({'additional_special_tokens': Config.STRATEGY_TOKENS})
#             self.strategy_ids = self.tokenizer.convert_tokens_to_ids(Config.STRATEGY_TOKENS)
#             print(f"Strategy IDs loaded: {self.strategy_ids}")
        
#         # Random 模式概率加载逻辑 (不变，省略...)
#         # ... (保留你之前的 CSV 读取代码) ...

#         # 2. 加载 Base Model
#         print(f"[{mode.upper()}] Loading Base Model...")
#         base_model = AutoModelForCausalLM.from_pretrained(
#             Config.MODEL_NAME_OR_PATH,
#             torch_dtype=torch.float16, 
#             device_map="auto",
#             trust_remote_code=True,
#             attn_implementation="sdpa"
#         )
        
#         # 【关键修正点】
#         # 只有 Strategy 模式训练时改了 Embedding 大小，所以只有它需要 resize
#         # Vanilla 模式训练时没改，加载时也不能改，否则 LoRA 权重维度对不上！
#         if mode == 'strategy':
#             print("Resizing token embeddings for Strategy mode...")
#             base_model.resize_token_embeddings(len(self.tokenizer))

#         # 3. 加载 LoRA 权重 (不管是 Vanilla 还是 Strategy 都要加载！)
#         if checkpoint_path:
#             print(f"[{mode.upper()}] Loading LoRA Adapter from {checkpoint_path}...")
#             self.model = PeftModel.from_pretrained(base_model, checkpoint_path)
#         else:
#             print("⚠️ Warning: No checkpoint path provided! Using Base Model only.")
#             self.model = base_model 

#         self.model.eval()
#         self.rouge = Rouge()

#     def generate_response(self, context, method="vanilla", ground_truth_strategy=None):
#         """
#         核心生成函数
#         method: 'vanilla' | 'joint' | 'oracle' | 'random'
#         """
#         inputs = self.tokenizer(context, return_tensors="pt").to(self.device)
#         input_ids = inputs.input_ids
        
#         gen_kwargs = {
#             "max_new_tokens": 128,
#             "do_sample": True,
#             "top_p": 0.9,
#             "temperature": 0.7,
#             "pad_token_id": self.tokenizer.pad_token_id,
#             "eos_token_id": self.tokenizer.eos_token_id
#         }

#         # === 核心逻辑：根据不同模式处理输入 ===
        
#         if method == "vanilla":
#             # 啥都不用做，直接生成
#             pass
            
#         elif method == "oracle":
#             # 【Oracle 逻辑】作弊模式
#             # 强制把正确答案的策略词拼在输入后面
#             if ground_truth_strategy not in Config.STRATEGY_TOKENS:
#                 ground_truth_strategy = "[Others]" # 兜底
            
#             # 把 "[Question]" 转成 ID
#             strategy_id = self.tokenizer.convert_tokens_to_ids(ground_truth_strategy)
#             # 拼接到 input_ids 尾部: [Context IDs] + [Strategy ID]
#             prefix = torch.tensor([[strategy_id]], device=self.device)
#             input_ids = torch.cat([input_ids, prefix], dim=1)
            
#         elif method == "random":
#             # 【Random 逻辑】盲猜模式
#             # 1. 从 self.strategy_probs 里提取 keys 和 values
#             strategies = list(self.strategy_probs.keys())
#             probs = list(self.strategy_probs.values())
#             # 归一化概率 (防止浮点误差导致和不为1)
#             probs = np.array(probs)
#             probs /= probs.sum()
            
#             # 2. 按概率随机抽一个策略
#             sampled_strategy = np.random.choice(strategies, p=probs)
            
#             # 3. 像 Oracle 一样拼接到输入后面
#             strategy_id = self.tokenizer.convert_tokens_to_ids(sampled_strategy)
#             prefix = torch.tensor([[strategy_id]], device=self.device)
#             input_ids = torch.cat([input_ids, prefix], dim=1)

#         elif method == "joint":
#             # 【Joint 逻辑】考试模式
#             # 不给提示，但强迫模型第一个词必须选策略
#             # 挂载我们写的 LogitsProcessor
#             logits_processor = LogitsProcessorList([
#                 StrategyConstraintLogitsProcessor(self.tokenizer, self.strategy_ids)
#             ])
#             gen_kwargs["logits_processor"] = logits_processor

#         # === 执行生成 ===
#         with torch.no_grad():
#             outputs = self.model.generate(input_ids, **gen_kwargs)
        
#         # === 后处理：提取回复 ===
#         # 注意：inputs.input_ids 是原始输入长度。
#         # 如果是 Oracle/Random，我们刚才手动加了一个词，所以 inputs.input_ids 变长了。
#         # 我们这里用 outputs[0] 的切片来获取新生成的内容。
        
#         # 这里的 input_len 应该是 generate 接收到的 input_ids 的长度
#         input_len = input_ids.shape[1] 
#         generated_tokens = outputs[0][input_len:] 
#         response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
#         # 提取模型预测的策略 (仅 Joint 模式需要单独返回)
#         pred_strategy = None
#         if method == "joint":
#             # 在 Joint 模式下，outputs 里包含的第一个新生成 token 就是策略
#             # outputs[0] 是完整序列，[input_len - 1] 是刚才生成的第一个 token 吗？
#             # 不，是 input_len 位置。
#             # 原始 input 长度是 input_len (未包含生成部分)
#             # 所以 outputs[0][input_len] 是第一个生成的 token
            
#             # 但稍微注意一下，Joint 模式下 input_ids 没有被我们在 python 层面修改
#             # 所以 outputs[0][原始input长度] 就是生成的第一个词
#             first_gen_idx = inputs.input_ids.shape[1]
#             first_token_id = outputs[0][first_gen_idx].item()
#             pred_strategy = self.tokenizer.decode([first_token_id])
            
#         return response_text, pred_strategy

#     def calculate_metrics(self, references, hypothesis):
#         """ 计算 BLEU/ROUGE 指标 """
#         ref_tokens = [ref.split() for ref in references]
#         hyp_tokens = hypothesis.split()
        
#         # BLEU-2
#         try:
#             bleu2 = sentence_bleu(ref_tokens, hyp_tokens, weights=(0.5, 0.5), smoothing_function=SmoothingFunction().method1)
#         except:
#             bleu2 = 0.0
        
#         # Rouge-L
#         try:
#             # rouge 库要求输入是字符串
#             scores = self.rouge.get_scores(hypothesis, references[0])
#             rouge_l = scores[0]['rouge-l']['f']
#         except:
#             rouge_l = 0.0
            
#         return {"bleu-2": bleu2, "rouge-l": rouge_l}
import os
import torch
import json
import csv
import numpy as np
import math
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
from peft import PeftModel
from config import Config
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge 

# 辅助计算余弦相似度
from torch.nn.functional import cosine_similarity

class StrategyConstraintLogitsProcessor(LogitsProcessor):
    """ Joint 模式核心：首词策略约束 """
    def __init__(self, tokenizer, strategy_ids):
        self.tokenizer = tokenizer
        self.strategy_ids = strategy_ids
        self.first_token_generated = False

    def __call__(self, input_ids, scores):
        if not self.first_token_generated:
            mask = torch.full_like(scores, float('-inf'))
            mask[:, self.strategy_ids] = scores[:, self.strategy_ids]
            self.first_token_generated = True
            return mask
        return scores

class Evaluator:
    def __init__(self, mode, checkpoint_path=None):
        self.mode = mode 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. 加载 Tokenizer
        print(f"[{mode.upper()}] Loading Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME_OR_PATH, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        if mode == 'strategy':
            self.tokenizer.add_special_tokens({'additional_special_tokens': Config.STRATEGY_TOKENS})
            self.strategy_ids = self.tokenizer.convert_tokens_to_ids(Config.STRATEGY_TOKENS)

        # 加载 Random 模式概率 (略，同前文，保持你的CSV读取逻辑)
        self.strategy_probs = {}
        dist_path = "data/processed/strategy_distribution.csv" 
        if os.path.exists(dist_path):
            with open(dist_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.strategy_probs[row['strategy']] = float(row['probability'])

        # 2. 加载模型
        print(f"[{mode.upper()}] Loading Model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_NAME_OR_PATH,
            torch_dtype=torch.float16, 
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa"
        )
        
        if mode == 'strategy':
            base_model.resize_token_embeddings(len(self.tokenizer))

        if checkpoint_path:
            print(f"Loading Adapter: {checkpoint_path}")
            self.model = PeftModel.from_pretrained(base_model, checkpoint_path)
        else:
            self.model = base_model 

        self.model.eval()
        self.rouge = Rouge()
        
        # 获取 Embedding 层用于计算 Extrema
        self.embedding_layer = self.model.get_input_embeddings()

    def generate_response(self, context, method="vanilla", ground_truth_strategy=None):
        """ 生成回复 (同前文逻辑) """
        inputs = self.tokenizer(context, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        
        gen_kwargs = {
            "max_new_tokens": 128,
            "do_sample": True,
            "top_p": 0.9,
            "temperature": 0.7,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id
        }

        # 策略注入逻辑
        if method == "oracle":
            if ground_truth_strategy not in Config.STRATEGY_TOKENS: ground_truth_strategy = "[Others]"
            strat_id = self.tokenizer.convert_tokens_to_ids(ground_truth_strategy)
            prefix = torch.tensor([[strat_id]], device=self.device)
            input_ids = torch.cat([input_ids, prefix], dim=1)
            
        elif method == "random":
            strats = list(self.strategy_probs.keys())
            probs = np.array(list(self.strategy_probs.values()))
            probs /= probs.sum()
            sampled = np.random.choice(strats, p=probs)
            strat_id = self.tokenizer.convert_tokens_to_ids(sampled)
            prefix = torch.tensor([[strat_id]], device=self.device)
            input_ids = torch.cat([input_ids, prefix], dim=1)

        elif method == "joint":
            proc = LogitsProcessorList([StrategyConstraintLogitsProcessor(self.tokenizer, self.strategy_ids)])
            gen_kwargs["logits_processor"] = proc

        with torch.no_grad():
            outputs = self.model.generate(input_ids, **gen_kwargs)
        
        input_len = input_ids.shape[1]
        generated_tokens = outputs[0][input_len:]
        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        pred_strategy = None
        if method == "joint":
            # Joint 模式下，input_ids 长度位置的那个词就是生成的第一个词(策略)
            # 注意：generate 包含 input，所以我们要找的是原始 input 长度处的 token
            # 原始 input_ids 是注入前的长度。但这里 input_ids 已经被注入逻辑修改了（比如Oracle/Random）
            # 对于 Joint，input_ids 没变。
            first_gen_idx = inputs.input_ids.shape[1]
            if len(outputs[0]) > first_gen_idx:
                first_token_id = outputs[0][first_gen_idx].item()
                pred_strategy = self.tokenizer.decode([first_token_id])
            else:
                pred_strategy = "[Others]" # 生成失败兜底

        return response_text, pred_strategy

    def calculate_ppl(self, context, target_text):
        """
        计算 PPL (困惑度) - 倒推法 (Right-to-Left Masking)
        防止因 Token 合并导致 Context 长度计算偏大，进而 Mask 掉所有 Label 导致 NaN
        """
        if not target_text:
            return 0.0

        # 1. 完整拼接并编码
        full_text = context + target_text
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        
        # 2. 单独编码 Target 以获取其长度
        # add_special_tokens=False 非常重要，防止统计到 BOS/EOS
        target_ids = self.tokenizer(target_text, return_tensors="pt", add_special_tokens=False).input_ids
        target_len = target_ids.shape[1]
        
        # 3. 确定 Context 的长度 (Mask 边界)
        # 逻辑：总长度 - 回复长度 = 上下文长度
        # 这样能保证至少保留 target_len 个 token 用于计算 Loss
        total_len = input_ids.shape[1]
        context_len = total_len - target_len
        
        # 【兜底保护】
        # 如果 Tokenizer 合并导致 target_len 比实际在 full_ids 里占用的还长 (极罕见)，
        # 或者 context_len < 0，强制至少保留最后一个 token
        if context_len < 0:
            context_len = 0
        if context_len >= total_len:
            context_len = total_len - 1 # 至少留一个给模型猜

        # 4. 构造 Labels
        labels = input_ids.clone()
        labels[:, :context_len] = -100 # Mask 掉 Context
        
        # 5. 计算 Loss
        with torch.no_grad():
            outputs = self.model(input_ids, labels=labels)
            loss = outputs.loss
            
        # 6. 数值处理
        if torch.isnan(loss):
            return float('nan')
            
        # 转 FP32 防止 exp 溢出
        ppl = torch.exp(loss.float()).item()
        
        if math.isinf(ppl):
            return 1e9 
            
        return ppl

    def calculate_extrema(self, reference, hypothesis):
        """
        计算 Embedding Extrema Score (余弦相似度)
        使用模型自身的 Token Embeddings
        """
        if not hypothesis.strip() or not reference.strip():
            return 0.0
            
        # 获取 Token IDs
        ref_ids = self.tokenizer(reference, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        hyp_ids = self.tokenizer(hypothesis, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        
        with torch.no_grad():
            # 查表得到向量 [1, seq_len, hidden_dim]
            ref_emb = self.embedding_layer(ref_ids).squeeze(0) 
            hyp_emb = self.embedding_layer(hyp_ids).squeeze(0)
            
        # Extrema Logic: 取每一维度的最大值 (Max Pooling)
        # 结果形状: [hidden_dim]
        ref_extrema, _ = torch.max(ref_emb, dim=0)
        hyp_extrema, _ = torch.max(hyp_emb, dim=0)
        
        # 计算余弦相似度
        score = cosine_similarity(ref_extrema.unsqueeze(0), hyp_extrema.unsqueeze(0))
        return score.item()

    def calculate_metrics(self, references, hypothesis):
        """ 计算 B-2, R-L, Extrema """
        # BLEU-2
        ref_tokens = [ref.split() for ref in references]
        hyp_tokens = hypothesis.split()
        try:
            bleu2 = sentence_bleu(ref_tokens, hyp_tokens, weights=(0.5, 0.5), smoothing_function=SmoothingFunction().method1)
        except: bleu2 = 0.0
        
        # ROUGE-L
        try:
            scores = self.rouge.get_scores(hypothesis, references[0])
            rouge_l = scores[0]['rouge-l']['f']
        except: rouge_l = 0.0
        
        # Extrema
        extrema = self.calculate_extrema(references[0], hypothesis)
        
        return {"bleu-2": bleu2 * 100, "rouge-l": rouge_l * 100, "extrema": extrema * 100}