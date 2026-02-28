import json
import torch
from torch.utils.data import Dataset
from config import Config

class DialogDataset(Dataset):
    """
    自定义数据集类，用于加载处理好的 JSON 数据并进行分词。
    """
    def __init__(self, data_path, tokenizer, max_length=1024):
        self.data = self.load_data(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 【新增/修改】 极其重要！设置截断方向为“左侧”
        # 这样当对话太长时，会切掉最久远的历史，保留最新的回复
        if self.tokenizer.truncation_side != 'left':
            print("Warning: Tokenizer truncation side was not left. Setting to 'left'.")
            self.tokenizer.truncation_side = 'left'
    
    def load_data(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    # def __getitem__(self, idx):
    #     sample = self.data[idx]
    #     input_text = sample['input']   # 历史对话 (User: ... Assistant:)
    #     output_text = sample['output'] # 目标回复 (可能包含 [Question] ...)
        
    #     # 1. 构建完整的训练序列: Input + Output + EOS
    #     # DeepSeek/Qwen 的 tokenizer 会自动处理特殊 token，但最好显式加上 EOS
    #     full_text = input_text + output_text + self.tokenizer.eos_token
        
    #     # 2. 分词
    #     # return_tensors='pt' 会返回 pytorch tensor
    #     input_tokens = self.tokenizer(
    #         input_text, 
    #         add_special_tokens=False # 我们自己控制拼接，暂时不加特殊头
    #     )
    #     full_tokens = self.tokenizer(
    #         full_text, 
    #         max_length=self.max_length, 
    #         padding="max_length", 
    #         truncation=True, 
    #         return_tensors="pt"
    #     )
        
    #     input_ids = full_tokens["input_ids"][0]
    #     attention_mask = full_tokens["attention_mask"][0]
        
    #     # 3. 构建 Labels (用于计算 Loss)
    #     # 我们不希望模型计算 "User: Hello..." 这部分历史的 Loss，只计算回复部分的 Loss
    #     labels = input_ids.clone()
        
    #     # 计算 input_text 的长度 (token 数量)
    #     input_len = len(input_tokens["input_ids"])
        
    #     # 将 input 部分的 label 设为 -100 (PyTorch 中 -100 代表忽略计算 Loss)
    #     # 注意要防止截断导致 input_len 超过 max_length
    #     if input_len < self.max_length:
    #         labels[:input_len] = -100
            
    #     # 将 padding 部分也设为 -100
    #     labels[input_ids == self.tokenizer.pad_token_id] = -100
        
    #     return {
    #         "input_ids": input_ids,
    #         "attention_mask": attention_mask,
    #         "labels": labels
    #     }
    def __getitem__(self, idx):
        sample = self.data[idx]
        input_text = sample['input']
        output_text = sample['output']
        
        # 1. 分别把 Input 和 Output 转成 ID 列表（注意不加特殊 Token）
        input_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
        output_ids = self.tokenizer.encode(output_text, add_special_tokens=False)
        
        # 2. 手动加上 EOS Token (这很重要)
        # 注意：有些模型 EOS 和 PAD 是同一个，有些不是，这里取 eos_token_id
        output_ids += [self.tokenizer.eos_token_id]

        # 3. 处理截断 (Truncation) - 这一步非常关键！
        # 如果总长度超过 max_length，我们要优先切掉 Input 的前面（保留最近的对话）
        # 而不是切掉 Output
        total_len = len(input_ids) + len(output_ids)
        if total_len > self.max_length:
            # 需要保留的 input 长度 = 最大长度 - output 长度
            keep_input_len = self.max_length - len(output_ids)
            # 如果 output 本身就比 max_length 还长（极少见），那只能切 output
            if keep_input_len < 0:
                output_ids = output_ids[:self.max_length]
                input_ids = []
            else:
                # 切掉 input 的头部 (左侧截断)
                input_ids = input_ids[-keep_input_len:]

        # 4. 拼接 (Concatenate)
        input_ids_tensor = torch.tensor(input_ids + output_ids, dtype=torch.long)
        
        # 5. 构建 Labels
        labels = input_ids_tensor.clone()
        # 将 input 部分设为 -100
        labels[:len(input_ids)] = -100

        # 6. 处理填充 (Padding)
        # 因为我们要 batch 训练，所以必须补齐到 max_length 或者 batch 内最长
        # 这里演示补齐到 max_length 的逻辑
        padding_len = self.max_length - len(input_ids_tensor)
        
        if padding_len > 0:
            # 构造 padding tensor
            padding = torch.full((padding_len,), self.tokenizer.pad_token_id, dtype=torch.long)
            padding_labels = torch.full((padding_len,), -100, dtype=torch.long) # padding 的 label 也是 -100
            
            # 拼接 (注意：Padding 通常加在右边)
            input_ids_tensor = torch.cat([input_ids_tensor, padding])
            labels = torch.cat([labels, padding_labels])
            
            # Attention Mask: 非 padding 部分为 1，padding 部分为 0
            attention_mask = torch.cat([
                torch.ones(len(input_ids) + len(output_ids), dtype=torch.long),
                torch.zeros(padding_len, dtype=torch.long)
            ])
        else:
            attention_mask = torch.ones_like(input_ids_tensor)

        return {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask,
            "labels": labels
        }