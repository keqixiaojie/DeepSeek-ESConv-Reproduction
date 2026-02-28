import os
import argparse
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType
)
from config import Config
from utils import DialogDataset

def train(mode, use_turn=False):
    """
    主训练函数
    mode: 'vanilla' 或 'strategy'
    use_turn: True 表示使用带轮次信息的数据 (Phase 3.5)
    """
    # 0. 动态定义后缀和打印信息
    suffix = "_turn" if use_turn else ""
    print(f"========== 开始训练: {mode} 模式 (Turn-Aware: {use_turn}) ==========")
    
    # 1. 确定路径 (动态拼接 suffix)
    if mode == 'vanilla':
        # 例如: train_vanilla.json 或 train_vanilla_turn.json
        train_file = f"train_vanilla{suffix}.json"
        val_file = f"dev_vanilla{suffix}.json"
        
        # 基础路径 + 后缀 -> models/output_vanilla_turn
        output_dir = f"{Config.OUTPUT_DIR_VANILLA}{suffix}"
        modules_to_save = None 
    else:
        train_file = f"train_strategy{suffix}.json"
        val_file = f"dev_strategy{suffix}.json"
        output_dir = f"{Config.OUTPUT_DIR_STRATEGY}{suffix}"
        
        modules_to_save = ["embed_tokens", "lm_head"]

    print(f"📂 读取数据: {train_file}")
    print(f"💾 输出目录: {output_dir}")
        
    train_path = os.path.join(Config.PROCESSED_DATA_DIR, train_file)
    val_path = os.path.join(Config.PROCESSED_DATA_DIR, val_file)
    
    # 2. 加载分词器 (Tokenizer)
    print(f"正在加载 Tokenizer: {Config.MODEL_NAME_OR_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(
        Config.MODEL_NAME_OR_PATH, 
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 3. 加载模型 (Base Model)
    print("正在加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME_OR_PATH,
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa"
    )
    
    # Strategy 模式添加 Token (Turn 模式不需要加新 Token，因为 [Turn: 1] 是纯文本)
    if mode == 'strategy':
        print(f"Strategy 模式: 正在添加 {len(Config.STRATEGY_TOKENS)} 个策略 Token...")
        tokenizer.add_special_tokens({'additional_special_tokens': Config.STRATEGY_TOKENS})
        model.resize_token_embeddings(len(tokenizer))

    # 4. 配置 LoRA
    print("正在配置 LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=Config.LORA_R, 
        lora_alpha=Config.LORA_ALPHA, 
        lora_dropout=Config.LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=modules_to_save
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 5. 准备数据集
    print("正在处理数据集...")
    train_dataset = DialogDataset(train_path, tokenizer, Config.MAX_LENGTH)
    val_dataset = DialogDataset(val_path, tokenizer, Config.MAX_LENGTH)
    
    # 6. 配置训练参数
    training_args = TrainingArguments(
        output_dir=output_dir, # 这里的 output_dir 已经带了 _turn 后缀
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        learning_rate=Config.LEARNING_RATE,
        warmup_ratio=Config.WARMUP_RATIO,
        num_train_epochs=Config.NUM_EPOCHS,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=100,             
        save_strategy="steps",
        save_steps=100,             
        save_total_limit=2,         
        # 日志目录也会自动分开: output_dir/logs
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=4,
        load_best_model_at_end=True, 
        metric_for_best_model="loss",
        greater_is_better=False,
        fp16=False,                 
        bf16=True,                  
        report_to="tensorboard",
        
        gradient_accumulation_steps=16, 
        # gradient_checkpointing=True,
        
        group_by_length=True,           
        dataloader_num_workers=4,       
        dataloader_pin_memory=True,     
        optim="paged_adamw_8bit",       
    )
    
    # 7. 开始训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        # 建议设为 True，防止 batch_size > 1 时报错
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True) 
    )
    
    # 简化的标签检查
    print("正在抽样检查验证集标签...")
    if len(val_dataset) > 0:
        item = val_dataset[0]
        labels = item['labels']
        valid_count = sum(1 for x in labels if x != -100)
        if valid_count == 0:
            print("❌ 错误：验证集第一条数据全是 -100，请检查 MAX_LENGTH！")
            exit()
        else:
            print("✅ 验证集数据格式检查通过。")

    print("开始训练循环...")
    # resume_from_checkpoint=True
    trainer.train()
    
    # 8. 保存
    print(f"训练结束，正在保存模型到 {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=['vanilla', 'strategy'], help="训练模式")
    # 【新增】开关参数，不加就是 False，加了就是 True
    parser.add_argument("--use_turn", action="store_true", help="是否使用带轮次信息的数据")
    
    args = parser.parse_args()
    
    train(args.mode, args.use_turn)