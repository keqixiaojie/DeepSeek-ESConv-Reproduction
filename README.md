# DeepSeek-ESConv-Reproduction
Unofficial reproduction &amp; modernization of ACL 2021 ESDS. Emotional support dialog generation using DeepSeek-R1, LoRA, and Constrained Decoding.

**📖 Introduction**

本项目是对 ACL 2021 顶会论文 *Towards Emotional Support Dialog Systems* 的现代化复现与底层升级。我们将原论文基于传统 Seq2Seq 的特征注入范式，重构为完全适用于现代 Decoder-Only 大模型（**DeepSeek-R1**）的特殊 Token 控制方案。
基于 ESConv 数据集，本项目不仅打通了全流程的 LoRA 微调，还实现了严格对齐学术标准的评估体系。

**✨ 核心亮点 (Key Features)**

* 🔄 **架构重构与复现**：摒弃传统拼接，采用特殊 Token 注入与动态 `[Turn: N]` 标识，解决大模型轮次意识缺失问题。
* 🧠 **受限解码 (Constrained Decoding)**：底层重写 `StrategyConstraintLogitsProcessor`，在 Joint 模式下干预 Logits 分布，强制实现“先选策略，再出回复”的精准控制。
* ⚡️ **单卡极限优化**：深度整合 Flash Attention 2、Gradient Checkpointing 与全量字符串编码防截断策略。项目对消费级旗舰显卡（如 RTX 5090）极度友好，完美攻克长对话上下文带来的 OOM 与指标失真难题。

