# 第16章 学习笔记

## 16.1 嵌入层 vs 位置编码

- 嵌入层回答"这个词是什么"（token id → 语义向量），位置编码回答"这个词在第几位"（位置索引 → 位置向量）
- 两者相加得到带位置信息的语义向量
- 本项目4个 nn.Embedding：src_embed(32000,512)、tgt_embed(32000,512)、src_pos.embedding(96,512)、tgt_pos.embedding(96,512)，占93.5M参数的35%

---

## 16.2 词嵌入共享 vs 不共享

- 原始Transformer（英德）共享词嵌入，因同语系词根同源
- 本项目（中英）不共享更合理：中文孤立语vs英文屈折语，语法差异极大
- 规则：中英差异大→不共享更灵活；参数紧张→共享差距也不大

---

## 16.3 学习的位置嵌入 vs 正弦 vs RoPE

| 方式 | 训练长度内效果 | 外推 | 参数量 | 本项目适用 |
|------|:---:|:---:|------|:---:|
| 学习的 | 最好 | 不支持 | max_len×d_model | ✅ 现状最优 |
| 正弦 | 足够好 | 支持 | 0 | 换了也不亏 |
| RoPE | 好 | 支持 | 0 | ❌ 需手写attention丢融合kernel |

- RoPE是LLaMA/Qwen/DeepSeek等现代大模型主流方案
- 本项目结论：学习的位置嵌入最优，固定max_seq_len=96不需要外推；正弦换了效果持平不必要；RoPE反而亏（丢CUTLASS优化+模型太小不体现优势）

---

## 16.4 BPE vs 结巴分词

- 粒度：BPE子词级别 vs 结巴词级别
- 语言：BPE中英共享词表 vs 结巴只切中文
- OOV：BPE天然无OOV vs 结巴词典外词可能切错
- 翻译场景选BPE的原因：中英不对等、OOV灾难、embedding维度爆炸、共享词表天然对齐

---

## 16.5 embed_scale = sqrt(d_model)

模型中的代码：`src_emb = self.src_pos(self.src_embed(src_ids) * self.embed_scale)`。

- 原始 Transformer 论文的做法，`embed_scale = math.sqrt(d_model)` ≈ 22.6（d_model=512）
- 目的：把 embedding 的方差从 ~1 放大到 ~d_model，让词嵌入和位置编码在数值范围上匹配
- 不乘这个的话，位置编码的贡献会远大于词嵌入，语义信息被淹没

---

## 16.6 output_proj — 输出投影层

- `nn.Linear(512, 32000)`，参数量 **16.4M**
- 把 decoder 最后一层的 512 维向量投射回 32000 维词表空间 → 每个词的得分 → softmax 选下一个 token
- 和 `tgt_embed` 是互逆操作：
  - `tgt_embed`: token id → 512 维向量（输入端）
  - `output_proj`: 512 维向量 → 词表得分（输出端）
- 两者权重矩阵形状恰好转置：`tgt_embed.weight(32000, 512)` 和 `output_proj.weight(32000, 512)`

---

## 16.7 Weight Tying（权重共享）

**方案1（推荐）**：共享 `tgt_embed + output_proj`
- 参数量 93.6M → **77.2M**（省 17.5%）
- 原始 Transformer 论文标准做法
- 更不易过拟合，训练更稳定

**方案2**：共享全部三个（`src_embed + tgt_embed + output_proj`）
- 参数量 93.6M → **60.8M**（省 35%）
- 中英差异大，不推荐共享 `src_embed`

**对训练计算量的影响**：基本不变
- 前向/反向 FLOPs 不变（矩阵乘该算还得算）
- 只省 65MB 显存（16.4M × 4 bytes）

**结论**：weight tying 收益在效果（更稳、不过拟合），不在速度。本项目当前未共享。

---

## 16.8 提参方案：把省下的参数加回 body

基线（weight tying 后）：**77.0M**

| 方案 | 配置 | 参数量 | 速度 | 显存 |
|------|------|--------|------|------|
| **A8（推荐）** | d512 **L8** ff2048 | 91.7M | 0.8× | ~20G |
| C | d512 L6 **ff3072** | 89.6M | 1.0× | ~19.5G |
| D3 | d512 **L7 ff2560** | 91.7M | 0.9× | ~19.8G |

**推荐 A8**（L6→L8，加 4 层共 16 层）：
- 参数量回到原模型水平（~93M），参数不浪费
- 深度 > 宽度（Transformer 论文已验证）
- val loss 预计 -0.1~0.2
- 每 epoch 3h→3.75h，显存仍在 22G 安全区内
- 1884 万数据对 91M 模型充足，不过拟合

---

## 16.9 优化器选择

**当前**：AdamW (fused CUDA) + Noam 调度器，`betas=(0.9, 0.98)`, `eps=1e-9`。

| 选择 | 推荐度 | 理由 |
|------|:---:|------|
| AdamW（当前） | ✅ | NMT 标配，和 Noam 配合成熟 |
| Lion | ⚠️ | 只有 momentum 无二阶矩，需要重新调 LR |
| SGD+Momentum | ❌ | NMT 很少用，收敛慢 |
| Adafactor | ❌ | 为 B+ 参数模型设计的内存优化，93M 不需要 |

**结论**：AdamW 不用换。

### 值得做的实验：Noam → Cosine 调度器

- Noam（2017 原论文）：`lr ∝ min(step^(-0.5), step × warmup^(-1.5))`
- Cosine：从 max 余弦衰减到 0，配合 warmup
- 预期：收敛略快，最终效果通常更好

---

## 16.10 激活函数选择

**当前**：ReLU (`max(0, x)`) 

| 选择 | 推荐度 | 理由 |
|------|:---:|------|
| ReLU（当前） | ✅ | 简单快速，但负区间梯度为 0 |
| **GELU** | ✅✅ | BERT/GPT 标配，`nn.Transformer` 一行改，预计 +0.3~0.8 BLEU |
| SiLU/Swish | ⚠️ | 和 GELU 相当，新 PyTorch 支持 |
| SwiGLU | ❌ | 效果好但需改 FFN 结构，不能用 `nn.Transformer` 一行搞定 |

**推荐**：`config.py` 改 `activation = "gelu"`，一行代码，零成本，确定性的提升。
