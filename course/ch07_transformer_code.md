# 第 7 章：手写 Transformer——逐行实现

## 本章目标

1. 熟练使用 PyTorch `nn.Transformer` 搭建翻译模型
2. 理解 embedding、位置编码、encoder-decoder 的代码实现
3. 会分析模型参数分布（哪个模块占了多少参数）

---

## 7.1 模型总览

```python
# config.py
d_model = 512         # 模型维度
nhead = 8             # 注意力头数
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 2048  # FFN 隐藏层
vocab_size = 32000
max_seq_len = 96
dropout = 0.1
```

总参数量：**93.5M**（详见 7.4 节）。

---

## 7.2 Embedding 层

```python
class TranslationTransformer(nn.Module):
    def __init__(self, ...):
        self.src_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.tgt_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.src_pos = LearnedPositionalEmbedding(max_seq_len, d_model)
        self.tgt_pos = LearnedPositionalEmbedding(max_seq_len, d_model)

    def forward(self, src_ids, tgt_ids):
        # src_ids: (B, S_src), tgt_ids: (B, S_tgt)
        src_emb = self.src_embed(src_ids) * math.sqrt(d_model)
        src_emb = self.src_pos(src_emb)
        # → (B, S_src, d_model)
```

要点：
- `padding_idx=0`：pad 位置的 embedding 始终为 0，不参与梯度更新
- `* sqrt(d_model)`：缩放技巧，防止 embedding 太小导致梯度消失
- 中英各有一个 embedding 表——每个是 `(32000, 512)` = **16.4M 参数**

---

## 7.3 位置编码

Transformer 本身没有序列顺序概念，需要显式告诉模型"这是第几个词"。

我们使用**可学习位置编码**（LearnedPositionalEmbedding）：

```python
class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device)
        return x + self.embedding(positions)
```

每个位置 0..95 都有一个唯一向量，直接加到 token embedding 上。= **`max_len * d_model` = 96 × 512 = 49K 参数**（几乎可忽略）。

---

## 7.4 参数分析

```
模块                    参数量        占比    说明
─────────────────────────────────────────────────────
src_embed (32K×512)     16,384,000    17.5%   中文嵌入表
tgt_embed (32K×512)     16,384,000    17.5%   英文嵌入表
encoder self-attn ×6     6,297,600     6.7%   QKV 投影
encoder FFN ×6          12,598,272    13.5%   d→4d→d
decoder self-attn ×6     6,297,600     6.7%
decoder cross-attn ×6    6,297,600     6.7%
decoder FFN ×6          12,598,272    13.5%
output_proj (512×32K)   16,384,000    17.5%   预测词概率
位置编码 ×2                 262,144     0.3%
LayerNorms                   32,768    ~0%
─────────────────────────────────────────────────────
总计                     93,586,688   100%
```

**关键发现**：Embedding（35%）+ Output（18%）= **超过一半的参数在词表相关层**。真正做翻译"推理"的 transformer 层只占 47%。

---

## 7.5 Encoder 内部

每层 encoder 做的事情：

```python
# self-attention
x = self_attn(x, x, x, key_padding_mask=src_pad_mask)
x = x + residual          # 残差连接
x = norm1(x)

# FFN
x = ffn(x)
x = x + residual
x = norm2(x)
```

我们使用 `norm_first=True`（**Pre-Norm**）：LayerNorm 在 attention/FFN 之前，训练更稳定。

---

## 7.6 Decoder 内部

比 encoder 多一个 cross-attention：

```python
# Masked self-attention
x = self_attn(x, x, x, tgt_mask=causal_mask, key_padding_mask=tgt_pad_mask)
x = norm1(x + residual)

# Cross-attention: Q from decoder, K/V from encoder
x = cross_attn(x, memory, memory, key_padding_mask=src_pad_mask)
x = norm2(x + residual)

# FFN
x = norm3(ffn(x) + x)
```

Decoder 比 Encoder 多 **3 个 norm**（norm1/2/3 vs norm1/2）和多一个 cross-attention。

---

## 7.7 练习

1. 打开 `model.py`，找到 `TranslationTransformer.forward()`，画一张数据流图
2. 把 `d_model=512` 改成 `d_model=256`，手算新的参数量
3. 运行 `python analyze_model.py`，验证你的计算结果
