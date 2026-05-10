# 第 3 章：Transformer 原理——手算注意力

## 本章目标

1. 能手算 (B, S, d) → QKV → Attention → Output 每一步的矩阵维度变化
2. 理解 Multi-Head 为什么把 d_model 分成 h 个头
3. 理解 encoder-decoder 结构中 mask 的作用

---

## 3.1 整体结构

Transformer 由两部分组成：

```
输入 → [Encoder × 6层] → 中间表示 → [Decoder × 6层] → 输出投影 → 预测词
```

每层内部：

```
Encoder 层:  Self-Attention → Add+Norm → FFN → Add+Norm
Decoder 层:  Masked Self-Attn → Cross-Attn → Add+Norm → FFN → Add+Norm
```

---

## 3.2 手算：输入到 QKV

假设一个 batch：

```
B = 2        # batch size
S = 5        # 序列长度
d = 512      # 模型维度 (d_model)
h = 8        # 注意力头数
d_k = d/h = 64  # 每个头的维度
```

**输入**：一个 (2, 5, 512) 的张量——2 句话，每句 5 个 token，每个 token 用 512 维向量表示。

**QKV 投影**：

```
输入 X:      (2, 5, 512)
权重 W_Q:    (512, 512)
权重 W_K:    (512, 512)
权重 W_V:    (512, 512)

Q = X × W_Q: (2, 5, 512)
K = X × W_K: (2, 5, 512)
V = X × W_V: (2, 5, 512)
```

PyTorch 实际把 QKV 合并为 `in_proj_weight`：(512, 1536)，一次矩阵乘出 Q、K、V。

---

## 3.3 手算：Scaled Dot-Product Attention

这是 Transformer 的核心公式：

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

**Step 1：计算相似度 QK^T**

```
Q:  (2, 5, 512)  → reshape → (2, 8, 5, 64)  # 分成 8 个头
K^T: (2, 8, 64, 5)
scores = QK^T: (2, 8, 5, 5)

scores[i, j] = 第 i 个 token 对 第 j 个 token 的 "关注度"
```

**Step 2：缩放 + Softmax**

```
scores = scores / √64 = scores / 8  # 防止梯度消失
attn_weights = softmax(scores, dim=-1)  # (2, 8, 5, 5)
                                         # dim=-1 表示对每行做归一化
```

**Step 3：加权求和**

```
V: (2, 8, 5, 64)
output = attn_weights × V: (2, 8, 5, 64) → reshape → (2, 5, 512)
```

---

## 3.4 Mask 的作用

### Padding Mask

不等长句子 pad 到相同长度。mask 让模型忽略 pad 位置：

```python
src_pad_mask = (input_ids == 0)  # (2, 5), True where padding
```

pad 位置的 attention score 被设为 -∞，softmax 后变成 0。

### Causal Mask（Decoder Self-Attention）

解码时第 i 个 token 只能看到前面 i-1 个：

```
    t0  t1  t2  t3  t4
t0   0  -∞  -∞  -∞  -∞    ← 只能看自己
t1   0   0  -∞  -∞  -∞    ← 只能看 t0 和 t1
t2   0   0   0  -∞  -∞
t3   0   0   0   0  -∞
t4   0   0   0   0   0
```

---

## 3.5 Multi-Head 为什么有效

如果只用 1 个头 (512 维)，模型只看一种"关系"。分成 8 个头 (各 64 维)，不同头可以学不同类型的关系：

- Head 1：关注相邻词（局部搭配）
- Head 2：关注主语-谓语关系
- Head 3：关注标点符号
- ...

---

## 3.6 Encoder vs Decoder

| | Encoder | Decoder |
|---|---|---|
| Self-Attention | ✅ 双向（能看到全部） | ✅ 因果（只能看前面的） |
| Cross-Attention | ❌ 无 | ✅ Q 来自 decoder，K/V 来自 encoder |
| 输出 | 上下文表示（给 decoder 用） | 下一个 token 的概率 |

Decoder 的 Cross-Attention 是翻译的关键——Q（我要生成什么）去查 K（源语言说了什么），找到最相关的源语言信息。

---

## 3.7 FFN（前馈网络）

每层 self-attention 后接一个两层全连接：

```
(2, 5, 512) → Linear(512, 2048) → ReLU → Linear(2048, 512) → (2, 5, 512)
                         ↑                        ↑
                      升维 4×                   降维回来
```

升维-降维的设计给了模型更大的表示空间来组合 attention 的输出。

---

## 3.8 完整前向流程（带维度）

以一个翻译为例：中文 "今天天气很好" → 英文 "The weather is nice today"

```
Encoder:
  src_embed:      (1, 5, 512)    # 5 个中文 token
  + pos_embed:    (1, 5, 512)
  → Encoder×6:    (1, 5, 512)    # 每层不变形状
  → memory:       (1, 5, 512)

Decoder（教师强制，训练时）:
  tgt_embed:      (1, 6, 512)    # "The weather is nice today <eos>"
  + pos_embed:    (1, 6, 512)
  → Self-Attn:    (1, 6, 512)    # causal mask 防止偷看后面
  → Cross-Attn:   (1, 6, 512)    # Q: decoder, K/V: encoder memory
  → FFN:          (1, 6, 512)
  → Decoder×6:    (1, 6, 512)

输出:
  → output_proj:  (1, 6, 32000)  # 每个位置输出词表概率
  → loss = CrossEntropy(预测, 真实下一个token)
```

---

## 3.9 练习

1. 给定 B=4, S=10, d=512, h=8，手算 Q、K、V、attention_scores、output 各是什么形状
2. 解释 causal mask 为什么在训练时需要、推理时不需要
3. 思考：FFN 的维度从 512 升到 2048 再降回 512，为什么不一直用 512？
