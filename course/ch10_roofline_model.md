# 第 10 章：GPU 性能分析——Roofline 模型

## 本章目标

1. 会算任意算子的 Arithmetic Intensity（FLOPs / 搬运字节数）
2. 能用 Roofline 模型判断算子瓶颈在算力还是带宽
3. 理解为什么 attention 的 softmax 是"内存洞"

---

## 10.1 两种瓶颈

GPU 执行一个算子时，耗时取决于两个上限：

- **Compute-bound**：算力吃满，矩阵乘法等重计算操作
- **Memory-bandwidth-bound**：数据来不及搬运，element-wise 操作等

怎么判断？算 **Arithmetic Intensity（AI）**：

```
AI = FLOPs / Bytes_Moved
```

- AI **高** → compute-bound（算力是瓶颈）
- AI **低** → memory-bound（带宽是瓶颈）

---

## 10.2 RTX 2080 Ti 的 Roofline

```
理论峰值:
  FP16 Tensor Core: 107.6 TFLOPS  (算力)
  显存带宽:         616 GB/s     (搬运速度)

Roofline Ridge Point:
  107,600 GFLOPS / 616 GB/s = 175 FLOP/byte

  AI > 175 → compute-bound（算力不够）
  AI < 175 → memory-bound（带宽不够）
```

---

## 10.3 手算各算子的 AI

以 B=128, S=128, d=512 为例：

### QKV 投影（AI = 375 → Compute-bound）

```
FLOPs:    2 × 128 × 128 × 512 × (3×512) = 25.8 GFLOPs
Bytes:    读 X(16.8MB) + 读 W(1.5MB) + 写 QKV(50.3MB) ≈ 69 MB
AI:       25.8G / 69M = 375 FLOP/byte
```

375 > 175，**算力瓶颈**。GPU Tensor Core 能充分利用。

### Attention QK^T（AI = 21 → Memory-bound）

```
FLOPs:    2 × 8 × 128 × 128 × 64 = 2.2 GFLOPs
Bytes:    读 Q(67MB) + 读 K(67MB) + 写 scores(33MB) ≈ 167 MB
AI:       2.2G / 167M = 13 FLOP/byte
```

13 << 175，**内存瓶颈**。GPU 大部分时间在等数据，Tensor Core 闲置。

### FFN（AI = 400 → Compute-bound）

### LayerNorm（AI = 2 → 严重 Memory-bound）

### Softmax（AI = 0.8 → 严重 Memory-bound）

---

## 10.4 各算子 AI 汇总表

```
算子                 GFLOPs    MB moved    AI       瓶颈
──────────────────────────────────────────────────────
QKV projection        25.8       68.7      375     Compute ✅
Attention QK^T         2.2      100.7       21     Memory  ⚠️
Attention output@V     2.2      100.7       21     Memory  ⚠️
Attn out projection    8.6       34.1      252     Compute ✅
FFN up                34.4       86.0      400     Compute ✅
FFN down              34.4       86.0      400     Compute ✅
LayerNorm              0.03      16.8        2     Memory  ⚠️
Softmax                0.05      67.1        0.8   Memory  ⚠️
Output projection    536.9     1098.1      489     Compute ✅
──────────────────────────────────────────────────────
总 AI (加权)          388       —          —       Compute ✅
```

**结论**：整体 compute-bound，但 attention 内部的几个子操作（QK^T、softmax、LayerNorm）是内存洞，浪费 GPU 算力。

---

## 10.5 这和你的训练速度有什么关系

知道哪些算子是内存瓶颈后，优化方向就很明确：

- Compute-bound 算子：用 FP16 Tensor Core 加速（第 12 章）
- Memory-bound 算子：融合 kernel，减少数据搬运次数（第 11、12 章）
- 无法优化：接受，或升级硬件（HBM 带宽更高的 GPU）

---

## 10.6 练习

1. 手算 FFN (512→2048→512) 的 FLOPs 和 AI
2. 如果你的 GPU 显存带宽从 616 GB/s 升级到 2000 GB/s（H100），ridge point 会变成多少？attention 还会是内存瓶颈吗？
3. 打开 `analyze_model.py`，跑一遍，验证 AI 数值
