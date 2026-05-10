###
第16章 学习笔记  

### 16.1 环境与硬件基础

**GPU 关键参数（RTX 2080 Ti）**：

| 概念 | 通俗解释 | 数值 |
|------|---------|------|
| VRAM | GPU 的"内存"，存模型和中间结果 | 22 GB |
| SM | GPU 的计算核心 | 68 个 |
| Tensor Core | 专门加速矩阵乘法的硬件（FP16 比 FP32 快 2x） | 每 SM 8 个 |
| 显存带宽 | 每秒能搬运多少数据 | 616 GB/s |
| Compute Capability | GPU 架构版本号 | 7.5（Turing） |

**环境配置**：
```bash
conda create -n insta360 python=3.9 -y
conda activate insta360
pip install torch torchvision torchaudio
```

**训练时监控 GPU**：
```bash
watch -n 2 'nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader'
```
正常训练：利用率 85-100%，温度 60-80°C，显存 18-22GB。

---

### 16.2 数据流水线全貌

```
CSV(2475万行) → extract_data.py → train.zh/en(1884万) → pre_tokenize.py → .npy mmap → DataLoader
                    ↑                    ↑                        ↑
               csv.reader解析      5道过滤清洗            SentencePiece BPE
```

**关键教训**：永远不要用 `str.split(',')` 解析 CSV。中文含半角逗号会导致字段错位（6.9%中文句损坏）。

**五道数据清洗**：

| 过滤项 | 淘汰占比 | 方法 |
|--------|---------|------|
| 精确重复句对 | 16.61% | `set()` 去重 |
| 过短/过长 | 2.84% | `len < 4` 或 `> 400` |
| 长度比异常 | 1.72% | `ratio < 0.25` 或 `> 4.0` |
| 语种错配 | 1.17% | CJK/Latin 字符占比检测 |
| 合计 | 22.33% | 保留 ~1922 万句对 |

---

### 16.3 BPE 分词

**BPE（Byte Pair Encoding）核心思想**：从字符开始，反复合并最频繁的相邻字符对。

**为什么不用结巴分词**：

| 维度 | BPE | 结巴 |
|------|-----|------|
| 切分粒度 | 子词（可拆可合） | 词（词典为界） |
| 语言覆盖 | 中英共享词表 | 主要中文 |
| OOV 处理 | 天然无 OOV | 词典外可能 OOV |
| 词表大小 | 固定 32K | 可达几十万 |

**实际分词效果**：
- `自然` `语言` `处理` → 高频词组保留为完整 token
- `饕餮` → 罕见字拆成独立 token
- `Transformer` → 拆成 `Trans` `form` `er`

**SentencePiece 训练参数**：
```python
spm.SentencePieceTrainer.train(
    vocab_size=32000,
    model_type="bpe",
    character_coverage=0.9995,
    input_sentence_size=10_000_000,  # 从3768万行中采样1000万句
)
```

---

### 16.4 预分词与 DataLoader

**预分词流程**：
```python
ids = sp.encode(text, out_type=int)           # ① 分词
ids = [BOS_ID] + ids[:max_len-2] + [EOS_ID]   # ② 加头尾，截断
# ③ 追加写入 .npy
```

**存储格式**（mmap 零拷贝读取）：
```
data_tokenized/
├── train_src_ids.npy      # 所有 token id 平铺
├── train_src_offsets.npy  # 每句的起始偏移量
├── train_src_lengths.npy  # 每句的长度
```

取第 `i` 句：`all_ids[offsets[i] : offsets[i]+lengths[i]]` → O(1) 直接切片。

**mmap 优势**：操作系统按需映射到虚拟内存，零拷贝、零解析开销。

---

### 16.5 Transformer 架构关键点

**手算注意力维度（B=2, S=5, d=512, h=8）**：

```
X:        (2, 5, 512)
Q=K=V:    (2, 5, 512)    # 经过 W_Q/W_K/W_V 投影
Q:        (2, 8, 5, 64)  # 分成 8 个头
K^T:      (2, 8, 64, 5)
scores:   (2, 8, 5, 5)   # QK^T
attn:     (2, 8, 5, 5)   # softmax(scores/√64)
output:   (2, 8, 5, 64)  # attn × V → reshape → (2, 5, 512)
```

**核心公式**：`Attention(Q,K,V) = softmax(QK^T / √d_k) × V`

**Mask 类型**：
- **Padding Mask**：忽略 `<pad>` 位置（`score=-inf` → `softmax→0`）
- **Causal Mask**（Decoder）：第 i 个 token 只能看到前 i-1 个

**Encoder vs Decoder**：

| | Encoder | Decoder |
|---|---|---|
| Self-Attention | 双向（看全部） | 因果（只看前面） |
| Cross-Attention | 无 | Q来自decoder, K/V来自encoder |
| 输出 | 上下文表示 | 下一个 token 的概率 |

---

### 16.6 模型参数分布（93.5M）

```
模块                    参数量        占比
─────────────────────────────────────────
src_embed (32K×512)     16,384,000    17.5%
tgt_embed (32K×512)     16,384,000    17.5%
encoder self-attn ×6     6,297,600     6.7%
encoder FFN ×6          12,598,272    13.5%
decoder self-attn ×6     6,297,600     6.7%
decoder cross-attn ×6    6,297,600     6.7%
decoder FFN ×6          12,598,272    13.5%
output_proj (512×32K)   16,384,000    17.5%
位置编码 ×2                 262,144     0.3%
LayerNorms                   32,768    ~0%
─────────────────────────────────────────
总计                     93,586,688   100%
```

**关键发现**：Embedding（35%）+ Output（18%）= **超过一半参数在词表相关层**。

---

### 16.7 训练循环核心机制

**一个 step 的完整流程**：
```python
# 1. 数据准备
dec_input = tgt[:, :-1]     # "<bos> The weather is nice today"
dec_label = tgt[:, 1:]      # "The weather is nice today <eos>"

# 2. Forward（AMP FP16）
with torch.amp.autocast("cuda", dtype=torch.float16):
    logits = model(src, dec_input)
    loss = F.cross_entropy(logits.reshape(-1, V),
                           dec_label.reshape(-1),
                           ignore_index=0, label_smoothing=0.1)

# 3. Backward + Optimizer（梯度累积每2步更新）
scaler.scale(loss).backward()
if step % 2 == 0:
    scaler.unscale_(optimizer)
    clip_grad_norm_(max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    scheduler.step()  # Noam 调度器
```

**AMP 混合精度**：大部分计算用 FP16（Tensor Core 加速），敏感操作（softmax、loss、norm）自动保持 FP32。

**Noam 学习率调度器**：
```
lr = d_model^(-0.5) × min(step^(-0.5), step × warmup^(-1.5))
```
前 4000 步线性增长（warmup），之后按 `1/√step` 衰减。

**Label Smoothing**：把正确答案概率从 1.0 降到 0.9，剩余 0.1 分给其他词，防止过度自信。

**Teacher Forcing**：训练时 decoder 输入真实的上一个词（而非自己预测的词），加速收敛。

---

### 16.8 性能优化三板斧

| 优化 | 效果 | 原理 |
|------|------|------|
| AMP FP16 | ×2.7 | Tensor Core 硬件加速 FP16 矩阵乘 |
| Fused Label Smoothing | +66% | PyTorch 融合 CUDA kernel，7个操作→1个 |
| Pre-tokenize + mmap | +14% | 零拷贝读取，省去实时分词开销 |
| non_blocking 传输 | +3-5% | CPU→GPU 异步拷贝 |
| Padding Bucket | 有效×1.8 | 按长度排序，减少 padding 浪费 |

**累计加速：~3.5x（4天 → 1.5天）**

---

### 16.9 Roofline 模型速查

**公式**：`AI = FLOPs / Bytes_Moved`

**RTX 2080 Ti Roofline**：
- 算力峰值：107.6 TFLOPS（FP16 Tensor Core）
- 带宽峰值：616 GB/s
- Ridge Point：175 FLOP/byte
  - AI > 175 → Compute-bound（算力瓶颈）
  - AI < 175 → Memory-bound（带宽瓶颈）

**各算子 AI 速查**：

| 算子 | AI | 瓶颈 |
|------|-----|------|
| QKV 投影 | 375 | Compute |
| FFN | 400 | Compute |
| Attention QK^T | 21 | Memory |
| Softmax | 0.8 | Memory |
| LayerNorm | 2 | Memory |

**结论**：整体 compute-bound，但 attention 内部的 softmax/LayerNorm 是内存洞。

---

### 16.10 Profiler 热点速查

**RTX 2080 Ti, B=128, AMP FP16**：

| 热点 | 占比 | 类型 |
|------|------|------|
| GEMM 核（forward matmul） | ~47% | Tensor Core 算力 |
| aten::copy_ | ~17% | 显存带宽 |
| elementwise（ReLU/Dropout/Add） | ~14% | 显存带宽 |
| fused_adamw | ~8% | 优化器 |
| fmha_cutlass（attention） | ~7% | 混合 |

**从 Profiler 到行动**：
- `aten::copy_` 占比高 → 检查不必要的 CPU↔GPU 拷贝
- `elementwise` 占比高 → 尝试融合 kernel（torch.compile）
- GEMM 用了 `turing_fp16_s1688gemm` → Tensor Core 已启用

---

### 16.11 torch.compile 踩坑总结

| Backend | 结果 | 原因 |
|---------|------|------|
| inductor（默认） | CRASH | PyTorch 2.8 sympy bug（动态shape） |
| eager | 无效果 | Dynamo 开销抵消优化收益 |
| cudagraphs | CRASH | GradScaler 与 CUDA Graph 冲突 |

**定位框架 bug 的标准流程**：复现 → 缩小范围 → 读堆栈 → 搜 GitHub → 确认修复 → workaround。

**当前状态**：`compile_mode=""`（禁用），等待 PyTorch nightly 修复。

---

### 16.12 断点续训与监控

**Checkpoint 内容**：
```python
ckpt = {
    "model":       model.state_dict(),       # 93.5M 参数
    "optimizer":   optimizer.state_dict(),   # AdamW momentum+variance
    "scheduler":   scheduler.state_dict(),   # 当前步数
    "step":        step,
    "best_loss":   best_val_loss,
}
```
每个约 1.1 GB，自动保留最近 5 个 + best.pt。

**训练日志解读**：
```
ep 1/20 | 3.4% | step 5000 | loss 3.1652 | lr 4.37e-04 | 87,677 tok/s | ETA 306min
```

**何时停训**：val loss 连续 3-5 次不创新低 + train loss 还在降 = 过拟合开始。

---

### 16.13 推理与评估

**Greedy Decode**：每步选概率最高的词 → `<bos>` → "The" → "weather" → "is" → ... → `<eos>`。

**Beam Search**：同时保留 k 条候选路径，更大概率找到全局最优，但速度是 greedy 的 k 倍。

**翻译效果分析**（best.pt, val loss 2.70）：
- 短句和常见表达好（"it's a fine day today"）
- 复杂语义出错（"非常有趣" → "not always funny"）
- 个别幻觉（"起床跑步" → "runs his bed"）

**BLEU 评分**：衡量候选译文与参考译文的 n-gram 重叠率。

---

### 16.14 核心经验教训

1. **永远用标准库解析数据格式**（CSV → `csv.reader`，不要 `str.split`）
2. **BPE 是 NMT 的标准选择**：无 OOV、中英共享词表、词表大小可控
3. **Embedding 占一半参数**：93.5M 模型中 35% 是词嵌入
4. **AMP FP16 是性价比最高的优化**：一行代码 ×2.7 加速
5. **Roofline 模型指导优化方向**：先算 AI，判断是算力瓶颈还是带宽瓶颈
6. **Profiler 验证理论**：Roofline 告诉你理论上瓶颈在哪，Profiler 告诉你实际上时间花在哪
7. **torch.compile 不是免费的**：动态 shape + 新版 PyTorch = 可能崩溃
8. **分桶比调 batch_size 更有效**：减少 padding 浪费比增大 batch 更直接
9. **断点续训是必需品**：训练可能随时中断，checkpoint 机制是基本保障
10. **数据质量决定模型上限**：22% 的数据是噪音，清洗不可跳过

---

### 16.15 扩展方向速查

| 方向 | 方法 | 预期效果 |
|------|------|---------|
| d_model 扫参 | 跑 `bench_speed_vs_quality.py` | d=384 L=4 是甜点（1.7x 加速） |
| 多 GPU 训练 | `nn.DataParallel` 或 `torch.distributed` | 接近线性缩放 |
| 知识蒸馏 | 大模型（d=512）教小模型（d=256） | 速度↑ + 质量保留 |
| 量化部署 | `torch.quantization` FP32→INT8 | 推理 2-4x，模型减半 |
| 在线服务 | FastAPI + translate.py + 请求队列 | REST API |
