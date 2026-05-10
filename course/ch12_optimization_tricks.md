# 第 12 章：优化三板斧——不动架构榨出 3.5× 加速

## 本章目标

1. 理解 AMP FP16 的原理和 2.7× 加速来源
2. 会用 fused label smoothing 消除 7 个多余 kernel launch
3. 理解分桶如何让有效 token 翻倍

---

## 12.1 优化前后的效果

```
优化项                          tok/s      增速      累计加速
─────────────────────────────────────────────────────────────
起点（FP32 基线）              41,060       —         ×1.0
+ AMP FP16                    110,515     +169%      ×2.69
+ Fused label smoothing        98,000 → 实际效果已含在上面
+ Pre-tokenize + mmap         113,281      +14%      ×2.87
+ non_blocking 传输                   +3-5%
+ Padding bucket                    有用 token 翻倍（有效 ×1.8）
─────────────────────────────────────────────────────────────
最终                            ~90K 有效  ~3.5×     4 天→1.5 天
```

---

## 12.2 第一板斧：AMP FP16（+169%）

Turing GPU 有 **Tensor Core**——专门做 FP16 矩阵乘法的硬件。FP16 矩阵乘是 FP32 的 2 倍快：

```python
with torch.amp.autocast("cuda", dtype=torch.float16):
    # 所有 Linear/Matmul 自动走 Tensor Core
    logits = model(src, dec_input)
```

搭配 GradScaler 防止 FP16 梯度下溢：

```python
scaler = torch.amp.GradScaler()
scaler.scale(loss).backward()    # backward 前放大 loss
scaler.step(optimizer)           # optimizer 前恢复
```

---

## 12.3 第二板斧：Fused Label Smoothing（+66%）

自定义 loss 每次 forward 要启动 7 个 CUDA kernel：

```python
# 自定义实现（慢）
log_probs = F.log_softmax(logits)       # kernel 1
smooth = torch.full_like(log_probs...)  # kernel 2
smooth.scatter_(...)                    # kernel 3
smooth.masked_fill_(...)                # kernel 4
loss = -(smooth * log_probs).sum()      # kernel 5,6,7
```

PyTorch 内置版本一个 kernel 完成：

```python
loss = F.cross_entropy(logits, labels,
                       label_smoothing=0.1,
                       ignore_index=0)
```

从 56K → 99K tok/s，一条改动。

---

## 12.4 第三板斧：Padding Bucket（有效 token 翻倍）

别让 5 个 token 的短句和 72 个 token 的长句进同一个 batch：

```python
# dataset_tokenized.py
# 按源语言长度排序 buffer
sorted_order = np.argsort(chunk_lens)
chunk = chunk[sorted_order]
```

效果：batch 内最大长度差从 60+ 降到 ~5，padding 浪费从 66% 降到 31%。

**注：tok/s 的显示值会降低**（因为每个 "token" 含的 padding 更少），但 epoch 完成时间确实从 8.7h 降到 4.8h。比较的是"每 epoch 时间"不是 tok/s。

---

## 12.5 练习

1. 跑 `python bench_optimize.py`，验证 AMP 的加速效果
2. 把 `label_smoothing=0.1` 改成 `label_smoothing=0`，观察速度变化（说明发生了什么）
3. 在 `dataset_tokenized.py` 里注释掉分桶排序，跑一个 epoch 看时间变化
