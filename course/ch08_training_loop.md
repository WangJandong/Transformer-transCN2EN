# 第 8 章：训练循环——AMP、调度器、损失函数

## 本章目标

1. 理解混合精度训练（AMP）的 forward-backward 流程
2. 理解 Noam 学习率调度器的 warmup 机制
3. 知道 label smoothing 做了什么、为什么快

---

## 8.1 一个训练 step 的完整流程

```python
# 1. 加载数据
src, tgt = batch
dec_input = tgt[:, :-1]     # "<bos> The weather is nice today"
dec_label = tgt[:, 1:]      # "The weather is nice today <eos>"

# 2. Forward（FP16 自动混合精度）
with torch.amp.autocast("cuda", dtype=torch.float16):
    logits = model(src, dec_input)         # (B, S, 32000)
    loss = F.cross_entropy(logits.reshape(-1, 32000),
                           dec_label.reshape(-1),
                           ignore_index=0, label_smoothing=0.1)

# 3. Backward（梯度缩放防下溢）
scaler.scale(loss).backward()

# 4. Optimizer step（每 grad_accum_steps=2 步更新一次）
if step % 2 == 0:
    scaler.unscale_(optimizer)            # 恢复梯度
    clip_grad_norm_(max_norm=1.0)         # 梯度裁剪
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    scheduler.step()                      # 更新学习率
```

---

## 8.2 为什么用 AMP（混合精度）

FP32 精度高但慢，FP16 借助 Tensor Core 快 2.7× 但数值范围小（`6×10⁻⁸ ~ 65504`）。

AMP 策略：**大部分计算用 FP16，敏感操作（loss、softmax、norm）自动保持 FP32**。

```python
with torch.amp.autocast("cuda", dtype=torch.float16):
    # 这里面的矩阵乘法自动用 FP16 Tensor Core
    logits = model(src, dec_input)
    # softmax、cross_entropy 自动用 FP32
```

实测效果：**41K tok/s → 110K tok/s（2.7×）**。

---

## 8.3 GradScaler：防止梯度下溢

FP16 的最小正数是 `6×10⁻⁸`，比它小的梯度变成 0。GradScaler 在 backward 前把 loss 放大（默认 2^16），梯度也跟着放大，减少下溢。optimizer step 前再 `unscale_` 恢复。

---

## 8.4 Noam 学习率调度器

来自原始 Transformer 论文，先 warmup 再衰减：

```
lr = d_model^(-0.5) × min(step^(-0.5), step × warmup^(-1.5))
```

图示：
```
lr
↑        ╱╲
│       ╱    ╲
│      ╱        ╲___________
│     ╱
│    ╱
│   ╱
└──┴────────────────────→ step
   warmup=4000
```

前 4000 步线性增长（避免初期不稳定），之后按 `1/√step` 衰减。

---

## 8.5 Label Smoothing：为什么快 66%

普通交叉熵想让模型对正确答案输出 100% 概率。但这会让模型"过于自信"。

Label Smoothing 把正确答案的概率从 1.0 降到 0.9，把剩余的 0.1 分给所有其他词：

```
无 smoothing: [0, 0, 1.0, 0, 0]     ← one-hot
有 smoothing: [ε, ε, 0.9, ε, ε]     ← smoothed, ε = 0.1/(V-1)
```

PyTorch 内置的 `F.cross_entropy(label_smoothing=0.1)` 是一个**融合 CUDA kernel**——把 log_softmax + scatter + mask + sum 七个操作合并成一个。实测快了 **66%**。

---

## 8.6 练习

1. 用 `--no_amp` 跑 1000 步，对比 tok/s
2. 把 warmup_steps 从 4000 改成 1000，观察 loss 曲线的变化
3. 把 label_smoothing 从 0.1 改成 0，观察 val loss 的变化
