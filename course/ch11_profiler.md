# 第 11 章：PyTorch Profiler 实战

## 本章目标

1. 会用 `torch.profiler.profile()` 采集训练性能数据
2. 能读懂 profiler 报表，识别 compute、memory、copy 三类热点
3. 会据此判断下一步优化方向

---

## 11.1 为什么需要 Profiler

Roofline 模型告诉你"理论上"瓶颈在哪，Profiler 告诉你"实际上"时间花在哪。

---

## 11.2 最小可用 Profiler

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for step in range(5):
        # 你的训练代码
        loss.backward()
        optimizer.step()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
```

---

## 11.3 我们的 Profiler 输出

RTX 2080 Ti, B=128, AMP FP16, 每 step ~54ms：

```
Kernel                         CUDA time (ms)    %
─────────────────────────────────────────────────
forward (encoder-decoder)        140.6         34.2%
aten::mm (matmul)                109.8         26.7%   ← Tensor Core GEMM
aten::copy_                       71.2         17.3%   ← 数据搬运 ⚠️
aten::addmm (backward matmul)    39.2          9.5%
turing_fp16 gemm kernels         88.8         21.6%   ← 融合了 ReLU 的 GEMM
elementwise_kernel               27.7          6.8%   ← ReLU/Dropout/Add ⚠️
fused_adamw                      33.5          8.2%   ← 优化器
aten::add_                       29.4          7.2%   ← 残差连接 ⚠️
fmha_cutlass (attention)         77.4          6.9%   ← CUTLASS 注意核
```

---

## 11.4 热点分类

| 类别 | 占比 | 操作 | 瓶颈类型 |
|------|------|------|---------|
| **Compute** | ~47% | GEMM 核、backward matmul | Tensor Core 算力 |
| **Memory** | ~17% | aten::copy_ | 显存带宽 |
| **Element-wise** | ~14% | add, relu, dropout, norm | 显存带宽 |
| **Attention** | ~7% | fmha_cutlass | 混合 |
| **Optimizer** | ~8% | fused_adamw | — |

---

## 11.5 从 Profiler 到行动

看到 `aten::copy_` 占 17% → 数据搬运太多：

→ 启用 `non_blocking=True` 传输（第 6 章）
→ 检查是否有不必要的 FP32↔FP16 转换

看到 `elementwise_kernel` 占 14% → 太多小 kernel：

→ 尝试 torch.compile 融合（第 13 章）
→ 不行的话考虑 dropout=0（省 24% 但有泛化风险）

看到 `aten::mm` GEMM 核用了 `turing_fp16_s1688gemm` → Tensor Core 已启用 ✅。

---

## 11.6 练习

1. 运行 `python profile_hotspots.py`，看懂自己的 profiler 输出
2. 把 batch_size 从 128 改成 32，重新 profile，观察各热点占比的变化
3. 找出 `aten::copy_` 在你的 profiler 输出中的占比，思考它的来源
