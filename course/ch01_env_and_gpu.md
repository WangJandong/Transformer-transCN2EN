# 第 1 章：环境准备与 GPU 基础

## 本章目标

1. 用 conda 创建独立 Python 环境并安装 PyTorch
2. 用 `nvidia-smi` 看懂 GPU 关键参数
3. 理解 SM、Tensor Core、显存带宽、Compute Capability 的含义

---

## 1.1 检查你的 GPU

打开终端输入：

```bash
nvidia-smi
```

如果看到 GPU 信息表，说明驱动正常。输出中重点关注：

```
GPU-Util: 利用率百分比
Memory-Usage: 已用显存 / 总显存
Temp: 温度
Pwr:Usage/Cap: 功耗/上限
```

我们的项目全程使用 RTX 2080 Ti，改造后 22GB 显存。

---

## 1.2 GPU 硬件概念卡片

| 概念 | 通俗解释 | 2080 Ti 数值 |
|------|---------|-------------|
| VRAM | GPU 的"内存"，存模型和中间结果 | 22 GB |
| SM | GPU 的计算核心，类似 CPU 的核 | 68 个 |
| Tensor Core | 专门加速矩阵乘法的硬件（FP16 比 FP32 快 2×） | 每 SM 8 个 |
| 显存带宽 | 每秒能搬运多少数据 | 616 GB/s（理论） |
| Compute Capability | GPU 架构版本号 | 7.5（Turing） |

**为什么要知道这些？** 显存决定模型+batch 能多大，带宽决定数据搬运速度，Tensor Core 决定矩阵乘法多快。第 10 章会用它分析瓶颈。

---

## 1.3 安装 conda 和 PyTorch

```bash
# 创建环境
conda create -n insta360 python=3.9 -y
conda activate insta360

# 安装 PyTorch（CUDA 12.6）
pip install torch torchvision torchaudio
```

验证：

```python
import torch
print(torch.cuda.is_available())   # True
print(torch.cuda.get_device_name(0))  # RTX 2080 Ti
```

---

## 1.4 训练时怎么监控 GPU

```bash
# 实时刷新
watch -n 2 'nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader'
```

正常训练时的参考值：利用率 85-100%，温度 60-80°C，显存 12-22GB。

---

## 1.5 练习

1. 跑通 `nvidia-smi`，记录 GPU 名、显存、CUDA 版本
2. 创建 conda 环境，安装 PyTorch，验证 CUDA 可用
3. 训练后面章节时，常开 `watch nvidia-smi` 观察 GPU 指标变化
