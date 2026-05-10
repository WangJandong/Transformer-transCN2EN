# 第 9 章：监控、日志和断点续训

## 本章目标

1. 读懂训练日志的每一列
2. 会判断训练是否正常（loss 下降趋势、GPU 指标）
3. 实现训练中断后从 checkpoint 无缝恢复

---

## 9.1 训练日志解读

```
ep 1/20 |   3.4% | step    5000 | loss 3.1652 | lr 4.37e-04 | 87,677 tok/s | ETA 306min
```

逐列解释：

| 列 | 含义 | 怎么做判断 |
|----|------|----------|
| `ep 1/20` | 第 1 个 epoch，共 20 个 | 通常不需要跑满，val loss 不再降就停 |
| `3.4%` | 当前 epoch 完成百分比 | 走到 100% 进入下一个 epoch |
| `step 5000` | 当前全局步数 | 配合 val_interval 判断何时出 val loss |
| `loss 3.1652` | 当前训练损失 | 应持续下降，偶尔波动正常 |
| `lr 4.37e-04` | 当前学习率 | Noam 调度下先升后降 |
| `87,677 tok/s` | 每秒处理的 token 数 | 稳定后反映真实训练速度 |
| `ETA 306min` | 当前 epoch 预计剩余时间 | 越往后越准 |

---

## 9.2 验证日志

每 5000 步做一次验证：

```
--- val loss 2.6989 (best) ---   ← 创了新低
--- val loss 2.7760 ---          ← 没创新低
--- val loss 2.8849 ---          ← 开始反弹
```

**什么时候该停训**：val loss 连续 3-5 次不创新低 + train loss 还在降 = 过拟合开始。

---

## 9.3 日志双写

```python
class Tee:
    """同时写到终端和文件"""
    def __init__(self, path):
        self.file = open(path, "a", buffering=1)  # 行缓冲
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
```

每条日志同时出现在终端和 `logs/train_YYYYMMDD_HHMMSS.log`。

---

## 9.4 Checkpoint 里存了什么

```python
ckpt = {
    "model":       model.state_dict(),       # 93.5M 参数的 FP32 权重
    "optimizer":   optimizer.state_dict(),   # AdamW 的 momentum + variance
    "scheduler":   scheduler.state_dict(),   # 当前 step 数（恢复 LR 曲线用）
    "step":        step,                     # 全局步数
    "best_loss":   best_val_loss,            # 历史最佳 val loss
}
```

每个 checkpoint 约 **1.1 GB**。自动保留最近 5 个 + best.pt。断点续训从编号最大的 `step_*.pt` 恢复。

---

## 9.5 断点续训实践

训练中按 `Ctrl+C` 停止，然后：

```bash
bash run.sh    # 自动检测 checkpoints/ 中最新的 step_*.pt，从那里继续
```

输出会显示：
```
Resumed from step 315000, best loss 2.6989
```

---

## 9.6 兼容性设计

使用 `torch.compile` 时模型会被包装成 `OptimizedModule`，state_dict 的 key 会多 `_orig_mod.` 前缀。我们的 `load_checkpoint` 自动处理了两种情况：

```python
# compile 版本加载无前缀的 checkpoint → 自动加前缀
# 无 compile 版本加载有前缀的 checkpoint → 自动去前缀
```

---

## 9.7 练习

1. 启动训练，观察第一行 step 500 的 ETA 和 step 1000 的 ETA，解释为什么不同
2. 手动 `Ctrl+C` 停止训练，再 `bash run.sh` 恢复，验证 step 是否接上了
3. 打开 `checkpoints/` 目录，查看文件大小和时间戳
