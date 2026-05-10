# 第 15 章：课程总结与扩展方向

## 15.1 你学会的东西

```
完整项目流程:
  CSV 解析 → 数据清洗 → 分词 → Transformer 手写
  → 训练循环 → AMP → 断点续训 → Profiling
  → 性能优化 → torch.compile 踩坑 → 推理
```

这不是调包——你自己动手写了训练代码、分析了 GPU 热点、榨了 3.5× 性能。

---

## 15.2 关键数字

| 指标 | 值 |
|------|-----|
| 训练数据 | 1884 万中英句对 |
| 模型参数 | 93.5M |
| 训练时间 | ~4 天（RTX 2080 Ti） |
| 最优 val loss | 2.70 |
| 吞吐量 | ~90K tok/s |
| 累计加速 | ~3.5× (vs 原始基线) |

---

## 15.3 五个扩展方向

### 1. d_model 扫参

运行 `python bench_speed_vs_quality.py`，已有 5 组对比。把 d=384 L=4 跑完整训练，和 d=512 比 BLEU。

### 2. 多 GPU 训练

把 `nn.DataParallel` 或 `torch.distributed` 把 batch 分到多张卡上。速度接近线性缩放（2 卡 2×，4 卡 4×）。

### 3. 知识蒸馏

用大模型（d=512）当教师，训练小模型（d=256）去模仿输出分布。小模型速度更快 + 保留大模型质量。

### 4. 量化部署

训练后用 `torch.quantization` 把 FP32 权重转 INT8，推理速度 2-4×，模型文件减半。适合部署到手机或浏览器。

### 5. 在线翻译服务

用 FastAPI 包装 translate.py，加上请求队列、缓存常用翻译，做成一个 REST API。

---

## 15.4 推荐进阶资源

- **Attention Is All You Need**（原始论文）：理解每个设计选择的原因
- **Roofline Model 论文**（Williams et al. 2009）：性能分析的经典框架
- **PyTorch Profiler 文档**：更多 profiling 技巧
- **NVIDIA NSight Systems**：比 nvidia-smi 更底层的 GPU 性能分析工具

---

## 15.5 练习

1. 选上面 5 个方向之一，尝试跑起来
2. 翻一遍 `HANDOFF.md`，确认你能独立接手这个项目
3. 把你自己的翻译数据放进去，训练一个领域定制模型
