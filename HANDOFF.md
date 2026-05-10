# 中→英 翻译模型训练 — Handoff 文档

## 环境

- **conda env**: `insta360`
- **Python**: `/home/wjd/miniforge3/envs/insta360/bin/python`（用 `python` 不是 `python3`）
- **GPU**: NVIDIA RTX 2080 Ti，改造 22GB VRAM，Turing SM 7.5
- **PyTorch**: 2.8.0+cu126
- **OS**: WSL Ubuntu-22.04

---

## 快速启动

```bash
cd /home/wjd/project/train
bash run.sh
```

`run.sh` 会 source conda、activate insta360、执行 `python train.py`。

**断点续训**：直接重新跑 `bash run.sh`，自动从 `checkpoints/` 恢复最新 checkpoint。兼容 compile/no-compile 切换。

**命令行覆盖参数**：
```bash
bash run.sh --epochs 1                    # 跑 1 个 epoch 测试
bash run.sh --d_model 384 --epochs 1      # 换小模型对比
bash run.sh --batch_size 96               # 降 batch size 省显存
bash run.sh --no_compile                  # 禁用 torch.compile（当前默认已是禁用）
bash run.sh --no_amp                      # 禁用混合精度
```

---

## 项目结构

```
train/
├── config.py              ← 所有超参数（单点修改）
├── train.py               ← 训练主入口，自动写日志到 logs/
├── trainer.py             ← 训练循环、Noam 调度器、混合精度、checkpoint
├── model.py               ← Transformer Encoder-Decoder（PyTorch nn.Transformer）
├── dataset_tokenized.py   ← 从预 tokenize 的 .npy mmap 文件加载数据
├── tokenizer.py           ← SentencePiece BPE 分词器训练/加载
├── translate.py           ← 推理/翻译脚本
├── extract_data.py        ← 从原始 CSV 提取+过滤+分片（用 csv.reader，不要 naive split）
├── pre_tokenize.py        ← 将文本数据预 tokenize 为 .npy（流式写入，低内存）
├── run.sh                 ← 一键启动

├── bench_speed_vs_quality.py   ← d_model/L 扫参对比脚本
├── bench_compile_backends.py   ← torch.compile backend 测试
├── bench_optimize.py           ← batch size sweep
├── bench_dataloader.py         ← DataLoader 瓶颈分析
├── bench_tokenized.py          ← 实时 vs 预 tokenize 对比
├── profile_hotspots.py         ← PyTorch Profiler 热点分析
├── analyze_model.py            ← 模型参数结构分析
├── check_csv.py                ← 原始 CSV 格式验证
├── data_quality.py             ← 数据质量分析

├── data/                  ← 清洗后的平行语料（文本，18.8M 行）
│   ├── train.zh / train.en
│   ├── val.zh   / val.en
│   └── test.zh  / test.en
├── data_tokenized/        ← 预 tokenize 的 mmap .npy 数组
├── checkpoints/           ← 训练 checkpoint（自动创建）
├── logs/                  ← 训练日志（自动创建，tee 到终端+文件）
├── spm_bpe.model          ← 已训练的 BPE 分词器（32K vocab）
└── spm_bpe.vocab          ← 词表文件
```

---

## 看日志

```bash
# 实时追踪
tail -f logs/train_*.log

# 最近 20 行
tail -20 logs/train_*.log

# 只看关键指标
grep "ep " logs/train_*.log
```

日志格式：
```
ep 1/20 |   2.4% | step   3500 | loss 4.3904 | lr 3.06e-04 | 97,690 tok/s | ETA 155min
         ↑              ↑          ↑               ↑              ↑               ↑
     epoch 进度      步数         损失值          学习率          吞吐量          本 epoch 剩余时间
```

验证在每 5000 步打印：
```
--- val loss 4.2166 (best) ---
```

日志同时写终端和文件（train.py 里 Tee 类实现）。

---

## 配置说明 (config.py)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `epochs` | 20 | 训练轮数 |
| `batch_size` | 128 | 每批大小 |
| `d_model` | 512 | 模型维度 |
| `nhead` | 8 | 注意力头数 |
| `num_encoder_layers` | 6 | 编码器层数 |
| `num_decoder_layers` | 6 | 解码器层数 |
| `dim_feedforward` | 2048 | FFN 隐藏维度 |
| `max_seq_len` | 96 | 最大序列长度（p99<63，96 够用且有显存余量） |
| `dropout` | 0.1 | Dropout 比例 |
| `lr` | 1.0 | Noam 调度器学习率因子 |
| `warmup_steps` | 4000 | LR warmup 步数 |
| `label_smoothing` | 0.1 | 标签平滑 |
| `max_train_samples` | 0 | 0=全部数据 |
| `compile_mode` | "" | torch.compile 禁用（见下文说明） |
| `grad_accum_steps` | 2 | 梯度累积步数 |
| `log_interval` | 500 | ~每 45 秒打印一次 |
| `val_interval` | 5000 | ~每 8 分钟验证一次 |
| `save_interval` | 10000 | ~每 15 分钟保存 checkpoint |

---

## 数据来源与处理

原始 CSV：`WMT-Chinese-to-English-Machine-Translation-Training-Corpus-new/wmt_zh_en_training_corpus.csv`
- 2475 万行，标准 CSV（2 字段：zh_text, en_text）
- **必须用 `csv.reader` 解析**，不能用 `str.split(',')`（中文含半角逗号会导致错位）

重新提取全量数据：
```bash
python extract_data.py --train_lines 0     # 全部 2475 万行 → 过滤后 ~1884 万训练句对
```

重新预 tokenize（如果改了 max_seq_len 或重提数据后必须做）：
```bash
python pre_tokenize.py                     # 流式写入，内存安全
```

---

## 性能优化历史

| 优化 | 效果 | 状态 |
|------|------|------|
| AMP FP16（Tensor Cores） | ×2.7 | ✅ 默认开启 |
| `F.cross_entropy(label_smoothing=)` 替代自定义 loss | +66% | ✅ 已写入 |
| 预 tokenize 数据（mmap .npy） | +14% | ✅ 已写入 |
| `non_blocking=True` 数据传输 | +3-5% | ✅ 已写入 |
| `persistent_workers=True` | minor | ✅ 已写入 |
| Fused AdamW | minor | ✅ 已写入 |
| TF32 关闭（Turing 不支持） | 避免减速 | ✅ 已写入 |
| `batch_size=128`（22GB 安全值） | — | ✅ |
| `max_seq_len=96`（VRAM 余量） | — | ✅ |
| torch.compile (eager) | 无效果 | ❌ Dynamo 开销抵消收益 |
| torch.compile (inductor) | 崩溃 | ❌ PyTorch 2.8 sympy bug（动态 shape） |
| torch.compile (cudagraphs) | 崩溃 | ❌ GradScaler 与 CUDA graph 冲突 |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` | 慢 | ❌ 导致 backward 变慢，已移除 |

---

## 预估训练时间

```
当前配置: B=128, d=512, L=6, max_seq_len=96
实测吞吐: ~98,000 tok/s
每 epoch token 数: ~1.07B
每 epoch 耗时: ~3.0 小时
20 epoch 总耗时: ~60 小时 ≈ 2.5 天
```

---

## d_model 加速对比

想加速可以缩模型。benchmark 结果（50K 子集，1 epoch）：

```
Config                  Params    tok/s    加速比    final loss    vs baseline
─────────────────────────────────────────────────────────────────────────────
baseline  d=512 L=6     93.5M     51K      ×1.0     6.60         —
smaller   d=384 L=6     61.8M     66K      ×1.3     6.74        +0.14
shallow   d=384 L=4     53.6M     87K      ×1.7     6.72        +0.12  ← 推荐
tiny      d=256 L=6     35.7M     93K      ×1.8     6.98        +0.39
micro     d=256 L=4     32.0M    119K      ×2.3     6.99        +0.39
```

**`d=384, L=4` 是甜点**：loss 只高 0.12，速度 1.7×。预估 20 epoch ~1.5 天。

切换方法：修改 `config.py` 中 `d_model=384, num_encoder_layers=4, num_decoder_layers=4, nhead=8`。

---

## GPU 监控

```bash
nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,memory.used --format=csv
```

正常训练时：GPU 利用率 99%，温度 65-75°C，显存 18-22GB（PyTorch 分配器会预留，不会爆）。

---

## 当前状态

- 所有优化已落地，训练脚本就绪
- 上次训练跑到 epoch 1 step 5000，val loss 4.22（已删除 checkpoint 做干净测试）
- 直接 `bash run.sh` 即可启动全新训练
