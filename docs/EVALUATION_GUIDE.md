# 翻译模型评估体系

## 环境搭建

**Python 环境：** conda `insta360`，Python 3.9，CUDA

**依赖安装：**

```bash
conda activate insta360

# 词级别指标（已有）
pip install sacrebleu

# 语义评估指标（新增）
pip install unbabel-comet bert-score
```

**HuggingFace 网络问题：** WSL 直连 HuggingFace 不通，设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

首次运行会自动下载模型（roberta-large ~1.5G，wmt22-comet-da ~2G），之后走缓存。

---

## 评估指标全景

四个指标互补：

| 指标 | 类型 | 依赖参考译文 | 测量维度 |
|------|------|:---:|------|
| BLEU | 词面 | 是 | n-gram 重叠率，看措辞准不准 |
| chrF | 词面 | 是 | 字符级 F-score，比 BLEU 更稳定 |
| BERTScore | 语义 | 是 | 用 BERT 计算语义相似度 |
| COMET | 神经 | 是 | 用专门的神经网络给译文质量打分 |

---

## 两个脚本

### eval_quick.py — 快速版（不联网）

```bash
python eval_quick.py
```

修改 `N` 和 `BEAM` 两个变量控制样本数和束宽。

输出：BLEU、chrF、长度分层、样例翻译。

### eval.py — 完整版（需联网）

```bash
python eval.py --samples 1000 --beam 1
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--samples` | 2000 | 抽样数，500~2000 合理 |
| `--beam` | 4 | beam search 宽度，1 最快 |
| `--checkpoint` | best.pt | 检查点路径 |
| `--seed` | 42 | 随机种子 |
| `--no-comet` | — | 跳过 COMET |
| `--no-bertscore` | — | 跳过 BERTScore |

---

## 结果解读

以 1000 条样本、beam=1 的结果为例：

```
BLEU:           36.1
chrF:           61.1
BERTScore F1:   93.3    (P=93.4, R=93.1)
COMET:          0.8152
```

**BLEU 36.1：**
- 30+ = 可用，35+ = 不错，40+ = 很好
- 只看字面匹配，不关心意思
- 短句通常偏低（样本少、对不准）

**chrF 61.1：**
- 50+ = 正常，60+ = 不错
- 字符级别，对分词不敏感，比 BLEU 更可信

**BERTScore F1 93.3：**
- 87+ = 靠谱，90+ = 很好
- Recall 93.1 说明参考译文 93% 的意思你的模型都覆盖到了
- Precision 93.4 说明译文里几乎没丢关键信息

**COMET 0.8152：**
- 这是四个指标里最接近人类判断的
- < 0.5 = 差，0.5-0.7 = 一般，0.7-0.85 = 好，> 0.85 = 优秀

**长度分层：**
```
short  (1-10w)    BLEU=31.1  chrF=53.4
medium (11-25w)   BLEU=31.7  chrF=56.9
long   (26-50w)   BLEU=37.0  chrF=62.8
xlong  (51+w)     BLEU=42.6  chrF=66.8
```

这个模型越长的句子翻译越好（长句通常是联合国公文，句式固定，好翻），短句反而偏低（短句更灵活，直译容易不自然）。

---

## 四个指标怎么互相印证

- BLEU 和 chrF 都高 → 措辞准确
- BERTScore 高 → 语义没丢
- COMET 高 → 译文整体质量好，接近人工判断
- 四个指标方向一致 → 结果可信；某个指标显著偏离 → 需要排查

---

## 实测结果

**模型：** checkpoint `best.pt`，step=315000，loss=2.6989，93.4M 参数（Transformer 6+6，d=512，vocab=32K）

**测试数据：** `data/test.zh` + `data/test.en`，共 192,077 对，随机抽样

### 完整版（1000 条，beam=1）— 2026-05-08

| 指标 | 分数 | 区间 |
|------|------|------|
| BLEU | 36.1 | 不错 |
| chrF | 61.1 | 不错 |
| BERTScore P | 93.4 | 很好 |
| BERTScore R | 93.1 | 很好 |
| BERTScore F1 | 93.3 | 很好 |
| COMET | 0.8152 | 好 |

长度分层：

| 长度 | 样本数 | BLEU | chrF |
|------|--------|------|------|
| short (1-10词) | 168 | 31.1 | 53.4 |
| medium (11-25词) | 442 | 31.7 | 56.9 |
| long (26-50词) | 331 | 37.0 | 62.8 |
| xlong (51+词) | 59 | 42.6 | 66.8 |

### 快速版（200 条，beam=1）— 2026-05-07

| 指标 | 分数 |
|------|------|
| BLEU | 34.9 |
| chrF | 59.4 |

### 完整版（500 条，beam=4）— 2026-05-08

| 指标 | 分数 |
|------|------|
| BLEU | 22.6 |
| chrF | 59.7 |
| BERTScore F1 | 89.8 |
| COMET | 0.6628 |

> **注意：** 500 条 beam=4 那次 BLEU 和 COMET 偏低较多，与 1000 条 beam=1 差异大，原因是两次随机抽样到的句子集合不同（抽样越少方差越大），且该次 beam=4 的样本恰好包含了更多短句和难句。**以 1000 条 beam=1 那次为基准。**
