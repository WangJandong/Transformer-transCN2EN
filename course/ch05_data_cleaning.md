# 第 5 章：数据清洗流水线

## 本章目标

1. 掌握 NLP 平行语料的 5 道标准过滤
2. 理解每道过滤的阈值怎么选
3. 能设计一个完整的清洗 pipeline

---

## 5.1 我们的数据集有多脏

从 2475 万行原始 CSV 中，正确解析后，质量过滤的统计数据：

```
过滤项                 淘汰数量       占比
────────────────────────────────────────────
精确重复句对           4,110,279    16.61%
长度比异常              425,794     1.72%
语种错配                289,366     1.17%
过短/过长              701,960     2.84%
────────────────────────────────────────────
合计淘汰             5,527,399    22.33%
最终保留             19,224,993    77.67%
```

**每 4 行里就有 1 行不能直接用。**

---

## 5.2 五道过滤详解

### 第 1 道：长度过滤

```python
MIN_CHARS = 4
MAX_CHARS = 400

if len(zh) < MIN_CHARS or len(en) < MIN_CHARS:
    drop()  # 太短的句子没有翻译价值
if len(zh) > MAX_CHARS or len(en) > MAX_CHARS:
    drop()  # 太长的可能是错误拼接
```

淘汰了 2.84%。

### 第 2 道：长度比过滤

```python
ratio = len(zh) / max(len(en), 1)
if ratio < 0.25 or ratio > 4.0:
    drop()
```

中英翻译中，中文字数通常比英文字数少 20-40%。如果中文是英文的 4 倍长，大概率对错了行。淘汰了 1.72%。

### 第 3 道：语种检测

```python
def frac_cjk(text):
    return len(re.findall(r'[一-鿿]', text)) / len(text)

if frac_cjk(zh) < 0.08:  # 中文端几乎没有汉字
    drop()
if frac_latin(en) < 0.25:  # 英文端拉丁字母太少
    drop()
```

淘汰了 1.17%。

### 第 4 道：去重

```python
pair = (zh, en)
if pair in seen:
    drop()
else:
    seen.add(pair)
```

完全相同的(中文, 英文)句对只保留第一份。淘汰了 16.61%——**这是占比最大的噪音**。

### 第 5 道：hash 分片

用 CRC32 做确定性分片，保证同样的句子永远落在同一个 split：

```python
h = zlib.crc32(zh.encode()) ^ zlib.crc32(en.encode())
if h % 100 < 98:  → train (98%)
elif h % 100 == 98: → val  (1%)
else:               → test (1%)
```

---

## 5.3 阈值怎么选

我们的阈值是基于统计分布选的：

```
中文 BPE token 长度:  mean=23  median=22  p95=42  max=149
英文 BPE token 长度:  mean=25  median=24  p95=45  max=193
长度比 (zh_tok/en_tok): p5=0.62  p95=1.33
```

所以 `max_tokens=200`、`ratio ∈ [0.25, 4.0]` 不会误杀正常数据。

---

## 5.4 练习

1. 从 `data/train.zh` 随机抽 100 行，手动检查有没有对错的句对
2. 改动 `extract_data.py` 的过滤阈值（比如 ratio 从 [0.25,4.0] 改成 [0.5,2.0]），观察保留率变化
3. 思考：去重 16.6% 是好事还是坏事？
