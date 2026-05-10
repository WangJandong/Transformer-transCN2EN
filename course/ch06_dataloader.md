# 第 6 章：高效数据加载

## 本章目标

1. 理解为什么 DataLoader 可以成为训练瓶颈
2. 掌握预 tokenize + mmap 的流式数据方案
3. 理解 padding bucket 的原理和效果

---

## 6.1 问题：数据 I/O 拖慢训练

如果每个 batch 都要：
1. 从磁盘读文本 → 2. SentencePiece 编码 → 3. pad 到等长 → 4. 送到 GPU

那么 CPU tokenization 占 ~15% 的训练时间。GPU 等 CPU 就是浪费。

---

## 6.2 方案 1：预 tokenize

**一次性把所有文本编码成 int32，存为 .npy 文件。**

### 预分词过程

```python
# pre_tokenize.py
ids = sp.encode(text, out_type=int)           # ① SentencePiece 分词
ids = [BOS_ID] + ids[:max_len-2] + [EOS_ID]   # ② 加 BOS/EOS，截断到 max_seq_len
# ③ 追加写入 .npy
```

**注意第 ② 步**：
- `BOS_ID = 1`：句子开头标记（decoder 的第一个输入）
- `EOS_ID = 2`：句子结尾标记（告诉模型"生成结束"）
- `[:max_len-2]`：给 BOS 和 EOS 留位置，确保最终长度不超过 `max_seq_len`

例如 `max_seq_len=96`：原始分词结果最多 94 个 token，加头尾刚好 96。

### 存储格式

预分词生成 3 种 `.npy` 文件：

```
data_tokenized/
├── train_src_ids.npy      (1.9G)  # 所有 token id 平铺在一起
├── train_src_offsets.npy  (144M)  # 每句的起始偏移量
├── train_src_lengths.npy  (72M)   # 每句的长度
```

结构示意：
```
ids:     [1, 502, 30, 2,  1, 80, 15, 2,  1, 200, 45, ...]
          ↑句子0        ↑句子1         ↑句子2
offsets: [0,            4,             8,            ...]
lengths: [4,            4,             3,            ...]
```

取第 `i` 句：`all_ids[offsets[i] : offsets[i]+lengths[i]]` → O(1) 直接切片。

### 为什么用 mmap

训练时直接用 `np.load(path, mmap_mode='r')` 做内存映射——操作系统按需把文件映射到虚拟内存，**零拷贝、零解析开销**。

效果：DataLoader 从 650K tok/s 变成纯 I/O bound，不再拖慢 GPU。

---

## 6.3 方案 2：Padding Bucket（分桶）

### 问题

随机 batching 时，长短句混在一起。短句被迫 pad 到 batch 内最长句的长度。

```
Batch 内句子长度: [5, 8, 12, 45, 72] → 全部 pad 到 72
有用 token: 5+8+12+45+72 = 142
浪费 token: 72×5 - 142 = 218
有效比例: 142/360 = 39%
```

实测：随机 batching 的**有效 token 仅 34%**，其余 66% 是 padding。

### 解决方案

把长度相近的句子放进同一个 batch：

```python
# 在 Dataset 迭代时，按 src_length 排序缓冲区
chunk_lens = self.src_lengths[chunk]
sorted_order = np.argsort(chunk_lens)
chunk = chunk[sorted_order]
```

排序后 batch 内长度差异从 60+ 降到 ~5：

```
有效 token 比例: 34% → 69%（翻倍）
每 epoch 时间: 8.7h → 4.8h（减半）
```

---

## 6.4 Worker 分区

使用 `num_workers > 1` 时，每个 worker 默认遍历全量数据。必须显式分区：

```python
worker_info = torch.utils.data.get_worker_info()
if worker_info is not None:
    per_worker = len(indices) // worker_info.num_workers
    lo = worker_info.id * per_worker
    hi = lo + per_worker
    indices = indices[lo:hi]
```

否则 `num_workers=2` 会让每个 epoch 跑两遍。

---

## 6.5 non_blocking 传输

```python
src, tgt = src.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
```

`non_blocking=True` 让 CPU→GPU 数据传输与 GPU 计算重叠——上一 batch 的 backward 还在跑时，下一 batch 的数据已经开始搬运了。实测 +3-5% 吞吐提升。

---

## 6.6 练习

1. 跑 `python pre_tokenize.py --max_lines 50000`，观察 .npy 文件大小
2. 修改 `dataset_tokenized.py`，去掉分桶排序，测量 padding 率的变化
3. 对比 `num_workers=0` 和 `num_workers=4` 的 tok/s
