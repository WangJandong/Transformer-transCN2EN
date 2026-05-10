# 第 4 章：CSV 解析的血案

## 本章目标

1. 理解为什么 `str.split(',')` 解析 CSV 会静默损坏数据
2. 会用 Python `csv.reader` 正确解析含逗号的中文 CSV
3. 学会验证平行语料的对齐质量

---

## 4.1 故事的开始

我们有一个 6.3 GB 的原始 CSV——2475 万行中英句对：

```
表演 的 明星 是 X 女孩 团队 ...,"the show stars the X Girls..."
```

有人写了个脚本把 CSV 拆成 `train.zh` 和 `train.en`：

```python
# 原始脚本（有 bug！）
for line in csv_file:
    parts = line.split(',', 1)
    zh = parts[0]
    en = parts[1]
```

结果：**1884 万行训练数据里，6.9% 的中文句以引号开头，4.7% 的英文句混入了中文**。

---

## 4.2 Bug 是怎么发生的

中文里含半角逗号 `,`。当 `split(',', 1)` 碰到这行：

```
"高 台上 原 建有 沙丘 寺 , 现 仅存 石碑 两通 .","high platform of..."
```

它把句子切成了：
```
parts[0] = "高 台上 原 建有 沙丘 寺 "
parts[1] = 现 仅存 石碑 两通 .","high platform of..."
```

part[0] 被当成中文（丢了后半句），part[1] 前面混进了中文（因为 CSV 的引号边界被打破了）。

---

## 4.3 正确做法：用 csv.reader

```python
import csv

with open(csv_path) as f:
    reader = csv.reader(f)
    for zh, en in reader:
        # zh 和 en 已经被正确解析，逗号在引号内不会误切
        pass
```

`csv.reader` 懂 RFC 4180 标准——引号内的逗号不会被当成字段分隔符。

---

## 4.4 怎么验证数据对齐质量

写一个简单的检查脚本：

```python
import re
cjk = re.compile(r'[一-鿿]')

zh_bad = 0  # 中文行以引号开头
en_bad = 0  # 英文行含中文

for zh, en in zip(open('train.zh'), open('train.en')):
    if zh.startswith('"'):
        zh_bad += 1
    if len(cjk.findall(en)) > 5:
        en_bad += 1

print(f'中文行引号残留: {zh_bad} ({zh_bad/total:.1%})')
print(f'英文行中文混入: {en_bad} ({en_bad/total:.1%})')
```

修复前：6.9% 和 4.7%。修复后：0.01% 和 0.003%。

---

## 4.5 教训

1. **永远不要手动解析 CSV/JSON/XML**——用标准库
2. **数据出了问题再去排查，成本远高于一开始做对**
3. 中文 NLP 项目的 CSV 尤其容易踩坑——中文含英文标点是常态

---

## 4.6 练习

1. 手写一行含逗号的中文 CSV，分别用 `split(',')` 和 `csv.reader` 解析，对比结果
2. 打开 `extract_data.py`，找到 `csv.reader` 的使用位置
3. 想一想：如果 CSV 里的英文也用逗号分隔（而非引号包裹），还能正确解析吗？
