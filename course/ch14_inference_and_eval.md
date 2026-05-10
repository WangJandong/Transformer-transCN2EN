# 第 14 章：推理与效果评估

## 本章目标

1. 理解 greedy decode 和 beam search 的推理过程
2. 会用训练好的模型做翻译
3. 能初步评价翻译质量

---

## 14.1 Greedy Decode

训练好的模型推理时不能像训练那样"看到正确答案"——必须逐词生成：

```
Step 1: <bos> → 模型预测 → "The"
Step 2: <bos> The → 模型预测 → "weather"
Step 3: <bos> The weather → 模型预测 → "is"
...
直到生成 <eos> 或达到最大长度
```

每次选概率最高的下一个词，这叫**贪心解码**。

---

## 14.2 Beam Search

贪心解码的问题：第一步选错了，后面全错。Beam search 同时保留 k 条候选路径：

```
beam=2:
  第 1 步:  "The"(0.8),  "It"(0.6)           ← 保留 2 个
  第 2 步:  "The weather"(0.64), "It is"(0.48)  ← 各扩展 2 个，保留最好的 2 个
  第 3 步:  ...
```

更大概率找到全局最优，但推理速度是 greedy 的 k 倍。

---

## 14.3 推理脚本使用

```bash
# 单句翻译
python translate.py --text "今天天气很好。" --checkpoint checkpoints/best.pt

# 文件批量翻译
python translate.py --file input.txt --checkpoint checkpoints/best.pt --beam 4

# 交互模式
python translate.py --checkpoint checkpoints/best.pt
> 今天天气很好。
it's a fine day today.
```

---

## 14.4 实际翻译效果

用 best.pt（val loss 2.70）跑了几句中文：

```
今天天气很好。           → it's a fine day today.
人工智能正在改变世界。    → AI is in the process of transforming the world
我昨天去了超市买东西。    → I went to the supermarket for things.
这本书非常有趣，我推荐你读。→ this book is not always funny to recommend you read...
他每天早上六点起床跑步。  → he runs his bed every morning at six o'clock.
```

**分析**：
- 短句和常见表达好（"AI is transforming", "fine day today"）
- 复杂语义出错（"非常有趣"→"not always funny"）
- 个别幻觉（"起床跑步"→"runs his bed"）

改进方向：多训几个 epoch、更大的模型、或用 BLEU 量化评估。

---

## 14.5 BLEU 评分

BLEU 衡量机器翻译和参考译文的 n-gram 重叠率。粗略解释：

```
候选: "the weather is nice today"
参考: "the weather is good today"

1-gram 匹配: the, weather, is, today → 4/5 = 0.8
2-gram 匹配: "the weather", "is today" → 2/4 = 0.5
...
BLEU ≈ 0.65
```

可以用 `sacrebleu` 库在测试集上算 BLEU。

---

## 14.6 练习

1. 用 `python translate.py --text "..." --beam 1` 和 `--beam 4` 翻译同一句，对比结果
2. 自己写 5 句中文，看模型翻译得如何，分类哪些好哪些坏
3. 思考：为什么训练时用 teacher forcing、推理时用 autoregressive？
