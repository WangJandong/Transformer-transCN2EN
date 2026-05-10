# 第 13 章：torch.compile 翻车记

## 本章目标

1. 理解 torch.compile 三大 backend 的区别
2. 知道 PyTorch 2.8 + Turing 上动态 shape 的已知 bug
3. 学会定位框架 bug 的完整流程

---

## 13.1 期望：一行代码 +15%

```python
model = torch.compile(model)  # 理想：自动融合 kernel，快 10-30%
```

现实：三个 backend 全崩。

---

## 13.2 Backend 1: inductor（默认）→ CRASH

```python
model = torch.compile(model, backend="inductor")
```

第一个 batch forward 就崩溃：

```
torch._inductor.exc.InductorError:
AssertionError: -377531338003753/1000000000000000
```

堆栈指向 `torch/_inductor/tiling_utils.py:483`。根因：inductor 在分析动态 shape 的内存合并模式时，sympy 表达式算出负数，触发了内部断言。

这是 PyTorch 2.8 的已知回归 bug（PR #169726 修了）。我们基准测试中能跑通（固定 S=50），但真实训练中序列长度每 batch 不同就崩。

---

## 13.3 Backend 2: eager → 没效果

```python
model = torch.compile(model, backend="eager")
```

不崩溃，但也不快——Dynamo 图捕获的开销恰好抵消了优化收益（95K vs 98K tok/s）。只做 Python 字节码优化，没有 CUDA kernel 融合。

---

## 13.4 Backend 3: cudagraphs → 崩溃

```python
model = torch.compile(model, backend="cudagraphs")
```

CUDA Graph 把整个训练步骤录制下来，消除 kernel launch 开销。但和 `GradScaler` 冲突：

```
RuntimeError: accessing tensor output of CUDAGraphs that has been
overwritten by a subsequent run.
```

因为 GradScaler 在 backward 期间原地修改了 CUDA Graph 捕获的 tensor。

---

## 13.5 Benchmark 对比

测试脚本 `bench_compile_backends.py`，B=128, 2000 句对：

```
backend          tok/s      状态
──────────────────────────────────
无 compile       78,266     ✅ 基线
inductor         101,951    ⚠️ 小数据 ok，真实训练崩溃
eager            104,309    ⚠️ 小数据 +33%，真实训练无效果
cudagraphs       105,476    ❌ GradScaler 冲突
```

注意：benchmark 用了小数据集（序列长度变化小），所以 inductor 没崩。真实训练才触发 bug。

---

## 13.6 定位框架 bug 的标准流程

1. **复现**：找到能稳定触发的最小输入（我们的 case：动态长度的真实训练数据）
2. **缩小范围**：换 backend/参数确认是哪个组件的问题
3. **读堆栈**：找到崩溃的文件和行号（`tiling_utils.py:483`）
4. **搜索 GitHub**：用文件名+错误信息搜 issues/PRs
5. **确认修没修**：看修复 commit 的合入时间 vs 你的 PyTorch 版本
6. **workaround**：换 backend、降级、或放弃等新版

---

## 13.7 后续：nightly 可能修了

```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126
```

安装后跑 `bench_compile_backends.py`，如果 inductor 稳定跑下来，白捡 +15%。

---

## 13.8 练习

1. 跑 `python bench_compile_backends.py`，看看你的环境哪个 backend 能用
2. 找到 PyTorch GitHub 的 PR #169726，读一下修复了什么
3. 思考：为什么小数据集 benchmark 和大数据集真实训练，行为不同？
