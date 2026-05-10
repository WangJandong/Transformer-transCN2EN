import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Parse log
train_steps, train_loss, val_steps, val_loss, lr, tok_s = [], [], [], [], [], []
with open('logs/train_20260506_091955.log', 'r') as f:
    for line in f:
        # Training step
        m = re.search(r'step\s+(\d+)\s*\|\s*loss\s+([\d.]+)\s*\|\s*lr\s+([\dee\-.]+)\s*\|\s*([\d,]+)\s*tok/s', line)
        if m:
            train_steps.append(int(m.group(1)))
            train_loss.append(float(m.group(2)))
            lr.append(float(m.group(3)))
            tok_s.append(float(m.group(4).replace(',', '')))
        # Validation
        m = re.search(r'val loss\s+([\d.]+)', line)
        if m:
            val_steps.append(train_steps[-1] if train_steps else 0)
            val_loss.append(float(m.group(1)))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Training Report — 93.4M params, RTX 2080 Ti', fontsize=14, fontweight='bold')

# 1. Loss
ax = axes[0][0]
ax.plot(train_steps, train_loss, alpha=0.3, color='steelblue', linewidth=0.5, label='Train loss (per step)')
ax.plot(val_steps, val_loss, 'o-', color='darkorange', markersize=5, linewidth=2, label='Val loss')
best_idx = val_loss.index(min(val_loss))
ax.annotate(f'Best: {val_loss[best_idx]:.4f} @ step {val_steps[best_idx]}',
            (val_steps[best_idx], val_loss[best_idx]),
            xytext=(val_steps[best_idx], val_loss[best_idx] + 0.08),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=9, color='red', fontweight='bold')
ax.set_xlabel('Step')
ax.set_ylabel('Loss')
ax.set_title('Training & Validation Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. LR
ax = axes[0][1]
ax.plot(train_steps, lr, color='green', linewidth=1)
ax.set_xlabel('Step')
ax.set_ylabel('Learning Rate')
ax.set_title('Learning Rate Schedule')
ax.grid(True, alpha=0.3)
ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

# 3. Throughput
ax = axes[1][0]
ax.plot(train_steps, tok_s, alpha=0.5, color='steelblue', linewidth=0.5)
# Highlight validation dips
val_mask = [s in val_steps for s in train_steps]
ax.scatter([train_steps[i] for i in range(len(train_steps)) if val_mask[i]],
           [tok_s[i] for i in range(len(tok_s)) if val_mask[i]],
           color='darkorange', s=8, alpha=0.6, label='Val steps (lower)')
ax.set_xlabel('Step')
ax.set_ylabel('tok/s')
ax.set_title('Throughput')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 4. Train loss moving average + val loss
ax = axes[1][1]
window = 500
ma = []
for i in range(len(train_loss)):
    start = max(0, i - window)
    ma.append(sum(train_loss[start:i+1]) / (i - start + 1))
ax.plot(train_steps, ma, color='steelblue', linewidth=1.5, label=f'Train loss (MA-{window})')
ax.plot(val_steps, val_loss, 'o-', color='darkorange', markersize=5, linewidth=2, label='Val loss')
ax.axvline(x=val_steps[best_idx], color='red', linestyle='--', alpha=0.4)
ax.set_xlabel('Step')
ax.set_ylabel('Loss')
ax.set_title(f'Train MA-{window} & Val Loss (gap ~{min(ma[-500:]) - val_loss[best_idx]:.2f})')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_report.png', dpi=150)
print(f'Saved training_report.png')
print(f'Best val loss: {min(val_loss):.4f} at step {val_steps[val_loss.index(min(val_loss))]}')
print(f'Best val steps: {[(s, v) for s, v in zip(val_steps, val_loss) if "(best)" in str(v)]}')
