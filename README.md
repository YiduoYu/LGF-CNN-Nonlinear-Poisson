# Darcy PDE — CNN Solver

求解 Darcy 型椭圆 PDE：

```text
-div(a(x,y) grad u) = f,   u = g on boundary,   Omega = [-1,1]^2
```

a(x,y) 为正系数场，u 允许正负值。

---

## 训练目标

**PDE 残差 only。** u_ref 仅用于生成 f、g 和评估指标，不参与训练。

```
R = -div_h(a * grad_h(u_pred)) - f
loss = mean((R / RMS(f))^2)
```

硬边界：`u_pred = boundary * g + (1-boundary) * u_raw`，无 positivity 约束。

---

## 数据生成

每个样本是**一道不同的 PDE**（operator-learning 风格）：

- 随机正系数场 a(x,y) = a_base + a_scale × smooth_field，clamp 到 a_min
- 随机 forcing f(x,y)，边界为零
- 随机 Dirichlet 边界 g(x,y)，可正可负
- 线性 FD solver（scipy.sparse.linalg.spsolve）求解 u_ref

---

## 模型架构

SimpleCNN：n 层 Conv3×3 + GroupNorm + GELU → 1×1 Conv 输出。6 通道输入：`[f, g, boundary_mask, a, x, y]`。

---

## Quick Start

```bash
# 训练
python Baseline.py --N 32 --epochs 40 --train-samples 2048 --val-samples 128 --retrain

# 随机 PDE 预测
python Baseline.py --load-path checkpoints/cnn/<key>/best.pt --predict --predict-samples 8 --plot

# sincos 测试（u_ref = sin(πx)·cos(πy)，a=1）
python Baseline.py --load-path checkpoints/cnn/<key>/best.pt --predict --predict-case sincos --plot

# 小规模调试
python Baseline.py --N 8 --epochs 2 --train-samples 32 --val-samples 8 --batch-size 4 --device cpu --retrain
```

---

## 主要 CLI 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--N` | 32 | 内部网格点数 |
| `--epochs` | 40 | 训练轮数 |
| `--train-samples` | 2048 | 训练样本数 |
| `--batch-size` | 16 | 批大小 |
| `--lr` | 1e-3 | 学习率 |
| `--base-ch` | 32 | 隐藏通道数 |
| `--n-layers` | 6 | 卷积层数 |
| `--a-base` | 1.5 | 系数场基准值 |
| `--a-scale` | 0.5 | 系数场波动幅度 |
| `--a-min` | 0.1 | 系数场最小值 |
| `--f-scale` | 3.0 | forcing 幅度 |
| `--g-scale` | 0.5 | 边界值幅度 |
| `--predict-case` | random | 预测模式：random / sincos |

---

## 存储结构

```
checkpoints/
└── cnn/N32_e40_darcy_ab1.5_as0.5_am0.1_fs3_gs0.5_tr2048/
    ├── best.pt / last.pt / history.csv / convergence.png / run_info.txt

results/
└── cnn/<key>/figures/
    ├── sample_000_panel.png
    └── metrics.csv
```

---

## 评估指标

| Metric | 含义 |
|--------|------|
| relL2 | 预测解 vs 参考解的相对 L2 误差（内部点） |
| resRMSE | PDE 残差的 RMS（训练目标） |
| linf | 内部节点最大绝对误差 |
| gradRelL2 | 梯度的相对 L2 误差 |
| bmax | 边界最大误差（硬边界下 = 0） |
| Oracle | 参考解在 FD 离散下的残差 RMS |

---

## 关键设计

- Hard Dirichlet BC：边界点直接替换
- u 允许正负：无 softplus 约束
- Conservative FD residual：交界面系数算术平均
- 6-channel input：`[f_norm, g, boundary_mask, a_norm, x, y]`
- 线性 FD solver：直接 spsolve，无需 Picard 迭代
