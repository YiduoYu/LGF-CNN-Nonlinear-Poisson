# Nonlinear Poisson Equation — CNN Solver

求解非线性通量形式 Poisson 方程：

```text
-div(u^k grad u) = f,   u = g on boundary,   Omega = [-1,1]^2
```

扩散系数为 u^k 本身（无外部 a(x,y) 场），u > 0，k > 0。

---

## 两条路线

| 文件 | 路线 | 用途 |
|------|------|------|
| `Baseline_nonlinear_picard.py` | **Route B: Picard** | 随机 PDE 生成 + 训练 + 预测（主力） |
| `Baseline_nonlinear_mms.py` | **Route A: MMS** | 人造解验证 / 过拟合 debug（辅助） |

### Route B — Picard

随机采样 f, g, k → Picard 迭代求解 u_ref → CNN 学习 (f, g, k) → u_ref。

### Route A — MMS

构造 `u_ref = 1 + c1*x + c2*y + 0.15*(1-x²)(1-y²)*smooth`，FD 反算 f（oracle = 0），小数据集过拟合验证。

---

## Quick Start

```bash
# 参数预检
python Baseline_nonlinear_picard.py --picard-sanity --N 32 --train-samples 16 --k 2

# 训练（physics-only）
python Baseline_nonlinear_picard.py --N 32 --epochs 40 --train-samples 128 --k 2 --retrain --plot

# 训练（supervised）
python Baseline_nonlinear_picard.py --N 32 --epochs 40 --train-samples 128 --k 2 --loss supervised --retrain --plot

# 训练（随机 k）
python Baseline_nonlinear_picard.py --N 32 --epochs 40 --random-k --k-min 1 --k-max 3 --retrain --plot

# 预测（随机 PDE）
python Baseline_nonlinear_picard.py --predict --predict-from checkpoints/cnn/<key>/best.pt --plot

# 预测（sincos 泛化测试）
python Baseline_nonlinear_picard.py --predict --predict-from checkpoints/cnn/<key>/best.pt --predict-case sincos --plot

# MMS 验证
python Baseline_nonlinear_mms.py --mms-sanity --N 32 --train-samples 8 --k 2
python Baseline_nonlinear_mms.py --N 32 --epochs 200 --train-samples 32 --k 2 --retrain --plot
```

---

## 主要 CLI 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--N` | 32 | 内部网格点数 |
| `--epochs` | 100 | 训练轮数（MMS 默认 200） |
| `--train-samples` | 128 | 训练样本数（MMS 默认 32） |
| `--batch-size` | 8 | 批大小 |
| `--lr` | 1e-3 | 学习率 |
| `--base-ch` | 32 | 隐藏通道数 |
| `--n-layers` | 6 | 卷积层数 |
| `--k` | 2.0 | 扩散系数指数（MMS 默认 k∈[0.5,3.0]） |
| `--random-k` | False | 随机采样 k ∈ [k_min, k_max] |
| `--f-scale` | 0.1 | forcing 幅度 |
| `--g-base` | 1.0 | 边界基准值 |
| `--g-scale` | 0.05 | 边界随机波动 |
| `--loss` | physics | 训练模式：`physics`（PDE 残差）或 `supervised`（\|u-u_ref\|²） |
| `--coeff-eps` | 1e-4 | u^k 的 clamp 下界 |
| `--picard-damping` | 0.5 | Picard 阻尼系数（仅 Picard） |
| `--picard-tol` | 1e-8 | Picard 收敛容差（仅 Picard） |
| `--predict-from` | — | 加载 checkpoint 进行预测 |
| `--predict-case` | random | 预测模式：`random` 或 `sincos`（仅 Picard） |
| `--retrain` | False | 覆盖已有 checkpoint |
| `--picard-sanity` / `--mms-sanity` | False | 参数预检后退出 |

---

## 存储结构

```
checkpoints/
├── cnn/N32_e100_tr128_k2_fs0.1_gs0.05_ch32_L6/   # Picard
│   ├── best.pt, last.pt, history.csv, convergence.png, run_info.txt
├── cnn_mms/N32_e200_tr32_k2_ch32_L6_mms/          # MMS
│   ├── best.pt, last.pt, history.csv, convergence.png, run_info.txt
├── cnn/latest -> N32_e100_tr128_k2_...
├── cnn_mms/latest -> N32_e200_tr32_k2_...
└── registry.json                                   # 实验索引（自动维护）

results/
└── cnn/<key>/figures/
    ├── history/2026-07-04_113800/                  # 旧结果自动归档
    ├── sample_000_panel.png, metrics.csv
```

---

## 评估指标

| Metric | 含义 |
|--------|------|
| relL2 | u_pred vs u_ref 相对 L2 误差 |
| resRMSE | u_pred PDE 残差 RMS |
| refResRMSE | u_ref PDE 残差 RMS（oracle 下限） |
| linf | 最大绝对误差 |
| gradRelL2 | 梯度相对 L2 误差 |

---

## 关键设计

- **向量化组装**：5 点差分矩阵用 `scipy.sparse.coo_matrix` 构建，无 Python 循环
- **physics-only 或 supervised**：`--loss` 选训练目标，默认 PDE 残差，每个样本独立按 f 的 RMS 归一化
- **Picard 收敛判断**：比较 `∥u_new − u_old∥ / ∥u_new∥`，damping=1 时不会假收敛
- **预测复用训练分布**：`--predict` 自动从 checkpoint 继承 k、f_scale、coeff_eps 等参数
- **Hard Dirichlet BC**：`u_pred = bnd * g + (1-bnd) * u_raw`
- **Conservative FD**：交界面系数 = 相邻点算术平均
- **6 通道输入**：`[f_norm, g, boundary_mask, k_channel, x, y]`，f 和 k 用 z-score 归一化
- **sincos 泛化测试**：`u_ref = sin(πx)·cos(πy)`，FD 反算 f（oracle = 0），测试模型泛化
- **MMS 对齐 Picard**：两文件共享相同的 loss 选项、命名规范、存储系统
- **优化器**：AdamW + gradient clipping + warmup → cosine annealing
