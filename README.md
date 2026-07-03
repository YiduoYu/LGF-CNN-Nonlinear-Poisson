# Darcy-Linear — Two PDE Solvers

> **Branch `Darcy-Linear`:** 线性 Darcy 方程的两种求解方法对比：data-driven CNN operator vs mesh-free PINN。

目标 PDE：

```text
-div(a(x,y) grad u) = f,   u = g on boundary,   Omega = [-1,1]^2
```

a(x,y) 为正系数场。纯物理驱动训练，不使用 reference solution 作为监督信号（u_ref 仅用于评估）。

---

## 文件说明

| 文件 | 方法 | 说明 |
|------|------|------|
| `Baseline.py` | CNN Operator | 在随机 PDE 族上训练 CNN，一次训练后对任意新 PDE 做零样本预测 |
| `PINN_sincos.py` | Physics-Informed NN | Raissi 风格 PINN，逐样本训练，求解单个特例 `u = sin(πx)cos(πy)` |

### Baseline.py — CNN 算子学习

学习从 `(f, g, a)` → `u` 的映射。一次训练后可以零样本泛化到未见过的 PDE。

- **训练：** 2048 个随机生成的 PDE，纯 PDE 残差 loss
- **推理：** 前向传播一次，< 1ms
- **适用场景：** 需要快速求解大量参数化 PDE

### PINN_sincos.py — 物理信息神经网络

对单个 PDE 逐样本训练，网络直接拟合解函数 `u(x, y)`。

- **训练：** Adam 5000 步 + L-BFGS 5000 步，~10k 内部配点 + ~400 边界点
- **推理：** 需要为每个新 PDE 重新训练
- **适用场景：** 高精度求解特定 PDE，无需训练数据

---

## Quick Start

### CNN Baseline

```bash
# 训练
python3 Baseline.py --N 32 --epochs 40 --train-samples 2048 --val-samples 128 --retrain

# 随机 PDE 预测
python3 Baseline.py --load-path checkpoints/cnn/<key>/best.pt --predict --predict-samples 8 --plot

# sincos 测试
python3 Baseline.py --load-path checkpoints/cnn/<key>/best.pt --predict --predict-case sincos --plot

# 小规模调试
python3 Baseline.py --N 8 --epochs 2 --train-samples 32 --val-samples 8 --batch-size 4 --device cpu --retrain
```

### PINN (sincos)

```bash
# 完整训练
python3 PINN_sincos.py --N-f 10000 --N-b 400 --adam-epochs 5000 --lbfgs-max-iter 5000 --plot

# 加载已训练模型评估
python3 PINN_sincos.py --predict --load-path results/pinn_sincos/lbfgs_final.pt --N 32 --plot

# 快速调试
python3 PINN_sincos.py --N-f 1000 --N-b 100 --adam-epochs 200 --lbfgs-max-iter 500
```

---

## 方法对比

| | CNN (Baseline) | PINN (Raissi) |
|---|---|---|
| 训练方式 | 监督学习（PDE 残差） | 无网格配点法 |
| 泛化 | 零样本，一次训练多 PDE | 逐样本重训 |
| 单次推理 | ~1 ms（前向传播） | 需完整训练流程 |
| 精度 (sincos) | ~10⁻⁴ relL2（需预训练） | **4.6×10⁻⁴** relL2 |
| 核心思想 | 学算子映射 | 学函数本身 |
| 参考 | — | Raissi, Perdikaris, Karniadakis (2019) |

---

## 存储结构

```
Baseline.py →
checkpoints/cnn/N32_e40_main_ab1.5_as0.5_am0.1_fs3_gs0.5_tr2048/
    ├── best.pt / last.pt / history.csv / convergence.png

PINN_sincos.py →
results/pinn_sincos/
    ├── adam_epoch{00500..05000}.pt   ← 每 500 步 checkpoint
    ├── adam_best.pt                  ← Adam 阶段最优
    ├── lbfgs_final.pt                ← L-BFGS 收敛后最终模型
    ├── history.csv
    └── pinn_sincos.png
```

---

## 评估指标

| Metric | 含义 |
|--------|------|
| relL2 | 预测解 vs 参考解的相对 L2 误差（内部点） |
| resRMSE | PDE 残差的 RMS（训练目标） |
| linf | 内部节点最大绝对误差 |
| gradRelL2 | 梯度的相对 L2 误差 |
| bmax | 边界最大误差 |

---

## CNN 关键设计

- Hard Dirichlet BC: `u_pred = bnd * g + (1-bnd) * u_raw`
- Conservative FD residual: 交界面系数 `a_face = 0.5*(a_c + a_neighbor)`
- 6-channel input: `[f_norm, g, boundary_mask, a_norm, x, y]`，f/a 用训练集统计量 z-score 归一化
- 线性 FD solver: `scipy.sparse.linalg.spsolve`，无需 Picard 迭代
- 优化器: AdamW + gradient clipping + cosine annealing（5% warmup）
- 每个样本独立归一化 loss: denom 为单样本 f 的 RMS

## PINN 关键设计

- 架构: tanh MLP `[2, 50, 50, 50, 50, 1]`，Xavier 初始化，输入缩放到 [-1, 1]
- 损失: PDE 残差 + 边界条件，自动微分计算拉普拉斯
- 训练: Adam (lr=1e-3) → L-BFGS (strong_wolfe, lr=0.8)
- 配点: 10k 内部（LHS 采样）+ 400 边界（均匀网格）
- 参考: Raissi, Perdikaris, Karniadakis, "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations", JCP 2019
