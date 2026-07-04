# Nonlinear Poisson Equation — CNN Solver

求解非线性通量形式 Poisson 方程（flux-form）：

```text
-div(u^k grad u) = f,   u = g on boundary,   Omega = [-1,1]^2
```

扩散系数为 u^k 本身（无外部 a(x,y) 场），u > 0，k > 0。物理驱动 + 监督联合训练。

---

## 两条路线

| 文件 | 路线 | 用途 |
|------|------|------|
| `Baseline_nonlinear_picard.py` | **Route B: Picard** | 随机 PDE 数据生成 + 训练 + 预测（主力脚本） |
| `Baseline_nonlinear_mms.py` | **Route A: MMS** | 人造精确解验证 / 过拟合 debug（辅助脚本） |

### Route B — Picard（主力）

随机采样 f, g, k，用 Picard 迭代（冻结系数线性求解）得到参考解 u_ref，训练 CNN 学习 (f, g, k) → u_ref。

### Route A — MMS（辅助）

构造正的人造精确解 u_exact(x,y)，用离散 FD 算子反算 f，验证残差 ≈ 0，然后在极小数据集上过拟合，用于检查 CNN 实现是否正确。

---

## Quick Start（Picard）

```bash
# 参数预检（不训练）
python Baseline_nonlinear_picard.py --picard-sanity --N 32 --train-samples 16 --k 2

# 训练（固定 k）
python Baseline_nonlinear_picard.py --N 32 --epochs 40 --train-samples 128 --k 2 --retrain --plot

# 训练（随机 k ∈ [1,3]）
python Baseline_nonlinear_picard.py --N 32 --epochs 40 --random-k --k-min 1 --k-max 3 --retrain --plot

# 从 checkpoint 预测
python Baseline_nonlinear_picard.py --predict --predict-from checkpoints/cnn/N32_e40_k2/best.pt --plot

# 小规模调试
python Baseline_nonlinear_picard.py --N 8 --epochs 2 --train-samples 16 --val-samples 4 --k 2 --device cpu --retrain --plot

# 汇编验证（fast vs slow）
python Baseline_nonlinear_picard.py --check-assembly
```

---

## 主要 CLI 参数（Picard）

| 参数 | 默认 | 说明 |
|------|------|------|
| `--N` | 32 | 内部网格点数 |
| `--epochs` | 100 | 训练轮数 |
| `--train-samples` | 128 | 训练样本数 |
| `--batch-size` | 8 | 批大小 |
| `--lr` | 1e-3 | 学习率 |
| `--base-ch` | 32 | 隐藏通道数 |
| `--n-layers` | 6 | 卷积层数 |
| `--k` | 2.0 | 扩散系数指数 |
| `--random-k` | False | 随机采样 k ∈ [k_min, k_max] |
| `--f-scale` | 0.1 | forcing 幅度 |
| `--g-base` | 1.0 | 边界基准值 |
| `--g-scale` | 0.05 | 边界随机波动 |
| `--picard-damping` | 0.5 | Picard 阻尼系数 |
| `--picard-tol` | 1e-8 | Picard 收敛容差 |
| `--lambda-sup` | 1.0 | 监督 loss 权重 |
| `--lambda-pde` | 0.1 | PDE 残差 loss 权重 |
| `--predict-from` | — | 加载 checkpoint 进行预测 |
| `--retrain` | False | 覆盖已有 checkpoint |
| `--picard-sanity` | False | 参数预检后退出 |

---

## 存储结构

```
checkpoints/
├── cnn/N32_e40_k2/                    # 实验运行目录
│   ├── best.pt                        # 最佳模型（val relL2）
│   ├── last.pt                        # 最后一轮模型
│   ├── history.csv                    # 训练曲线
│   ├── convergence.png                # 收敛图
│   └── run_info.txt                   # 运行元数据
├── cnn/latest -> N32_e40_k2           # 指向最近运行
└── registry.json                      # 实验索引（自动维护）

results/
└── cnn/N32_e40_k2/figures/
    ├── history/2026-07-04_113800/     # 旧结果自动归档
    ├── sample_000_panel.png           # 最新结果面板
    └── metrics.csv
```

---

## 评估指标

| Metric | 含义 |
|--------|------|
| relL2 | u_pred vs u_ref 相对 L2 误差（内部点） |
| resRMSE | u_pred 的 PDE 残差 RMS |
| refResRMSE | u_ref 的 PDE 残差 RMS（oracle） |
| linf | 内部节点最大绝对误差 |
| gradRelL2 | 梯度的相对 L2 误差 |
| bmax | 边界最大误差（硬边界下 = 0） |

---

## 关键设计

- **非线性扩散**：系数为 u^k，扩散随解变化；Picard 迭代冻结系数，每步求解线性 Darcy 问题
- **向量化组装**：scipy.sparse.coo_matrix 快速构建 5 点差分矩阵，`--check-assembly` 可验证
- **Hard Dirichlet BC**：`u_pred = bnd * g + (1-bnd) * u_raw`
- **Conservative FD**：交界面上系数取相邻点算术平均 `a_face = 0.5 * (u_c^k + u_neighbor^k)`
- **Hybrid loss**：`loss = λ_sup * L_sup + λ_pde * L_pde`，其中 PDE 残差为 `(h² * R)²` 的均值
- **6 通道输入**：`[f_norm, g, boundary_mask, k_channel, x, y]`，f 和 k 用 z-score 归一化
- **Picard 收敛**：相对变化 < `picard_tol`，最大迭代 `picard_max_iter`，带阻尼
- **预测复用训练分布**：`--predict` 自动继承 checkpoint 的数据生成参数
- **优化器**：AdamW + gradient clipping (max_norm=1.0) + warmup → cosine annealing
- **MMS 用于验证**：人造精确解确保 FD 算子和 Picard 求解器实现正确
