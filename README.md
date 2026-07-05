# Nonlinear Poisson Equation — CNN Solver

求解非线性通量形式 Poisson 方程：

```text
-div(u^k grad u) = f,   u = g on boundary,   Omega = [-1,1]^2
```

扩散系数为 u^k 本身（无外部 a(x,y) 场），u > 0，k 为标量。

---

## 三个独立文件（零交叉 import）

| 文件 | 路线 | 用途 |
|------|------|------|
| `Baseline_solver.py` | **正向 solver 数据** | 随机 f, g, k → 非线性 FD 求解 u_ref → CNN 训练（主力） |
| `Baseline_mms.py` | **MMS 反向数据** | 人造解 u_ref → FD 反算 f（oracle = 0）→ CNN 训练 |
| `PINN.py` | **PINN 对照** | MLP + autodiff，Raissi 式实现，固定 k=2 |

### Baseline_solver.py — 正向数据

随机采样 f, g（二维 Fourier 级数）、k ~ U(0,5) → **非线性有限差分求解器**求 u_ref → CNN 学习 (f, g, k) → u_ref。

非线性 FD 求解器 = **红黑 SOR 逐点松弛**：直接在 5 点 flux-form 差分方程上迭代，每个半扫描用当前迭代值重算 a = clamp(u,ε)^k 并逐点更新 u。纯 stencil 运算，无 Picard 外层线性化、无内层线性方程组求解。不收敛 / u 非正 / 残差过大的样本被拒绝。

### Baseline_mms.py — MMS 反向数据

构造正值 `u_ref = b(x,y) + 0.15*(1-x²)(1-y²)*(poly + Fourier)`（整体抬升至 ≥ 0.3），用同一离散算子反算 f，float64 残差为机器零（oracle = 0）。

---

## Quick Start

```bash
# 数据预检
python Baseline_solver.py --data-sanity --N 32 --train-samples 16
python Baseline_mms.py --mms-sanity --N 32 --train-samples 8

# 训练（physics loss，默认；随机 k ∈ [0,5]，默认）
python Baseline_solver.py --N 32 --epochs 100 --train-samples 128 --retrain --plot
python Baseline_mms.py --N 32 --epochs 200 --train-samples 32 --retrain --plot

# 训练（supervised loss）
python Baseline_solver.py --N 32 --epochs 100 --train-samples 128 --loss supervised --retrain --plot

# 训练（固定 k）
python Baseline_solver.py --N 32 --epochs 100 --no-random-k --k 2 --retrain --plot

# 预测（同分布测试集）
python Baseline_solver.py --predict --predict-from checkpoints/cnn/<key>/best.pt --test-seed 9999 --plot

# 预测（固定测试集；测试 k 默认固定为 2）
# solver 路线：fixed-solver（u_target 只造 f,g，u_ref 由非线性 FD 求解器重算）
python Baseline_solver.py --predict --predict-from checkpoints/cnn/<key>/best.pt --predict-case fixed-solver --plot
# MMS 路线：analytic（manufactured u 直接作 u_ref）
python Baseline_mms.py --predict --predict-from checkpoints/cnn_mms/<key>/best.pt --predict-case analytic --plot

# PINN 对照
python PINN.py --N 32 --epochs 5000
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
| `--k` | 2.0 | 固定 k（需 `--no-random-k`） |
| `--random-k` | True | 随机采样 k ∈ [k_min, k_max]，默认 [0, 5] |
| `--fourier-m` | 6 | 随机场每维 Fourier 模数 |
| `--f-scale` | 5.0 | forcing 幅度（仅 solver） |
| `--g-base` / `--g-scale` | 1.0 / 0.4 | 边界基准值 / 随机波动（仅 solver） |
| `--loss` | physics | 训练目标：`physics`（PDE 残差）或 `supervised`（\|u-u_ref\|²） |
| `--coeff-eps` | 1e-4 | u^k 的 clamp 下界 |
| `--max-fd-iter` | 5000 | SOR 最大扫描数（仅 solver） |
| `--fd-tol` | 1e-6 | 收敛判据：残差 RMSE < tol·max(1, ‖f‖_rms)（仅 solver） |
| `--fd-omega` | 1.5 | SOR 松弛因子，1 = Gauss-Seidel，须 < 2（仅 solver） |
| `--predict-from` | — | 加载 checkpoint 进行预测 |
| `--predict-case` | data | MMS: `data` / `analytic`；solver: `data` / `fixed-solver`（u_target 只造 f,g，u_ref 由非线性 FD 求解器重算，严格 solver 参照，可与 MMS analytic 对照） |
| `--predict-k` | 2.0 | 测试数据固定 k（训练用随机 k） |
| `--test-seed` | 1234 | 预测数据种子（不得与训练 seed 及 seed+1 重合） |
| `--retrain` | False | 覆盖已有 checkpoint |
| `--data-sanity` / `--solver-sanity` / `--mms-sanity` | False | 数据预检后退出 |

---

## 存储结构

```
checkpoints/
├── cnn/N32_e100_tr128_k0-5_M6_fs5_gb1_gs0.4_ch32_L6/     # solver
│   ├── best.pt, last.pt, history.csv, convergence.png, run_info.txt
├── cnn_mms/N32_e200_tr32_k0-5_M6_ch32_L6_mms/            # MMS
│   ├── best.pt, last.pt, history.csv, convergence.png, run_info.txt
├── cnn/latest, cnn_mms/latest                             # 符号链接
├── pinn/N32_e5000_h64_l4_bc50/                            # PINN
└── registry.json                                          # 实验索引（自动维护）

results/
└── cnn/<key>/figures/
    ├── history/2026-07-04_113800/                         # 旧结果自动归档
    ├── sample_000_panel.png, metrics.csv, summary.json
```

---

## 评估指标（逐样本计算再汇总）

| Metric | 含义 |
|--------|------|
| relL2 | u_pred vs u_ref 内点相对 L2 误差（主指标） |
| RMSE / MAE / maxError | 绝对误差统计 |
| predResRMSE / refResRMSE | u_pred / u_ref 的 PDE 残差 RMS（ref = oracle 下限） |
| bndErr | 边界误差（硬投影应为浮点零） |
| gradRelL2 / gradLinf | 梯度相对 L2 / 最大误差 |
| linearBaselineRelL2 | a≡1 线性 Poisson 解的 relL2（**CNN 必须显著打赢**） |
| constBaselineRelL2 | 常数预测基线 relL2 |

---

## 关键设计

- **只用有限差分**：空间离散为 5 点 flux-form 差分（交界面系数取相邻点算术平均）；非线性差分方程用红黑 SOR 逐点松弛求解——无 Picard / Newton / FEM / FNO / 谱方法
- **a≡1 线性基线**：线性 FD 方程组用 `scipy.sparse` 直接求解（仅作平凡基线对照，不参与非线性求解）
- **physics-only 或 supervised**：`--loss` 选训练目标；physics loss 为 mean((h²R)²)
- **Hard Dirichlet BC**：`u_pred = bnd * g + (1-bnd) * u_raw`，边界误差恒为浮点零
- **6 通道输入**：`[f_norm, g, boundary_mask, k_channel, x, y]`，f 和 k 用 z-score 归一化
- **float64 数据管线**：数据生成与参考残差在 float64 计算，训练存 float32；float32 舍入地板单独报告
- **训练随机 k，测试固定 k**：训练 k ∈ U(0,5)，预测/测试统一用 `--predict-k`（默认 2.0）
- **测试集在原始 u^k 意义下**：u1–u4 固定测试函数严格为正，clamp(u,ε)^k 数学上不激活，clamp 仅作训练期数值保护
- **两条测试线可对照**：MMS 的 `analytic`（manufactured u 即 u_ref）与 solver 的 `fixed-solver`（同一组 u_target 只造 f,g，u_ref 由非线性 FD 求解器重算）使用同一组函数；`target_vs_solver_ref_relL2` 检查两条参照是否一致
- **可复现**：全链路 seed、checkpoint 存 args+stats、registry.json、预测 seed 碰撞检查、旧图自动归档
- **优化器**：AdamW + gradient clipping + warmup → cosine annealing
