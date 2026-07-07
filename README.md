# Nonlinear Poisson Equation — CNN Solver

求解非线性扩散方程：

```text
-div(|u|^k grad u) = f,   u = g on boundary,   domain = [-1,1]^2
```

扩散系数 a(u,k) = |u|^k（退化椭圆，u=0 且 k>0 时 a=0），k 为整数标量，u 可为正可为负。

---

## 两个独立文件（零交叉 import）

| 文件 | 路线 | 用途 |
|------|------|------|
| `Baseline_solver.py` | **正向 solver 数据** | 随机 f, g, k → damped Newton FD 求解 u_ref → CNN 训练（主力） |
| `Baseline_mms.py` | **MMS 反向数据** | 人造解 u_ref → FD 反算 f（oracle = 0）→ CNN 训练 |

### Baseline_solver.py — 正向数据

随机采样 f, g（二维 Fourier 级数）、k ~ U(0,5) 整数 → **damped Newton FD solver**（k-continuation）求 u_ref → CNN 学习 (f, g, k) → u_ref。

FD solver：5 点 flux-form 差分，Jacobian 精确组装（scipy.sparse 直接求解），线搜索阻尼 Newton。k-continuation：k=0（线性 Poisson）→ k=1 → ... → k_target 逐步升温，克服 |u|^k 在 u≈0 处的奇异性。

### Baseline_mms.py — MMS 反向数据

构造 signed `u_ref = b(x,y) + 0.15*(1-x²)(1-y²)*(poly + Fourier)`，用同一离散算子反算 f，float64 残差为机器零（oracle = 0）。

---

## 与旧版 (commit 8b1207e) 的关键差异

| | 旧版 | 新版 |
|---|---|---|
| 方程 | -div(u^k grad u) = f | **-div(|u|^k grad u) = f** |
| 扩散系数 | u^k (sign 处理) | **|u|^k** |
| 椭圆性 | k 为奇数时 u<0 则 a<0，非椭圆 | a≥0，退化椭圆 |
| g-base 默认 | 1.0（全正数据） | **0.0（sign-changing 数据）** |
| u_min 过滤 | positive_floor=0.05 | **无过滤** |
| Newton solver | 直接 damped Newton | **k-continuation** |
| 收敛判据 | L2 norm（大 N 不过） | **RMS（N 无关）** |
| k 测试约束 | 测试强制 k=2 | **任意 k≥0** |

---

## Quick Start

```bash
# 数据预检
python Baseline_solver.py --sanity --N 32 --train-samples 16 --device cpu
python Baseline_mms.py --mms-sanity --N 32 --train-samples 16 --device cpu

# 训练（physics loss，默认；随机 k ∈ [0,5]）
python Baseline_solver.py --N 32 --epochs 200 --loss physics --random-k --device cpu
python Baseline_mms.py --N 32 --epochs 200 --loss physics --random-k --device cpu

# 训练（supervised loss）
python Baseline_solver.py --N 32 --epochs 200 --loss supervised --random-k --device cpu
python Baseline_mms.py --N 32 --epochs 200 --loss supervised --random-k --device cpu

# 预测 with 分析测试套件
python Baseline_mms.py --predict \
  --predict-from checkpoints/cnn_mms/<key>/best.pt \
  --predict-case analytic --predict-k 2 --plot --device cpu

python Baseline_solver.py --predict \
  --predict-from checkpoints/cnn/<key>/best.pt \
  --predict-case fixed-solver --predict-k 2 --plot --device cpu
```

---

## 主要 CLI 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--N` | 32 | 内部网格点数 |
| `--epochs` | 100 (solver) / 200 (mms) | 训练轮数 |
| `--train-samples` | 128 (solver) / 32 (mms) | 训练样本数 |
| `--batch-size` | 8 | 批大小 |
| `--lr` | 1e-3 | 学习率 |
| `--base-ch` | 32 | 隐藏通道数 |
| `--n-layers` | 6 | 卷积层数 |
| `--k` | 2 | 固定 k（需 --no-random-k） |
| `--random-k` | True | 随机整数 k ∈ [k_min, k_max]，默认 [0,5] |
| `--k-min` / `--k-max` | 0 / 5 | k 范围（整数） |
| `--fourier-m` | 6 | 随机场每维 Fourier 模数 |
| `--f-scale` | 5.0 | forcing 幅度（仅 solver） |
| `--g-base` / `--g-scale` | 0.0 / 0.4 | 边界 Dirichlet BC（仅 solver） |
| `--loss` | physics | `physics`（PDE 残差）或 `supervised`（|u-u_ref|²） |
| `--newton-max-iter` | 50 | Newton 每阶段最大迭代（仅 solver） |
| `--fd-tol` | 1e-10 | Newton 收敛容差（仅 solver） |
| `--max-attempt-factor` | 20 | 最大尝试次数 = factor × n_samples |
| `--predict-from` | — | 加载 checkpoint 预测 |
| `--predict-case` | data | MMS: `data` / `analytic`；solver: `data` / `fixed-solver` |
| `--predict-k` | 2 | 预测固定 k 值（整数） |
| `--test-seed` | 1234 | 预测数据种子 |
| `--retrain` | False | 覆盖已有 checkpoint |
| `--sanity` / `--mms-sanity` | False | 数据预检后退出 |

---

## 评估指标（逐样本计算再汇总）

| Metric | 含义 |
|--------|------|
| relL2 | u_pred vs u_ref 内点相对 L2 误差（主指标） |
| RMSE / MAE / maxError | 绝对误差统计 |
| predResRMSE / refResRMSE | u_pred / u_ref 的 PDE 残差 RMS |
| bndErr | 边界误差（hard Dirichlet 投影应为浮点零） |
| gradRelL2 / gradLinf | 梯度相对 L2 / 最大误差 |
| linearBaselineRelL2 | a≡1 线性 Poisson 解的 relL2（**CNN 必须显著打赢**） |
| constBaselineRelL2 | 常数预测基线 relL2 |

---

## 关键设计

- **只用有限差分**：空间离散为 5 点 flux-form 差分（算术平均交界面系数）；非线性方程用 damped Newton + k-continuation 求解
- **a≡1 线性基线**：线性 FD 方程组用 `scipy.sparse` 直接求解（平凡基线对照）
- **physics-only 或 supervised**：`--loss` 选训练目标
- **Hard Dirichlet BC**：`u_pred = bnd * g + (1-bnd) * u_raw`，边界误差恒为浮点零
- **6 通道输入**：`[f_norm, g, boundary_mask, k_channel, x, y]`
- **float64 数据管线**：数据生成与参考残差在 float64 计算，训练存 float32
- **训练随机 k，测试固定 k**：训练 k ∈ U{0,1,2,3,4,5}（整数），预测统一用 `--predict-k`
- **5 个 signed 分析测试函数**（u0-u4）：MMS 走 analytic（u_target 即 u_ref），solver 走 fixed-solver（u_target 造 f,g → Newton 重算 u_ref）；`target_vs_solver_ref_relL2` 检查两者一致性
- **优化器**：AdamW + gradient clipping + warmup → cosine annealing
- **checkpoint 类型标记**：`model_type` 字段防止 MMS/solver checkpoint 混用
