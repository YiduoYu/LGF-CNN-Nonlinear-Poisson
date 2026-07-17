# Variable-Coefficient Poisson Equation — UNet & PINN Solvers

求解线性变系数椭圆方程：

```text
-div(a(x,y) grad u) = f,   u = g on boundary,   domain = [-1,1]^2
```

扩散系数 `a(x,y) = exp(sigma * psi(x,y))` 是**外部给定的光滑正场**（log-normal
参数化：`psi` 为归一化到 `[-1,1]` 的随机 2D Fourier 场，`sigma` 控制对比度，
`a ∈ [exp(-sigma), exp(sigma)]`，动态范围 `exp(2*sigma)`）。**`a` 不依赖于解 `u`**，
`u` 可带符号，前向求解是**一次稀疏线性求解**（不再需要 Newton 迭代）。

> 本分支（`a(x,y)`）取代了此前 `-div(u^k grad u) = f`（`a(u,k)=u^k`，`u` 强制为正、
> 需要 damped Newton 求解）的非线性版本。`sigma` 是新的难度轴，取代了原来的整数 `k`。

---

## 三个独立文件（零交叉 import）

| 文件 | 路线 | 用途 |
|------|------|------|
| `UNet-solver.py` | **正向 solver 数据** | 随机 f, g, a(sigma) → 单次线性 FD 求解 u_ref → HourglassUNet 训练（主力） |
| `UNet-mms.py` | **MMS 反向数据** | 人造解 u_ref + 随机 a(sigma) → FD 反算 f（oracle = 0）→ HourglassUNet 训练 |
| `PINN.py` | **PINN 对照** | u1-u4 解析解 + a(sigma)，mesh-free 训练，autodiff 求 `-div(a grad u)` 残差 |

模型架构统一是 `HourglassUNet`：通道走向 8C-4C-2C-C-2C-4C-8C（沙漏形，瓶颈最窄，
`C = --base-ch`，默认 32 即 256-128-64-32-64-128-256），~4.1M 参数。此前的
`SimpleCNN`（固定膨胀卷积、无下采样、~48K 参数）已从两个文件中移除；历史训练结果
仍保留在 `checkpoints/cnn*/`、`results/axy_sig0.5-1.5/` 下，不受影响。

### UNet-solver.py — 正向数据

随机采样 f, g（二维 Fourier 级数）、a(x,y)（`sigma ~ U(sigma_min, sigma_max)` 或固定
`sigma`）→ 直接稀疏线性求解 `-div(a grad u)=f` → HourglassUNet 学习 `(f, g, a)` → `u_ref`。

FD 离散：5 点 flux-form 差分，界面系数算术平均；`scipy.sparse.linalg.spsolve`
一次求解，无需迭代、无拒绝采样。

### UNet-mms.py — MMS 反向数据

构造带符号 `u_ref = b(x,y) + 0.36*(1-x²)(1-y²)*Fourier` 与独立随机 `a(x,y)`，用同一
离散算子反算 `f = -div_h(a grad_h u_ref)`，float64 残差为机器零（oracle = 0）。

### PINN.py — 物理信息神经网络对照

标准 Raissi et al. (2017) 设计：MLP（Tanh + Xavier init）+ Adam + L-BFGS 微调，边界
条件用惩罚项软约束。PDE 残差通过**两次嵌套 autograd**计算 `-div(a grad u)`
（`q=a*grad(u)` 再对 `q` 求散度），链式法则自动展开出 `grad(a)·grad(u)` 项，无需手推
公式。mesh-free：训练不依赖网格，`a(x,y)` 用与 UNet 相同的 Fourier 系数、以解析公式
在任意配点求值（含 autodiff），因此**一个训好的模型可在任意 `--N` 下重新评估，无需
重新训练**（checkpoint key 不含 N）。测试目标为 `UNet-mms.py` 的 u1-u4 解析解。

---

## Quick Start

```bash
# 数据预检
python UNet-solver.py --sanity --N 32 --train-samples 16 --device cpu
python UNet-mms.py --mms-sanity --N 32 --train-samples 16 --device cpu

# 训练（physics loss；固定 sigma=0.5）
python UNet-solver.py --N 32 --epochs 200 --loss physics --sigma 0.5 --device cpu
python UNet-mms.py --N 32 --epochs 200 --loss physics --sigma 0.5 --device cpu

# 训练（随机 sigma ∈ [0.5, 1.5]，一次训练覆盖整个难度区间）
python UNet-solver.py --N 32 --epochs 200 --loss physics \
  --random-sigma --sigma-min 0.5 --sigma-max 1.5 --device cpu

# 训练（supervised loss）
python UNet-solver.py --N 32 --epochs 200 --loss supervised --sigma 0.5 --device cpu
python UNet-mms.py --N 32 --epochs 200 --loss supervised --sigma 0.5 --device cpu

# 预测 with 分析测试套件
python UNet-mms.py --predict \
  --predict-from checkpoints/cnn_mms/<key>/best.pt \
  --predict-case analytic --predict-sigma 0.5 --plot --device cpu

python UNet-solver.py --predict \
  --predict-from checkpoints/cnn/<key>/best.pt \
  --predict-case fixed-solver --predict-sigma 0.5 --plot --device cpu

# PINN：训练 u1，评估在 N=128（论文标准超参数：epochs=5000, Adam+L-BFGS）
python PINN.py --case u1 --sigma 0.5 --N 128 --retrain --device cpu

# 同一 checkpoint 换个 N 重新评估，无需重训
python PINN.py --case u1 --sigma 0.5 --N 32 --device cpu
```

---

## 主要 CLI 参数

### UNet-solver.py / UNet-mms.py（共有）

| 参数 | 默认 | 说明 |
|------|------|------|
| `--N` | 32 | 内部网格点数 |
| `--epochs` | 100 (solver) / 200 (mms) | 训练轮数 |
| `--train-samples` | 128 (solver) / 32 (mms) | 训练样本数 |
| `--batch-size` | 8 | 批大小 |
| `--lr` | 1e-3 | 学习率 |
| `--base-ch` | 32 | HourglassUNet 基宽 C（通道 8C-4C-2C-C-2C-4C-8C，瓶颈最窄） |
| `--sigma` | 0.5 | 固定 sigma（需 `--no-random-sigma`，默认已是固定） |
| `--random-sigma` | False | 随机 sigma ∈ [sigma_min, sigma_max] |
| `--sigma-min` / `--sigma-max` | 0.5 / 1.5 | sigma 范围 |
| `--fourier-m` | 6 | 随机场每维 Fourier 模数 |
| `--f-scale` | 6.0 | forcing 幅度（仅 solver） |
| `--g-base` / `--g-scale` | 0.0 / 0.6 | 边界 Dirichlet BC（仅 solver，u 可带符号，无需抬高边界） |
| `--loss` | physics | `physics`（PDE 残差）或 `supervised`（\|u-u_ref\|²） |
| `--predict-from` | — | 加载 checkpoint 预测 |
| `--predict-case` | data | MMS: `data` / `analytic`；solver: `data` / `fixed-solver` |
| `--predict-sigma` | 0.5 | 预测用固定 sigma |
| `--test-seed` | 9999 | 预测数据种子 |
| `--retrain` | False | 覆盖已有 checkpoint |
| `--sanity` / `--mms-sanity` | False | 数据预检后退出 |

### PINN.py

| 参数 | 默认 | 说明 |
|------|------|------|
| `--case` | u1 | 解析测试目标 u1-u4 |
| `--sigma` | 0.5 | a(x,y) 对比度 |
| `--fourier-m` | 6 | a(x,y) 的 Fourier 模数 |
| `--N` | 32 | **仅用于评估网格**，训练 mesh-free，不影响训练 |
| `--epochs` | 5000 | Adam 训练轮数 |
| `--n-int` / `--n-bnd` | 1024 / 256 | 每轮内部/边界配点数 |
| `--lr` | 1e-3 | 学习率 |
| `--bc-weight` | 50.0 | 边界惩罚项权重 |
| `--hidden-dim` / `--n-layers` | 64 / 4 | MLP 宽度/深度 |
| `--lbfgs-epochs` | 100 | L-BFGS 微调步数（0 跳过） |
| `--retrain` | False | 覆盖已有 checkpoint（checkpoint key 不含 N） |

---

## 评估指标（逐样本计算再汇总，三份代码一致）

| Metric | 含义 |
|--------|------|
| relL2 | u_pred vs u_ref 内点相对 L2 误差（主指标） |
| RMSE / MAE / maxError | 绝对误差统计 |
| predResRMSE / refResRMSE | u_pred / u_ref 的 PDE 残差 RMS |
| bndErr | 边界误差（hard Dirichlet 投影应为浮点零） |
| gradRelL2 / gradLinf | 梯度相对 L2 / 最大误差 |
| linearBaselineRelL2 | a≡1 线性 Poisson 解的 relL2（**UNet/PINN 必须显著打赢**） |
| constBaselineRelL2 | 常数预测基线 relL2 |
| f_RMSE | forcing 项 f 的 RMS（尺度参考，UNet 脚本） |

---

## 关键设计

- **线性变系数椭圆问题**：`a(x,y)` 外部给定、与 `u` 无关，前向求解退化为单次稀疏线性
  求解（`scipy.sparse.linalg.spsolve`），无 Newton 迭代、无拒绝采样
- **u 可带符号**：`enforce_dirichlet` 为纯投影 `u = bnd*g + (1-bnd)*u_raw`，不再需要
  softplus / 正性下限
- **a≡1 线性基线**：线性 FD 方程组用 `scipy.sparse` 直接求解（平凡基线对照）
- **physics-only 或 supervised**：`--loss` 选训练目标
- **6 通道输入**（UNet）：`[f_norm, g, boundary_mask, log(a)_norm, x, y]`，`a` 是逐点
  场（不是标量广播）
- **float64 数据管线**：数据生成与参考残差在 float64 计算，训练存 float32
- **难度轴 sigma**：`a=exp(sigma*psi)`，`sigma=0` 退化为常系数 Poisson（sanity
  check）；`sigma` 越大系数对比度 `exp(2*sigma)` 越大，问题越难
- **4 个分析测试函数**（u1-u4，直接用带符号原始公式，不再平移到正区间）：MMS 走
  analytic（u_target 即 u_ref），solver 走 fixed-solver（u_target 造 f,g → 线性重解
  u_ref）；`target_vs_solver_ref_relL2` 检查两者一致性（理论上应为机器精度）
- **PINN 的 flux-form 残差**：`-div(a grad u)` 通过两次嵌套 `torch.autograd.grad`
  计算（先算 `q=a*grad(u)`，再对 `q` 求散度），`grad(a)·grad(u)` 项通过链式法则自动
  展开，无需为每个 `a(x,y)` 手推残差公式
- **PINN mesh-free 复用**：checkpoint key 不含评估网格 `N`，同一模型可在任意 `--N`
  下重新评估而不重训
- **优化器**：UNet 用 AdamW + gradient clipping + warmup → cosine annealing；PINN 用
  Adam + L-BFGS 微调（Raissi et al. 2017 标准做法）
- **checkpoint 类型标记**：`model_type` 字段防止 MMS/solver checkpoint 混用；两个
  UNet 脚本还会拒绝加载重构前（`a(u,k)=u^k`）的旧 checkpoint（缺少 `"sigma"`
  字段会直接报错，避免静默产出错误预测）
