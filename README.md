# Variable-Coefficient Poisson Equation — CNN vs LGF-Unrolled Solver

求解线性变系数椭圆方程：

```text
-div(a(x,y) grad u) = f,   u = g on boundary,   domain = [-1,1]^2
```

扩散系数 `a(x,y) = exp(sigma * psi(x,y))` 是**外部给定的光滑正场**（log-normal
参数化：`psi` 为归一化到 `[-1,1]` 的随机 2D Fourier 场，`sigma` 控制对比度，
`a ∈ [exp(-sigma), exp(sigma)]`，动态范围 `exp(2*sigma)`）。**`a` 不依赖于解 `u`**，
`u` 可带符号，前向求解是**一次稀疏线性求解**（不再需要 Newton 迭代）。

本分支（`LGF+CNN`）在 `a(x,y)` 分支的基础上，给 `Baseline_mms.py` /
`Baseline_solver.py` 两条数据管线各加了一个新模型 **`LGFNet`**，跟原有的
`SimpleCNN` 做同预算对比（同网格、同 batch、同 epoch、同 loss）。

---

## 文件

| 文件 | 角色 |
|------|------|
| `Baseline_core.py` | **共享核心**：网格/flux-form 差分算子、稀疏 FD 求解器与线性基线、`SimpleCNN` 和 `LGFNet` 两个模型、归一化/batching/forward 辅助函数、训练循环辅助函数（`create_model`/`create_optimizer`/`train_one_epoch`/...）、逐样本指标、通用绘图。**无副作用**（import 不解析参数、不生成数据、不建目录、不设随机种子）|
| `Baseline_mms.py` | 入口脚本：MMS 反向数据管线（人造解 `u_ref` + 随机 `a(sigma)` → FD 反算 `f`，oracle 残差 = 0）、u1-u4 解析测试套件、`checkpoints/cnn_mms/` 布局 |
| `Baseline_solver.py` | 入口脚本：正向 solver 数据管线（随机 `f,g,a(sigma)` → 稀疏线性直接求解 `u_ref`）、fixed-solver 离散一致性测试、`checkpoints/cnn/` 布局 |

两个入口脚本的 CLI、训练循环、评估指标**完全一致**，只有各自的数据生成方式和
checkpoint 目录不同；模型本身（`SimpleCNN` / `LGFNet`）在 `Baseline_core.py`
里只有一份实现，两条管线共用，不会出现两份互相漂移的模型代码。

---

## 两个模型

### `SimpleCNN` —— 一次性映射

固定膨胀卷积（无下采样），`n_layers` 层，通道数 `hidden_ch`（默认 32、6 层
≈ 48K 参数）。输入 6 通道 `[f_norm, g, boundary_mask, log(a)_norm, x, y]`，
一次前向直接输出 `u_pred`。

### `LGFNet` —— 3×3 LGF 核展开迭代

不是一次性映射，而是把**离散 Green 函数（LGF）**的物理先验编码进一个展开
`T` 步的残差迭代里：

```text
u_0 = 边界投影(0)
for t in 0..T-1:
    r_t = f - A_h(a) u_t                     # PDE 残差（内点）
    q_t = r_t / a                            # 幅值预条件
    X_t = [u_t, q_t_norm, f_norm, g, bnd, log(a)_norm, x, y, t/T]   # 9 通道
    du  = Phi_theta(X_t)                     # 1x1 lift -> [多核LGF block]^L -> 1x1 head
    u_{t+1} = 边界投影(u_t + eta_t * du)
return u_T
```

- **固定的 3×3 核**：`build_lgf_kernel_bank()` 用 `LGF_python.lgf.LGF_FFT` 算出
  `M` 个节点（`a_m` 在 `[exp(-1.5), exp(1.5)]` 上 log-等距）对应的、修正 Helmholtz
  算子 `P_h(kappa)=-Delta_h+kappa^2 I` 的 3×3 stencil，按 `(N, M, N_fft,
  LGF_LOG_A_RANGE, h)` 缓存到 `checkpoints/lgf_bank/*.npz`，**不可学习**、
  训练前后逐位不变（有 assert 保证）。
- **多核插值**：每个格点按当地系数 `a_ij` 在两个最近节点间做 log-空间线性插值
  （`_select_kernel_weights`），一批只算一次、所有 T 步复用。
- **唯一可学习部分**：`Phi_theta` 里的 1×1 卷积 / GroupNorm / 每步一个的
  `eta_t`（初值 1.0）——**没有任何可学习的 3×3 空间卷积**。`Phi_theta` 的
  `head` 层做零初始化，训练一开始 `du≡0`（展开迭代退化为恒等映射），避免
  `u_t` 原始未归一化地逐步反馈造成的指数级发散。
- **代价**：T 步展开是天然串行的（`u_{t+1}` 依赖 `u_t`），参数量虽然只有
  `SimpleCNN` 的约 1/10（默认配置 ~4.8K），但总计算量和串行深度更高，训练
  墙钟时间反而更长（详见下方"已知结果"）。

---

## Quick Start

```bash
# 数据 + 核函数库正确性预检
python Baseline_mms.py --mms-sanity --model cnn --N 32 --train-samples 16
python Baseline_mms.py --mms-sanity --model lgf --N 32 --train-samples 16 --lgf-M 8
python Baseline_solver.py --sanity --model lgf --N 32 --train-samples 8

# 训练 SimpleCNN（supervised）
python Baseline_mms.py --model cnn --N 64 --loss supervised \
  --train-samples 2000 --val-samples 128 --epochs 300 --seed 1234

# 训练 LGFNet，同预算同种子（同预算对比的关键）
python Baseline_mms.py --model lgf --N 64 --loss supervised \
  --train-samples 2000 --val-samples 128 --epochs 300 --seed 1234 \
  --lgf-T 12 --lgf-M 8 --lgf-L 4

# 随机 sigma ∈ [0.5, 1.5]，一次训练覆盖整个难度区间
python Baseline_mms.py --model lgf --N 64 --loss supervised \
  --random-sigma --sigma-min 0.5 --sigma-max 1.5 \
  --train-samples 2000 --val-samples 128 --epochs 300 --seed 1234

# 预测：同分布（in-dist，固定 sigma）+ 解析测试套件（分布外泛化）
python Baseline_mms.py --predict --predict-from checkpoints/cnn_mms/<key>/best.pt \
  --predict-samples 32 --predict-sigma 0.5 --plot-dir results/<...>/indist_sig0.5
python Baseline_mms.py --predict --predict-from checkpoints/cnn_mms/<key>/best.pt \
  --predict-case analytic --predict-sigma 0.5 --plot --plot-dir results/<...>/analytic_sig0.5

# solver 管线同理，checkpoint 在 checkpoints/cnn/，predict-case 换成 fixed-solver
python Baseline_solver.py --predict --predict-from checkpoints/cnn/<key>/best.pt \
  --predict-case fixed-solver --predict-sigma 0.5 --plot
```

---

## 主要 CLI 参数（`Baseline_mms.py` / `Baseline_solver.py` 共有）

| 参数 | 默认 | 说明 |
|------|------|------|
| `--N` | 32 | 内部网格点数 |
| `--epochs` | 200 (mms) / 100 (solver) | 训练轮数 |
| `--train-samples` | 32 (mms) / 128 (solver) | 训练样本数 |
| `--val-samples` | 8 (mms) / 32 (solver) | 验证样本数 |
| `--batch-size` | 8 | 批大小 |
| `--lr` | 1e-3 | 学习率 |
| `--base-ch` | 32 | `SimpleCNN` 通道数 / `LGFNet` 的 `Phi_theta` 隐藏通道数 |
| `--model` | cnn | `cnn`（SimpleCNN 一次性映射）或 `lgf`（LGFNet 展开迭代） |
| `--lgf-T` | 12 | LGFNet 展开迭代步数 |
| `--lgf-M` | 8 | LGFNet 核函数库节点数 |
| `--lgf-L` | 4 | LGFNet 每步 `Phi_theta` 里多核 block 的层数 |
| `--n-layers` | 6 | SimpleCNN 卷积层数（LGFNet 不用这个参数） |
| `--sigma` | 0.5 | 固定 sigma |
| `--random-sigma` | False | 随机 sigma ∈ [sigma_min, sigma_max] |
| `--sigma-min` / `--sigma-max` | 0.5 / 1.5 | sigma 范围 |
| `--fourier-m` | 6 | 随机场每维 Fourier 模数 |
| `--loss` | physics | `physics`（PDE 残差）或 `supervised`（\|u-u_ref\|²） |
| `--predict-from` | — | 加载 checkpoint 预测 |
| `--predict-case` | data | mms: `data`/`analytic`；solver: `data`/`fixed-solver` |
| `--predict-sigma` | 0.5 | 预测用固定 sigma |
| `--test-seed` | 9999 | 预测数据种子 |
| `--retrain` | False | 覆盖已有 checkpoint |
| `--mms-sanity`（mms）/ `--sanity`（solver） | False | 数据 + LGF 核函数库正确性预检后退出 |

---

## 评估指标（逐样本计算再汇总，两条管线一致）

| Metric | 含义 |
|--------|------|
| relL2 | u_pred vs u_ref 内点相对 L2 误差（主指标） |
| RMSE / MAE / maxError | 绝对误差统计 |
| predResRMSE / refResRMSE | u_pred / u_ref 的 PDE 残差 RMS |
| bndErr | 边界误差（hard Dirichlet 投影应为浮点零） |
| gradRelL2 / gradLinf | 梯度相对 L2 / 最大误差 |
| linearBaselineRelL2 | a≡1 线性 Poisson 解的 relL2（模型必须显著打赢） |
| constBaselineRelL2 | 常数预测基线 relL2 |
| f_RMSE | forcing 项 f 的 RMS（尺度参考） |
| LGFNet per-step residual RMS | 仅 LGFNet：`evaluate()`/预测时打印 T 步展开过程中残差 RMS 的逐步轨迹（诊断用，训练时默认不计算，见下） |

---

## 关键设计

- **线性变系数椭圆问题**：`a(x,y)` 外部给定、与 `u` 无关，前向求解退化为单次稀疏
  线性求解，无 Newton 迭代、无拒绝采样
- **u 可带符号**：`enforce_dirichlet` 为纯投影 `u = bnd*g + (1-bnd)*u_raw`
- **LGF 核函数库缓存**：按 `(N, M, N_fft, LGF_LOG_A_RANGE, h)` 生成 `.npz` 缓存
  key，任何一个变化都不会静默复用旧缓存；核心正确性有两条 assert/检查：核函数库
  全部有限值且训练前后逐位不变、常数系数场下多核插值退化为单核响应（`--mms-sanity
  --model lgf` / `--sanity --model lgf` 会跑这两项）
- **LGF 诊断按需计算**：`LGFNet.forward(collect_diagnostics=...)` 默认不算逐步
  残差 trace（训练时的默认路径），只有 `evaluate()`/预测显式要看 trace 才算，且
  trace 全程留在 device tensor 上、循环结束后才做一次 `.cpu()` 转换，避免污染
  计时区间
- **checkpoint key 防碰撞**：`ckpt_key()` 在可读的超参数字符串后面加一段
  `seed/lr/weight_decay/batch_size/val_samples/eval_every/grad_clip` 的 md5
  短哈希，这些参数变了不会误用别的 run 的 checkpoint 目录；`model_type` 字段
  防止 mms/solver checkpoint 混用
- **训练/评估共享的辅助函数**：`create_model`/`create_optimizer`/`compute_loss`/
  `train_one_epoch`/`validate_and_checkpoint` 把训练循环拆成职责单一的小函数，
  `--model cnn`/`--model lgf` 都走同一套；linear baseline 只在验证/预测/解析
  测试第一次需要时惰性计算并缓存（`ensure_linear_baselines`），训练集从不计算
- **float64 数据管线**：数据生成与参考残差在 float64 计算，训练存 float32
- **难度轴 sigma**：`sigma=0` 退化为常系数 Poisson（sanity check）；`sigma` 越大
  系数对比度 `exp(2*sigma)` 越大，问题越难

---

## 已知结果（N=64, supervised, train=2000, val=128, epochs=300, seed=1234）

| | SimpleCNN | LGFNet (T=12, M=8, L=4) |
|---|---|---|
| 参数量 | 48,417 | 4,845（约 1/10） |
| 验证集 relL2 | 1.44e-02 | **5.77e-03** |
| In-dist relL2（sigma=0.5/1/1.5） | 9.74e-03 / 1.34e-02 / 1.99e-02 | **4.38e-03 / 5.24e-03 / 9.09e-03** |
| Analytic 泛化 relL2（u1-u3，跨 sigma 大致范围） | ~0.76-0.98 | **~0.20-0.24** |
| Analytic 泛化 relL2（u4，两者共同难点） | ~0.667 | ~0.47 |
| 训练墙钟时间（这台机器，MPS 后端） | ~42 分钟 | ~9 小时（T=12 步展开是天然串行的，见上方"两个模型"） |

LGFNet 用约 1/10 的参数量，在同分布数据（验证集、in-dist）和分布外解析函数
泛化测试上都明显优于 SimpleCNN——尤其在 analytic 泛化上，SimpleCNN 几乎失效
（u1-u3 相对误差 76%-98%），LGFNet 明显更稳（20%-24%），说明 3×3 LGF 物理先验
确实带来了更好的外推能力；代价是训练时间显著更长。u4 是两者共同的难点。
