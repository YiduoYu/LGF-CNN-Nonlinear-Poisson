# Nonlinear Poisson — CNN Baseline vs PINN

求解同一道非线性 Poisson 方程的两种方法对比：

```text
-div(u^2 grad u) = f,   u = g on boundary,   Ω = [-1,1]^2
制造解: u_exact = sin(πx)·cos(πy)
```

| | Baseline.py (CNN) | PINN.py (MLP) |
|---|---|---|
| 模型 | U-Net, grid→grid | MLP, (x,y)→u |
| 导数 | 有限差分 (O(h²)) | autodiff (精确) |
| 边界 | 硬约束（直接替换） | 软约束（罚函数） |
| 训练信号 | PDE 残差（无标签） | PDE 残差 + 边界罚函数 |
| 训练策略 | AdamW + warmup + cosine | Adam + L-BFGS (Raissi 2017) |

---

## Quick Start

```bash
# CNN demo（无参数：自动训练 40 epoch + 预测）
python Baseline.py

# 再次运行 → 自动跳过训练，直接预测
python Baseline.py

# 强制重新训练
python Baseline.py --retrain

# PINN 对比
python PINN.py --epochs 5000
```

---

## 存储结构

采用 **config-key** 命名——同一配置永远写入同一目录，自动覆盖，自动跳过训练。

```
checkpoints/
├── cnn/N32_e40_k2/           # Baseline, N=32, epochs=40, k=2.0
│   ├── best.pt               # 验证集 relL2 最优
│   ├── last.pt               # 最后一个 epoch
│   ├── history.csv           # 训练历史
│   ├── convergence.png       # 收敛曲线
│   └── run_info.txt
├── pinn/N32_e5000_h64_l4_bc50/  # PINN, N=32, hidden=64, layers=4, bc_weight=50
│   ├── best.pt / last.pt / history.csv / run_info.txt
├── cnn/latest -> N32_e40_k2
├── pinn/latest -> N32_e5000_h64_l4_bc50
└── registry.json             # 所有 run 的轻量索引

results/
├── cnn/N32_e40_k2/figures/   # CNN 预测面板 + metrics.csv
└── pinn/N32_e5000_h64_l4_bc50/figures/  # PINN 预测面板 + metrics.csv
```

**Config-key 命名规则**：
- CNN: `N{网格}_{epochs}e_k{k}`
- PINN: `N{网格}_e{epochs}_h{hidden}_l{n_layers}_bc{bc_weight}`

---

## Baseline.py 实现逻辑

### 1. 数学问题

$$
-\nabla \cdot \big(u^2 \nabla u\big) = f, \quad \Omega = [-1,1]^2, \quad u|_{\partial\Omega} = g
$$

- 非线性来源：扩散系数 $a(u) = u^2$，依赖于解本身
- 制造解：$u_{\text{exact}} = \sin(\pi x)\cos(\pi y)$，代入 PDE 反推出源项 $f$

### 2. 数据流

```
制造解 u = sin(πx)cos(πy)
    ↓ 解析求导
源项 f = -div(u²∇u)          边界值 g = u|边界
    ↓                            ↓
    └────── 6通道输入 ───────────┘
    [f, g, boundary_mask, k, x, y]
    ↓
    U-Net
    ↓
    u_raw（CNN 原始输出）
    ↓ 硬边界约束
    u_pred = boundary_mask·g + interior_mask·u_raw
    ↓
    有限差分计算 PDE 残差
    R = -div_h(u_pred² ∇_h u_pred) - f
    ↓
    loss = mean((R / RMS(f))²)
    ↓
    反向传播，更新 U-Net 参数
```

**关键设计：u_exact 全程不参与训练**，只用于生成 f、g 和最终评估。

### 3. 逐模块解析

#### 网格与输入

`make_grid(N)` 生成 $(N+2) \times (N+2)$ 的均匀网格（包含边界节点），返回坐标张量和 boundary/interior mask。

`build_input()` 拼接 6 通道输入 `[B, 6, H, W]`：

| 通道 | 含义 | 作用 |
|------|------|------|
| f | 归一化源项 | "PDE 右端项是什么" |
| g | 边界值 | "边界上 u 等于什么" |
| boundary_mask | 边界标记 (0/1) | "哪些点是边界" |
| k | 非线性指数 | "a(u)=u^k 中的 k" |
| x, y | 坐标网格 | "每个网格点的位置" |

`enforce_dirichlet()` 实现硬边界约束：
```python
u_final = boundary_mask * g + (1 - boundary_mask) * u_raw
```
边界点直接用 g 替换，内部点保留 CNN 输出。边界条件**精确满足**，不是罚函数近似。

#### PDE 残差（通量守恒有限差分）

对每个内部网格点 (i,j)，用中心差分计算散度：

```
东面通量: a_{i+1/2,j} · (u_{i+1,j} - u_{i,j}) / h
西面通量: a_{i-1/2,j} · (u_{i,j} - u_{i-1,j}) / h
北面通量: a_{i,j+1/2} · (u_{i,j+1} - u_{i,j}) / h
南面通量: a_{i,j-1/2} · (u_{i,j} - u_{i,j-1}) / h

散度 = (东 - 西 + 北 - 南) / h
R = -div_h - f
```

交界面的扩散系数用**算术平均**：$a_{i+1/2} = 0.5 \cdot (a_i + a_{i+1})$。这保证相邻控制体共享同一个通量，格式严格守恒。

`power_coefficient()` 计算 $a(u) = u^k$——对整数 k（如 k=2）不 clamp 负值；对非整数 k 才 clamp 防止 NaN。`_safe_pow()` 同时返回 $u^k$ 和 $k \cdot u^{k-1}$，`forcing()` 和 `flux_divergence()` 共用。

#### U-Net 结构

```
输入 [B, 6, H, W]
  ↓ enc1: ConvBlock(6 → 32)
  ↓ avg_pool → enc2: ConvBlock(32 → 64)
  ↓ avg_pool → enc3: ConvBlock(64 → 128)
  ↓ avg_pool → bottleneck: ConvBlock + DilatedBlock×2 (dilation=2,4)
  ↓ 上采样 + skip connection → dec3
  ↓ 上采样 + skip connection → dec2
  ↓ 上采样 + skip connection → dec1
  ↓ 1×1 conv
输出 [B, 1, H, W]
```

- `GroupNorm`（8 groups）替代 BatchNorm——小 batch 更稳定
- `GELU` 激活函数
- 空洞卷积（dilation=2,4）扩大感受野
- padding 用 `replicate` 模式——比 zero-padding 更符合物理边界
- 上采样用双线性插值

#### 训练循环

1. **生成数据**：`generate_dataset` 生成训练/验证集
2. **归一化**：`compute_stats` 计算 f 和 k 的均值/标准差，输入归一化
3. **Oracle floor**：计算精确解的 PDE 残差 RMS——有限差分离散精度的理论上限
4. **训练**：AdamW + warmup (前 5% steps) + cosine annealing
5. **保存**：验证 relL2 最优的 `best.pt` + 最后 epoch 的 `last.pt` + 训练历史 CSV + 收敛曲线
6. **跳过训练**：同一 config 再次运行时，自动检测已有 `best.pt` 并跳过训练。使用 `--retrain` 强制重训

#### 评估指标

| 指标 | 含义 | 依赖 u_exact |
|------|------|:---:|
| relL2 | 预测解 vs 精确解的相对 L2 误差 | ✓ |
| linf | 内部节点最大绝对误差 | ✓ |
| resRMSE | PDE 残差的 RMS（训练目标） | ✗ |
| resLinf | PDE 残差的最大绝对值 | ✗ |
| bmax | 边界最大误差 | ✓ |
| gradRelL2 | 梯度的相对 L2 误差 | ✓ |

### 4. 为什么这个设计有效

1. **无标签训练**：loss 完全来自物理规律，不需要数值解作为标签
2. **守恒格式**：通量在交界面上共享，保证物理一致性
3. **硬边界**：边界条件精确满足，避免了 PINN 罚函数法中 PDE 和边界条件的 trade-off
4. **全网格监督**：每个内部节点都参与 loss，不像 PINN 只能采样有限点

### 5. CNN vs PINN — 关键差异

在 N=32 网格上，CNN 的 relL2 (0.316) 远高于 PINN (0.004)。根本原因：

- **CNN 用有限差分**：训练信号 `-div_h(u²∇_h u) - f` 本身有 O(h²) 截断误差。CNN 学到了离散方程的精确解，而非连续方程的精确解
- **PINN 用 autodiff**：导数精确，学到的是真实 PDE 的解

解决方案：加密网格 (N=64+)，FD 截断误差 ∝ 1/N²。

---

## PINN.py 说明

PINN 遵循 Raissi et al. (2017) 标准设计：
- MLP (2→64→...→64→1)，Tanh 激活，Xavier 初始化
- `torch.autograd.grad` 计算精确导数
- 边界条件通过罚函数 `(u - g)²` 施加，`--bc-weight` 控制强度（默认 50）
- Adam 预训练 + L-BFGS 微调（`--lbfgs-epochs` 控制，默认 100）
- 每 epoch 随机重采样 collocation points
- 评估时使用与 CNN 相同的 FD residual（`fd_residual`），保证指标可比
- 输出 `results/pinn/*/figures/pinn_panel.png`（与 CNN 相同的 2×3 布局，方便直接对比）

---

## 自定义训练

```bash
# CNN — 不同网格和 epoch
python Baseline.py --N 64 --epochs 80

# CNN — 强制重新训练
python Baseline.py --N 32 --epochs 40 --retrain

# CNN — 只预测（不检查/训练）
python Baseline.py --load-path checkpoints/cnn/N32_e40_k2/best.pt --predict --plot

# PINN — 更强的边界约束
python PINN.py --N 32 --epochs 5000 --bc-weight 100

# PINN — 跳过 L-BFGS
python PINN.py --N 32 --epochs 5000 --lbfgs-epochs 0
```

---

## 输出文件总览

| 文件 | 来源 | 说明 |
|------|------|------|
| `checkpoints/cnn/{config}/convergence.png` | Baseline | CNN 收敛曲线 |
| `checkpoints/cnn/{config}/history.csv` | Baseline | CNN 训练历史 |
| `checkpoints/cnn/{config}/best.pt` | Baseline | CNN 最优模型 |
| `checkpoints/pinn/{config}/history.csv` | PINN | PINN 训练历史 |
| `checkpoints/pinn/{config}/best.pt` | PINN | PINN 最优模型 |
| `results/cnn/{config}/figures/*.png` | Baseline | CNN 预测面板 (2×3) |
| `results/pinn/{config}/figures/pinn_panel.png` | PINN | PINN 预测面板 (2×3) |
| `checkpoints/registry.json` | Baseline + PINN | 所有 run 的轻量索引 |
