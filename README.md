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
| 训练集 | 固定网格，相同 | 固定网格，相同 |
| 训练信号 | PDE 残差（无标签） | PDE 残差 + 边界罚函数 |

---

## Quick Start

```bash
# CNN baseline（无参数默认 demo：训练 40 epoch + 预测）
python3 Baseline.py

# 再次运行跳过训练，直接预测
python3 Baseline.py

# PINN 对比
python3 PINN.py --epochs 5000
```

自定义训练：

```bash
python3 Baseline.py --N 32 --epochs 200 --batch-size 8 --save-path checkpoints/unet_n32.pt
python3 Baseline.py --load-path checkpoints/unet_n32.pt --predict --plot
```

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

`power_coefficient()` 计算 $a(u) = u^k$——对整数 k（如 k=2）不 clamp 负值；对非整数 k 才 clamp 防止 NaN。

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

#### 制造解与数据集

`u_exact(x,y) = sin(πx)·cos(πy)` — 固定的解析解。

`forcing()` 用解析求导计算源项：

```
u_x = π·cos(πx)·cos(πy)
u_y = -π·sin(πx)·sin(πy)
laplacian = -2π²·u
grad_sq = u_x² + u_y²

f = -(2u·grad_sq + u²·laplacian)
```

`generate_dataset()` 生成 batch 数据。当前所有样本是同一道 PDE 的拷贝——这是一个单题 demo，目标是验证"CNN 能否只用 PDE 残差学会求解"，而非训练通用 PDE solver。

#### 训练循环

1. **生成数据**：`generate_dataset` 生成训练/验证集
2. **归一化**：`compute_stats` 计算 f 和 k 的均值/标准差，输入归一化
3. **Oracle floor**：计算精确解的 PDE 残差 RMS——这是有限差分离散精度的理论上限，任何方法都无法突破
4. **训练**：
   - 每个 epoch shuffle 数据
   - 每个 batch：前向 → 相对 PDE 残差 loss → 反向
   - loss = mean( (R / RMS(f))² )，除以 RMS(f) 使 loss 无量纲
   - 梯度裁剪 (max norm = 1.0)
5. **学习率**：warmup（前 5%）+ cosine annealing
6. **保存**：验证 relL2 最优的 checkpoint + 训练历史 CSV + 收敛曲线

#### 评估指标

| 指标 | 含义 | 依赖 u_exact |
|------|------|:---:|
| relL2 | 预测解 vs 精确解的相对 L2 误差 | ✓ |
| linf | 内部节点最大绝对误差 | ✓ |
| resRMSE | PDE 残差的 RMS（训练目标） | ✗ |
| resLinf | PDE 残差的最大绝对值 | ✗ |
| bmax | 边界最大误差 | ✓ |
| gradRelL2 | 梯度的相对 L2 误差 | ✓ |

resRMSE 是唯一不依赖 u_exact 的指标——它只取决于预测解是否满足 PDE。如果 resRMSE 接近 oracle floor，说明 CNN 已经把 PDE 残差降到了离散精度的极限。

### 4. 为什么这个设计有效

1. **无标签训练**：loss 完全来自物理规律，不需要任何数值解作为标签
2. **守恒格式**：通量在交界面上共享，保证物理一致性
3. **硬边界**：边界条件精确满足，避免了 PINN 罚函数法中 PDE 和边界条件的 trade-off
4. **全网格监督**：每个内部节点都参与 loss，不像 PINN 只能采样有限点

---

## Demo 输出

```
checkpoints/
├── demo.pt              # 最终模型
├── demo_best.pt         # 验证集 relL2 最优模型
├── demo.csv             # 训练历史 (epoch, loss, relL2, resRMSE, ...)
└── demo.png             # 收敛曲线图

results/demo_figures/
├── metrics.csv          # 预测指标
└── sample_000_panel.png # 2×3 对比面板
    └── sample_001~003_panel.png
```

每个 2×3 面板包含：
- 上排：true u | CNN prediction | signed error
- 下排：absolute error | pred PDE residual | exact PDE residual (oracle floor)

---

## PINN.py 说明

PINN 使用相同的固定网格作为训练点，不同之处在于：
- MLP 输入 (x,y) 坐标，输出该点的 u 值
- 用 `torch.autograd.grad` 计算精确导数
- 边界条件通过罚函数 `(u - g)²` 施加
- 输出 `results/pinn_demo/pinn_panel.png`（与 CNN 相同的 2×3 布局，方便直接对比）

---

## 输出文件总览

| 文件 | 来源 | 说明 |
|------|------|------|
| `checkpoints/demo.png` | Baseline | CNN 收敛曲线 |
| `checkpoints/demo.csv` | Baseline | CNN 训练历史 |
| `checkpoints/pinn_demo.csv` | PINN | PINN 训练历史 |
| `results/demo_figures/*.png` | Baseline | CNN 预测面板 |
| `results/pinn_demo/pinn_panel.png` | PINN | PINN 预测面板 |
