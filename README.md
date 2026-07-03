# Nonlinear Poisson — CNN vs PINN

求解通量形式非线性 Poisson 方程的神经网络方法：

```text
-div(u^k grad u) = f,   u = g on boundary,   Omega = [-1,1]^2
```

---

## 训练目标

**PDE 残差 only。** u_exact 仅用于生成 f、g 和评估指标，不参与训练。

```
R = -div_h(u_pred^k * grad_h(u_pred)) - f
loss = mean((R / RMS(f))^2)
```

硬边界：`u_pred = boundary * g + (1-boundary) * softplus(u_raw)`，保证 u > 0。

---

## 数据生成：随机 PDE Family

每个样本是**一道不同的 PDE**（operator-learning 风格）：

```
u = b(x,y) + (1-x^2)(1-y^2) * s(x,y)    随机参数
k ~ U(0.5, 2.5)                           随机非线性指数
f 通过 autograd 精确计算（--rhs-mode continuous）或 FD 模板反算（--rhs-mode discrete）
```

---

## 模型架构

SimpleCNN：n 层 Conv3×3 + GroupNorm + GELU → 1×1 Conv 输出。无池化、无上采样、无跳跃连接，~48k 参数（N=32）。

---

## Quick Start

```bash
# CNN baseline（主实验）
python Baseline.py --N 32 --epochs 40 --train-samples 2048 --val-samples 128 --rhs-mode continuous --retrain

# debug 模式（验证残差代码：R_h(u_exact) = 0）
python Baseline.py --N 32 --epochs 5 --train-samples 128 --val-samples 32 --rhs-mode discrete --retrain

# PINN 对照
python PINN.py --N 32 --epochs 5000

# 自定义 k 范围
python Baseline.py --k-min 1.0 --k-max 3.0
```

---

## 存储结构

```
checkpoints/
├── cnn/N32_e40_krand/               # CNN (physics mode)
│   ├── best.pt / last.pt / history.csv / convergence.png / run_info.txt
├── cnn/N32_e40_krand_discrete/      # CNN (debug/discrete mode)
├── pinn/N32_e5000_h64_l4_bc50/       # PINN
└── registry.json

results/
├── cnn/N32_e40_krand/figures/
│   ├── history/                      # 旧运行自动归档
│   ├── sample_000_panel.png
│   └── metrics.csv
```

---

## 评估指标

| Metric | 含义 | 依赖 u_exact? |
|--------|------|:---:|
| relL2 | 预测解 vs 精确解的相对 L2 误差（内部点） | Yes |
| resRMSE | PDE 残差的 RMS（训练目标） | No |
| linf | 内部节点最大绝对误差 | Yes |
| gradRelL2 | 梯度的相对 L2 误差 | Yes |
| bmax | 边界最大误差（硬边界下 = 0） | Yes |
| Oracle floor | 精确解在 FD 离散下的残差 RMS（截断误差地板） | Yes |

Best checkpoint 按 val resRMSE 选择，relL2 仅作为评估指标报告。

---

## 关键设计

- Hard Dirichlet BC：边界点直接替换，精确满足
- Softplus positivity：内部点保证 u > 0，非整数 k 时 u^k 合法
- Conservative FD residual：交界面通量算术平均，格式严格守恒
- 6-channel input：`[f, g, boundary_mask, k, x, y]`

---

## 实验结论

1. CNN + FD 的精度上限由 FD 截断误差决定（N=32 时最优 relL2 ~ 0.316）
2. 加密网格、增大模型、高阶 FD、延长训练——均无法突破天花板
3. PINN autodiff 精确导数是决定性优势（relL2 = 0.004 vs CNN 最优 0.316）
