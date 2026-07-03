# Nonlinear Poisson — CNN / TransUNet vs PINN

求解通量形式非线性 Poisson 方程的神经网络方法对比：

```text
-div(u^k grad u) = f,   u = g on boundary,   Omega = [-1,1]^2
```

| | Baseline.py (CNN) | TransUNet.py | PINN.py (MLP) |
|---|---|---|---|
| 模型 | U-Net, grid->grid | U-Net + Transformer bottleneck | MLP, (x,y)->u |
| 导数 | 有限差分 O(h^2) | 有限差分 O(h^2) | autodiff (精确) |
| 边界 | 硬约束（直接替换） | 硬约束（直接替换） | 软约束（罚函数） |
| 训练信号 | PDE 残差 + 可选监督 | PDE 残差 + 可选监督 | PDE 残差 + 边界罚函数 |
| 训练策略 | AdamW + warmup + cosine | AdamW + warmup + cosine | Adam + L-BFGS |

---

## 数据生成：随机 PDE Family（Operator-Learning 风格）

**每个样本是一道不同的 PDE。** CNN/TransUNet 学习的是算子 (f, g, k) → u，而不是背住某一题的答案。

### 每道题随机生成一个制造解

```
u(x,y) = b(x,y) + (1-x^2)(1-y^2) * s(x,y)
```

其中：
- `b(x,y) = c0 + c1*x + c2*y + c3*xy` — 随机边界多项式，c0~c3 每个样本不同
- `(1-x^2)(1-y^2)` — 边界消隐因子，保证 u 在边界上 = b
- `s(x,y) = a1*sin(pi*x)*sin(pi*y) + a2*cos(2*pi*x)*cos(pi*y)` — 随机内部调制，a1, a2 每个样本不同

### 随机非线性指数

```
k ~ U(0.5, 2.5)，每个样本不同
```

### 两种 RHS 生成模式

| 模式 | 方法 | f 的计算 | R_h(u_exact) |
|------|------|---------|:---:|
| **continuous**（主实验） | autograd 精确求导 | f = -div(u^k * grad(u)) 连续形式 | ≠ 0（oracle floor） |
| **discrete**（debug） | FD 模板反算 | f = -div_h(u^k * grad_h(u)) 离散形式 | = 0（机器精度） |

discrete 模式用于验证残差代码正确性——如果 R_h(u_exact) ≠ 0，说明实现有 bug。

### 每个样本包含

```
[f_i, g_i, boundary_mask, k_i, x, y] → u_i
```

2048 个训练样本 = 2048 道不同的 PDE 题。验证集同样随机生成。

---

## 三种训练模式

| 模式 | Loss | 说明 |
|------|------|------|
| **physics** | L_PDE = mean((R_h / RMS(f))^2) | 纯 PDE 残差，完全无监督 |
| **supervised** | L_sup = MSE(u_pred, u_exact) | 内部点监督学习（需要 u_exact） |
| **hybrid** | 前 N 轮 L_sup，之后 0.1*L_sup + L_PDE | 监督预训练 + physics 微调（推荐） |

---

## 核心设计

### 硬边界约束

```python
u_final = boundary_mask * g + (1 - boundary_mask) * u_raw
```

边界点直接用 g 替换。Dirichlet 条件精确满足到机器精度，不需要罚函数。

### 通量守恒有限差分残差

交界面扩散系数取相邻网格点的算术平均，相邻控制体共享同一通量，格式严格守恒。

### 六通道输入

| Channel | Content | Purpose |
|:----:|------|------|
| 1 | f (source term) | PDE 右端项 |
| 2 | g (boundary values) | 边界条件 |
| 3 | boundary_mask (0/1) | 哪些点是边界 |
| 4 | k (nonlinear exponent) | a(u) = u^k |
| 5 | x coordinate | 网格点位置 |
| 6 | y coordinate | 网格点位置 |

### 架构

| | Baseline (CNN) | TransUNet |
|---|---|---|
| Encoder | ConvBlock × 3 + avg_pool | 同 Baseline |
| Bottleneck | ConvBlock + DilatedBlock(d=2) + DilatedBlock(d=4) | ConvBlock + **Transformer ×4** + DilatedBlock(d=2) |
| Decoder | Upsample + skip connection × 3 | 同 Baseline |

---

## Quick Start

```bash
# CNN baseline
python Baseline.py --N 32 --epochs 40 --random-u

# CNN + hybrid training
python Baseline.py --N 32 --epochs 40 --random-u \
    --train-mode hybrid --pretrain-epochs 20

# TransUNet
python TransUNet.py --N 32 --epochs 40 --random-u

# debug 模式（验证残差代码）
python Baseline.py --N 32 --epochs 5 --random-u --rhs-mode discrete

# PINN 对照
python PINN.py --N 32 --epochs 5000

# 强制重新训练
python Baseline.py --N 32 --epochs 40 --random-u --retrain

# 只预测
python Baseline.py --load-path checkpoints/cnn/.../best.pt --predict --plot
```

---

## 存储结构

采用 config-key 命名——同一配置写入同一目录，自动跳过训练。新运行时，旧结果自动归档到 `history/`。

```
checkpoints/
├── cnn/N32_e40_k2_rand/              # CNN, random PDE, physics mode
│   ├── best.pt / last.pt / history.csv / convergence.png / run_info.txt
├── cnn/N32_e40_k2_rand_hybrid/       # CNN, random PDE, hybrid mode
├── transunet/N32_e40_k2_tf4_rand/   # TransUNet, random PDE
├── pinn/N32_e5000_h64_l4_bc50/       # PINN
├── cnn/latest -> N32_e40_k2_rand
└── registry.json

results/
├── cnn/N32_e40_k2_rand/figures/
│   ├── history/                      # 旧运行自动归档，按时间戳命名
│   ├── sample_000_panel.png          # 最新运行
│   └── metrics.csv
```

Config-key 命名：
- CNN: `N{网格}_e{epoch}_k{k}` + `_rand`（随机u）+ `_{mode}`（非physics时）
- TransUNet: 同上 + `_tf{layers}`
- PINN: `N{网格}_e{epoch}_h{hidden}_l{layers}_bc{bc_weight}`

---

## 评估指标

| Metric | 含义 | 依赖 u_exact? |
|--------|------|:---:|
| relL2 | 预测解 vs 精确解的相对 L2 误差（内部点） | Yes |
| linf | 内部节点最大绝对误差 | Yes |
| resRMSE | PDE 残差的 RMS（训练目标） | No |
| bmax | 边界最大误差（硬边界下应为 0） | Yes |
| gradRelL2 | 梯度的相对 L2 误差 | Yes |
| Oracle floor | 精确解在 FD 离散下的残差 RMS（截断误差地板） | Yes |

---

## 实验结论

1. CNN + FD 的精度上限由 FD 截断误差决定，N=32 时最优 relL2 ~ 0.316
2. 加密网格、增大模型、高阶 FD、延长训练、多分辨率训练——均无法突破天花板
3. Transformer bottleneck（TransUNet）同样不能突破，反而更差（全局注意力破坏局部算子偏置）
4. 随机 PDE family 数据生成让 CNN 从"背题"变为"学算子"
5. Supervised warm-start + physics fine-tuning 是最务实的训练策略
6. PINN autodiff 精确导数是决定性优势（relL2 = 0.004 vs CNN 最优 0.316）
