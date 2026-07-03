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
| 训练信号 | PDE 残差（无标签） | PDE 残差（无标签） | PDE 残差 + 边界罚函数 |
| 训练策略 | AdamW + warmup + cosine | AdamW + warmup + cosine | Adam + L-BFGS |

---

## 两种数据生成模式

### 固定 PDE（默认）

```text
u = sin(pi*x) * cos(pi*y)   ← 固定制造解
k = 2.0                     ← 固定非线性指数
→ 所有样本 = 同一个 PDE 的拷贝（CNN 在"背题"）
```

### 随机 PDE Family（推荐）

```text
u = b(x,y) + (1-x^2)(1-y^2) * s(x,y)
  b = c0 + c1*x + c2*y + c3*xy        ← 随机边界多项式
  s = a1*sin(pi*x)*sin(pi*y)          ← 随机内部调制
    + a2*cos(2*pi*x)*cos(pi*y)
k ~ U(0.5, 2.5)                        ← 随机非线性指数
f = -div(u^k grad u)                   ← autograd 精确算 f
→ 每个样本 = 一道不同的 PDE（CNN 在学"算子"）
```

RHS 模式：
- `--rhs-mode continuous`：autograd 精确 f（主实验）
- `--rhs-mode discrete`：FD 模板反算 f（debug，R_h(u_exact) = 0）

---

## 三种训练模式

| 模式 | 说明 |
|------|------|
| `--train-mode physics` | 纯 PDE 残差 loss（无监督） |
| `--train-mode supervised` | 内部点 MSE on u_exact（有监督） |
| `--train-mode hybrid` | 监督预训练 + physics 微调 |

---

## Quick Start

```bash
# CNN 固定 PDE（原版）
python Baseline.py --N 32 --epochs 40

# CNN 随机 PDE family（推荐）
python Baseline.py --N 32 --epochs 40 --random-u

# CNN 随机 PDE + 监督预训练 + physics 微调
python Baseline.py --N 32 --epochs 40 --random-u \
    --train-mode hybrid --pretrain-epochs 20

# TransUNet 随机 PDE family
python TransUNet.py --N 32 --epochs 40 --random-u

# debug 模式（验证残差代码）
python Baseline.py --N 32 --epochs 5 --random-u --rhs-mode discrete

# PINN 对照
python PINN.py --N 32 --epochs 5000

# 强制重新训练
python Baseline.py --N 32 --epochs 40 --random-u --retrain

# 只预测
python Baseline.py --load-path checkpoints/cnn/N32_e40_k2_rand/best.pt --predict --plot
```

---

## 存储结构

采用 **config-key** 命名——同一配置永远写入同一目录，自动覆盖，自动跳过训练。

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
│   ├── history/                      # 旧运行自动归档
│   │   ├── 20260703_120000/
│   │   └── 20260703_150000/
│   ├── sample_000_panel.png          # 最新运行
│   └── metrics.csv
└── transunet/N32_e40_k2_tf4_rand/figures/
```

**Config-key 命名规则**：
- CNN: `N{网格}_e{epoch}_k{k}` + `_rand`（随机u）+ `_{mode}`（非physics训练模式）
- TransUNet: 同上 + `_tf{layers}`
- PINN: `N{网格}_e{epoch}_h{hidden}_l{layers}_bc{bc_weight}`

结果归档：新运行时，旧的 png/csv 自动移入 `history/<timestamp>/`。

---

## 核心设计

### 硬边界约束
```python
u_final = boundary_mask * g + (1 - boundary_mask) * u_raw
```
边界点直接用 g 替换，Dirichlet 条件精确满足。

### 通量守恒有限差分残差
交界面扩散系数算术平均，相邻控制体共享同一通量，格式严格守恒。

### 六通道输入
`[f, g, boundary_mask, k, x, y]` → U-Net → u_pred

### 评估指标
| Metric | 含义 | 依赖 u_exact? |
|--------|------|:---:|
| relL2 | 预测解 vs 精确解的相对 L2 误差 | Yes |
| linf | 内部节点最大绝对误差 | Yes |
| resRMSE | PDE 残差 RMS（训练目标） | No |
| bmax | 边界最大误差 | Yes |
| gradRelL2 | 梯度的相对 L2 误差 | Yes |
| Oracle floor | 精确解的 FD 残差 RMS（截断误差地板） | Yes |

---

## 实验结论

1. CNN + FD 精度上限由 FD 截断误差决定（N=32 时 relL2 ~ 0.316）
2. 加密网格、增大模型、高阶 FD、延长训练、多分辨率训练、Transformer 架构——均无法突破天花板
3. 固定 u 的制造解导致 CNN "背题"而非"学算子"
4. 随机 PDE family + autograd f 是更合理的数据生成方式
5. Supervised warm-start + physics fine-tuning 是最务实的训练策略
6. PINN autodiff 精确导数是决定性优势（relL2 = 0.004 vs CNN 最优 0.316）
