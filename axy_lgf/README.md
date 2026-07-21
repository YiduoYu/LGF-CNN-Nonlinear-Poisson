# axy_lgf

`axy_lgf` 求解齐次 Dirichlet 边界条件下的线性变系数椭圆方程

\[
-\nabla\cdot\left(a(x,y)\nabla u\right)=f
\quad\text{in }(x_{\min},x_{\max})^2,
\qquad
u=0\quad\text{on }\partial\Omega.
\]

项目包含 PCG、Richardson、LGF bank 和 LGF-CNN。默认区域为
\([-1,1]^2\)，也可通过 `--domain-min` 和 `--domain-max` 统一配置。矩形区域上的
Dirichlet LGF 由 DST-I 实现，不使用 BAE。

## 环境

- Python 3.10+
- NumPy
- SciPy
- PyTorch
- Matplotlib（仅绘图需要）

以下命令都应在包含 `axy_lgf/` 的父目录中运行。

## 数据与预测约定

本项目支持两种相互独立的训练数据：

| `--data-kind` | 训练样本的生成方式 |
|---|---|
| `mms` | 先生成 \(u_\ast\)，再计算 \(f=A_hu_\ast\) |
| `solver` | 先生成 \(f\)，再由高精度 PCG 解出 \(u_\ast\) |

训练、验证和普通评估使用所选择的数据模式。`mms` 模型和 `solver`
模型分别训练，并保存在不同的实验目录中。

`--predict` 的含义固定不变：无论模型由哪种数据训练，最终都在同一组
MMS u1–u4 上预测。四个测试解共享同一个随机系数场，因而比较的是模型
面对不同解复杂度时的表现，而不是四个不同系数场带来的差异。

换句话说：

- MMS 训练模型 → 固定 MMS u1–u4 预测；
- solver 训练模型 → 同一组固定 MMS u1–u4 预测。

## 快速运行

### 经典求解器

```bash
python -m axy_lgf.run_demo --solver pcg
python -m axy_lgf.run_demo --solver richardson
python -m axy_lgf.run_demo --solver lgf_bank
python -m axy_lgf.run_demo --conv --problem smooth_sine
```

### 分别训练两个模型

```bash
python -m axy_lgf.train_lgf_cnn \
  --data-kind mms \
  --loss-mode supervised \
  --domain-min -1 --domain-max 1 \
  --epochs 120 --n 31 --K 6 --contrast 0.8

python -m axy_lgf.train_lgf_cnn \
  --data-kind solver \
  --loss-mode physics \
  --domain-min -1 --domain-max 1 \
  --epochs 120 --n 31 --K 6 --contrast 0.8
```

默认检查点目录为：

```text
axy_lgf/checkpoints/lgf_cnn_axy/<实验参数和哈希>/best.pt
axy_lgf/checkpoints/lgf_cnn_axy/<实验参数和哈希>/last.pt
axy_lgf/checkpoints/lgf_cnn_axy/<实验参数和哈希>/completed.json
```

实验目录名包含 `mms` 或 `solver`，两次训练不会相互覆盖。每个 epoch
会严格使用 `--samples-per-epoch` 指定的样本数；不足一个整 batch 的
最后部分会作为短 batch 处理。

每次训练或载入已有训练结果后，程序都会把图片和数值数据归纳到同一个
测试结果目录。训练参数由实验名和哈希区分，测试随机种子由末尾的
`s<test-seed>` 区分：

```text
axy_lgf/result/<实验参数和哈希>_s<test-seed>/
├── data/
│   ├── training_history.csv
│   ├── u1_u4_metrics.csv
│   ├── u1_u4_summary.json
│   ├── u1_u4_solver_comparison.csv
│   ├── u1_u4_residual_history.csv
│   ├── u1_u4_predictions.pt
│   ├── heldout_metrics.csv
│   ├── heldout_summary.json
│   ├── heldout_solver_comparison.csv
│   ├── heldout_residual_history.csv
│   └── heldout_predictions.pt
└── figures/
    ├── training_curve.png
    ├── u1_centerline.png
    ├── u1_panel.png
    ├── ...
    ├── u4_centerline.png
    └── u4_panel.png
```

默认一次绘制 `u1`–`u4`，四组图共享同一个测试系数场。可通过
`--plot-case u1|u2|u3|u4` 只绘制某一个 case；使用 `--no-plot`
可关闭自动绘图。所有图的坐标范围均来自 checkpoint 中保存的
`domain_min/domain_max`，不会在绘图代码中写死。

### 在同一组 MMS u1–u4 上比较

把下面两个路径替换为各自的 `best.pt`：

```bash
python -m axy_lgf.train_lgf_cnn \
  --predict \
  --predict-from axy_lgf/checkpoints/lgf_cnn_axy/<mms实验>/best.pt \
  --test-seed 123 \
  --output-dir results/predict_mms_model

python -m axy_lgf.train_lgf_cnn \
  --predict \
  --predict-from axy_lgf/checkpoints/lgf_cnn_axy/<solver实验>/best.pt \
  --test-seed 123 \
  --output-dir results/predict_solver_model
```

预测命令不需要 `--data-kind`。如果不指定 `--output-dir`，输出会自动放到
上面的参数安全结果目录；如果指定了该参数，其目录下仍按 `data/` 和
`figures/` 分类。

`u1_u4_metrics.csv` 和 `heldout_metrics.csv` 保存逐例的 `relL2`、RMSE、
MAE、最大误差、预测/参考离散残差、梯度误差、常数基线、
常系数 Poisson 基线、右端项 RMS 和 CNN 推理耗时。对应的
`*_summary.json` 保存每项指标的 mean/std/min/max。
（不再输出 `bndErr`：齐次 Dirichlet 边界由零 halo 精确施加，该指标
恒为零，不构成任何检验。）

`*_solver_comparison.csv` 保存 LGF-CNN、相同迭代步数的 Richardson 和
高精度 PCG 的误差、残差、耗时、迭代次数及收敛状态；
`*_residual_history.csv` 保存三种方法的逐步相对残差；
`*_predictions.pt` 保存系数场、右端项、参考解、网络预测和三种基线解。
三种方法的 `time_ms` 均按**逐样本**计时，CNN 使用不带诊断张量的
普通前向，以便跨方法直接比较。

### 分布外（OOD）扫描

固定的 u1–u4 和 heldout 批次都落在训练函数族内部
（\(p,q\in\{1,2,3\}\)、\(N(0,1)\) 幅值、单一 contrast），因此二者都
**没有**检验外推能力。`evaluate` 现在额外沿三个轴各做一次扫描，
结果写入 `ood_sweeps.csv`，并同时给出相同步数的 Richardson 作为对照：

| 扫描 | 取值 | 说明 |
|---|---|---|
| `frequency` | \(p=q=1,2,3,4,5,6,8\) | 训练只见过 \(\le 3\) |
| `amplitude` | \(\lVert f\rVert\times 10^{-2}\dots10^{2}\) | 由等变性保证恒定，作回归护栏 |
| `contrast` | \(0.2,0.5,0.8,0.95,1.0\) | 越界后 \(a\) 会被下界削平 |

两个命令必须使用相同的 `--test-seed`，这样系数场和 u1–u4 完全一致，
结果才可以直接比较。

### 独立评估与绘图

```bash
python -m axy_lgf.train_lgf_cnn \
  --eval-only path/to/best.pt \
  --test-seed 123
```

训练中的固定验证集使用 `seed + 1`，最终评估默认使用独立的
`--test-seed 123`，避免验证集和测试集复用。`--eval-only` 会载入指定
checkpoint，并通过唯一的 `plot_lgf_cnn.py` 自动生成 training curve、
u1–u4 panel 和 centerline；使用 `--no-plot` 可以只保存数据。

## 数学与实现要点

离散算子采用守恒通量形式的五点模板：

\[
A_hu=-\operatorname{div}_h(a\nabla_hu).
\]

常系数参考算子的逆

\[
\left(a_0(-\Delta_h)+\lambda I\right)^{-1}
\]

由二维正交 DST-I 精确应用。NumPy 和 PyTorch 的 LGF bank 均只执行
一次正变换，再批量处理全部 \(\lambda_j\)，以减少重复运算。PyTorch
版本使用 FFT 奇延拓实现 DST，不再保存稠密的 \(n\times n\) 正弦矩阵。

经典 LGF bank 的局部修正为

\[
\delta u=\omega\,\frac{a_0}{a}
\sum_j\alpha_jG_{\lambda_j}r.
\]

LGF-CNN 使用同样的 \(a_0/a\) 缩放，并学习逐点混合权重
\(\alpha_j(x)\) 与阻尼 \(\eta(x)\)。

### 精确的缩放等变性

方程是线性的：\(f\to cf\) 应给出 \(u\to cu\)，\(a\to ca\) 应给出
\(u\to u/c\)。softmax/sigmoid 是非线性的，若直接把原始幅值的 \(r\)、
\(z_j\) 喂给 mixer，\(\alpha\)、\(\eta\) 就会随问题的量级漂移。因此
每个 block 先用**逐样本**的尺度把 mixer 的输入无量纲化：

- \(r/\lVert r\rVert\)、\(z_j/\max_j\lVert z_j\rVert\)；
- 系数通道用 \(a/a_0\) 与 \(a_0/a\)，其中 \(a_0=\mathrm{mean}(a)\) 逐样本取值；
- LGF bank 的分母整体提出 \(a_0\)，即 \(a_0\bigl((-\Delta_h)+\lambda_j I\bigr)\)。

这样 \(\alpha\)、\(\eta\) 对上述两种缩放都是零次齐次，而 \(z\) 是 \(r\)
的线性函数，于是整个网络映射对 \(f\) 精确一次齐次、对 \(a\) 精确
\(-1\) 次齐次——线性方程本就该有的性质被写进结构，而不是指望网络
学出来。该性质与权重无关，对未训练的模型同样成立，实测在
\(10^{-6}\sim10^{6}\) 范围内偏差约 \(10^{-16}\)（机器精度）。

随机系数场要求 `contrast` 位于 `[0, 1]`，并保证 \(a\ge 0.05\)。
所有带空间频率的函数都按区域长度 \(L\) 缩放。

## 检查点安全

新检查点为格式版本 4，通过 `torch.load(..., weights_only=True)` 加载。
**版本 4 重写了 block 的缩放方式**（mixer 输入按逐样本尺度归一化、
LGF bank 使用逐样本 \(a_0\)），旧 checkpoint 的权重在新 block 下含义
不同，因此 `load_model` 会直接拒绝而不是给出错误的结果——需要用当前
代码重新训练。
训练先写入独立的临时实验目录，只有 `best.pt`、`last.pt` 和
`history.csv` 全部完成并通过哈希记录后才写入 `completed.json` 并提交。
同一参数的并发训练由文件锁串行化；中断训练不会被下一次普通运行误当作
完整实验。

## 快速自检

本包目前没有自动化测试。改动之后建议至少跑一次 1-epoch 的完整流水线：

```bash
python -m axy_lgf.train_lgf_cnn \
  --n 16 --J 3 --K 2 --hidden 8 --epochs 1 --eval-every 1 \
  --samples-per-epoch 8 --batch-size 4 --val-samples 4
```

绘图模块只在整个训练结束后才被导入使用，所以那里的导入错误不会在
任何早期检查中暴露，只有跑完一次完整训练才会出现
（`plots.py` 改名为 `plot_lgf_cnn.py` 时正是如此）。

## 主要文件

| 文件 | 作用 |
|---|---|
| `grid.py` | 方形内部网格 |
| `operators.py` | NumPy 离散算子与范数 |
| `lgf_inverse.py` | Dirichlet LGF |
| `lgf_bank.py` | NumPy screened LGF bank |
| `solvers.py` | PCG、Richardson、LGF bank |
| `manufactured.py` | 制造解问题 |
| `torch_ops.py` | PyTorch 算子、FFT-DST 与 LGF bank |
| `lgf_cnn.py` | 展开的 LGF-CNN |
| `data.py` | MMS/solver 随机训练数据 |
| `train_lgf_cnn.py` | 训练、评估和固定 u1–u4 预测 |
| `plot_lgf_cnn.py` | training curve、panel 和 centerline 绘图 |
| `evaluation.py` | 共享评估指标、求解器对比与 OOD 扫描 |
