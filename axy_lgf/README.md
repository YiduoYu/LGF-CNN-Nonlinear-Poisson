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

以下命令都应在包含 `axy_lgf/`、`tests/` 和 `tools/` 的目录中运行。

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

每次新训练结束后会自动按照 PINN 的布局生成以下图片：

```text
axy_lgf/result/<实验参数和哈希>/s<test-seed>/training_curve.png
axy_lgf/result/<实验参数和哈希>/s<test-seed>/u1_centerline.png
axy_lgf/result/<实验参数和哈希>/s<test-seed>/u1_panel.png
...
axy_lgf/result/<实验参数和哈希>/s<test-seed>/u4_centerline.png
axy_lgf/result/<实验参数和哈希>/s<test-seed>/u4_panel.png
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

预测命令不需要 `--data-kind`。输出包括：

- `metrics.csv`：u1–u4 各自的相对残差和相对解误差；
- `summary.json`：均值、最大值、训练数据模式、测试种子及检查点哈希。

两个命令必须使用相同的 `--test-seed`，这样系数场和 u1–u4 完全一致，
结果才可以直接比较。

### 独立评估与绘图

```bash
python -m axy_lgf.train_lgf_cnn \
  --eval-only path/to/best.pt \
  --test-seed 123

python -m axy_lgf.plot_lgf_cnn \
  --ckpt path/to/best.pt \
  --outdir results/figures
```

训练中的固定验证集使用 `seed + 1`，最终评估默认使用独立的
`--test-seed 123`，避免验证集和测试集复用。绘图要求显式传入检查点，
防止误用旧模型。

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
\(\alpha_j(x)\) 与阻尼 \(\eta(x)\)。该写法保证系数 \(a\) 整体乘以常数
时的缩放关系正确。

随机系数场要求 `contrast` 位于 `[0, 1]`，并保证 \(a\ge 0.05\)。
所有带空间频率的函数都按区域长度 \(L\) 缩放。

## 检查点安全

新检查点为格式版本 3，可通过
`torch.load(..., weights_only=True)` 加载，不再保存重复算子或稠密 DST
矩阵。包内示例检查点已经迁移。
普通训练、评估、预测和绘图路径始终使用安全加载；旧版 checkpoint
缺少纯 loss 元数据时会明确标记为 `legacy_mixed`。
训练先写入独立的临时实验目录，只有 `best.pt`、`last.pt` 和
`history.csv` 全部完成并通过哈希记录后才写入 `completed.json` 并提交。
同一参数的并发训练由文件锁串行化；中断训练不会被下一次普通运行误当作
完整实验。

## 测试

```bash
python -m unittest discover -s tests -v
python -m compileall -q axy_lgf tests tools
```

回归测试覆盖：

- MMS 和 solver 标签的一致性；
- 固定 MMS u1–u4 的跨模型预测约定；
- NumPy/Torch 离散算子与 LGF bank 一致性；
- FFT-DST 数值正确性和自动微分；
- LGF 系数缩放不变性和真实残差历史；
- 非默认区域边界下的收敛；
- `J=1` 绘图和安全检查点往返加载。

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
| `plot_lgf_cnn.py` | 解、残差和混合权重绘图 |
