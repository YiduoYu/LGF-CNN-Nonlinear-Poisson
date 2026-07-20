# axy_lgf

Neural / LGF solvers for the **linear variable-coefficient** elliptic equation

\[
-\nabla\cdot\bigl(a(x,y)\nabla u\bigr)=f
\quad\text{in }\Omega=(0,1)^2,
\qquad
u=0\quad\text{on }\partial\Omega.
\]

This package is self-contained. Domain is a rectangle with homogeneous Dirichlet BC, so the discrete Green’s function is applied by a fast sine transform (DST). **No BAE.**

---

## Requirements

- Python 3.10+
- `numpy`, `scipy`, `torch`, `matplotlib`

Activate your environment, then run commands from the parent directory that contains `axy_lgf/`:

```bash
source ~/myenv/bin/activate
cd /path/to/parent   # folder containing axy_lgf/
```

---

## Quick start

### Classical LGF solvers

```bash
python -m axy_lgf.run_demo
python -m axy_lgf.run_demo --n 63 --solver pcg
python -m axy_lgf.run_demo --solver richardson
python -m axy_lgf.run_demo --solver lgf_bank
python -m axy_lgf.run_demo --conv --problem smooth_sine
```

### Train / evaluate LGF-CNN

```bash
python -m axy_lgf.train_lgf_cnn --epochs 120 --n 31 --K 6 --contrast 0.8
python -m axy_lgf.train_lgf_cnn --eval-only axy_lgf/checkpoints/lgf_cnn_axy.pt
python -m axy_lgf.plot_lgf_cnn
```

Figures are written to `axy_lgf/figures/`.  
Checkpoint (if trained): `axy_lgf/checkpoints/lgf_cnn_axy.pt`.

---

## Mathematical setup

### Discretization

On an \(n\times n\) interior grid with spacing \(h=1/(n+1)\), the operator

\[
A_h u = -\operatorname{div}_h\bigl(a\,\nabla_h u\bigr)
\]

is a standard flux-form 5-point stencil with zero Dirichlet halo.

### Dirichlet LGF

For constant diffusivity \(a_0>0\) (and optional screening \(\lambda\ge 0\)),

\[
\bigl(a_0(-\Delta_h)+\lambda I\bigr)^{-1}
\]

is applied exactly by a 2D orthonormal DST-I. This is the rectangular Dirichlet lattice Green’s function.

### Classical iterative methods

| Method | Update / idea |
|--------|----------------|
| **PCG** | Krylov solve of \(A_h u=f\) with preconditioner \((-a_0\Delta_h)^{-1}\), \(a_0=\operatorname{mean}(a)\) |
| **Richardson** | \(u\leftarrow u+\omega\,(-a_0\Delta_h)^{-1}(f-A_h u)\) |
| **LGF bank** | Mix several screened LGFs \(G_{\lambda_j}\) with local \(1/a\) scaling |

### LGF-CNN

Fixed screened LGF filters + learned pointwise mixing:

\[
r=f-A_h u,\qquad
z_j=G_{\lambda_j}r,\qquad
u\leftarrow u+\eta_{\max}\,\eta\,\frac{1}{a}\sum_j\alpha_j z_j.
\]

- \(G_{\lambda_j}\) are **not** trained (physics prior).
- Only a small \(1\times1\) MLP learns \(\alpha_j(x)\) and damping \(\eta(x)\).
- The network is an unrolled residual iteration with \(K\) shared-weight blocks.

**Training data.** Random smooth \(a(x,y)\) and sine-series solutions \(u_\ast\) with homogeneous BC; forcing is discrete-consistent, \(f=A_h u_\ast\).

**Metrics.** Relative discrete \(L^2\) residual and solution error:

\[
\operatorname{rel}_r=\frac{\|f-A_h u\|_2}{\|f\|_2},\qquad
\operatorname{rel}_u=\frac{\|u-u_\ast\|_2}{\|u_\ast\|_2}.
\]

---

## Package layout

| File | Role |
|------|------|
| `grid.py` | Square interior grid |
| `operators.py` | NumPy `-div(a grad)`, norms |
| `lgf_inverse.py` | Dirichlet LGF via DST |
| `lgf_bank.py` | Screened LGF bank (NumPy) |
| `solvers.py` | PCG / Richardson / bank solvers |
| `manufactured.py` | Fixed test problems |
| `torch_ops.py` | PyTorch operators + LGF bank |
| `lgf_cnn.py` | Unrolled LGF-CNN model |
| `data.py` | Random training batches |
| `run_demo.py` | Classical solver CLI |
| `train_lgf_cnn.py` | Train / evaluate LGF-CNN |
| `plot_lgf_cnn.py` | Solution / residual / mixing plots |

---

## Suggested experiments

1. Run `run_demo --conv` and check ~2nd-order \(L^2\) convergence (analytic manufactured forcing).
2. Compare PCG vs Richardson vs LGF-CNN at fixed iteration budget \(K\).
3. Vary `--contrast` and see how many PCG iterations / CNN residual change.
4. Plot mixing weights: the Poisson (\(\lambda=0\)) channel usually dominates.

---

## Notes

- For this **linear** \(a(x,y)\) problem, classical PCG with an LGF preconditioner is the strongest solver if you allow it to converge tightly. LGF-CNN is mainly a fixed-budget / learned-iteration baseline and a stepping stone to nonlinear problems.
- Quasilinear \(a(u)\) lives in a separate package: `quasi_lgf/`.
