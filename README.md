# Nonlinear Poisson CNN Baseline

This project trains a CNN baseline for the original flux-form nonlinear
Poisson equation:

```text
-div(u^k grad u) = f,  u = g on boundary
```

The baseline uses a grid-to-grid U-Net, exact Dirichlet boundary enforcement,
and one finite-difference PDE residual on the interior nodes:

```text
R_flux(u) = -div_h(u^k grad_h u) - f
```

## Quick Start

Run the full one-click demo. This is also what happens when `Baseline.py` is
launched with no command-line arguments, for example by clicking Run in VSCode:

```bash
python3 Baseline.py
```

or explicitly:

```bash
python3 Baseline.py --demo
```

Check the manufactured oracle residual floor:

```bash
python3 Baseline.py --check-oracle-floor --N 32 --device cpu
```

Train a small baseline:

```bash
python3 Baseline.py --N 32 --epochs 200 --batch-size 8 --save-path checkpoints/unet_n32.pt
```

## Residual Training Method

Recommended run:

```bash
python3 Baseline.py --N 32 --epochs 200
```

The data is manufactured continuously: choose analytic `u_exact(x,y)`, sample
`k`, and compute `f = -div(u_exact^k grad u_exact)` with analytic derivatives.
The training objective is only the normalized conservative flux residual:

```text
loss_pde = mean(([-div_h(u_pred^k grad_h u_pred) - f] / rms(f))^2)
```

There is no supervised MSE warmup or mixed supervised/PDE objective. `u_exact`
is used to manufacture `f`, provide boundary values `g`, and report evaluation
metrics/plots after training.

## One-Click Demo

Demo mode runs the full pipeline with small but meaningful defaults:

```text
N=32, train_samples=2048, val_samples=128, epochs=40,
batch_size=16, predict_samples=4
```

It saves:

- Final checkpoint: `checkpoints/demo.pt`
- Best validation checkpoint: `checkpoints/demo_best.pt`
- Demo metrics: `results/demo_figures/metrics.csv`
- Demo panels: `results/demo_figures/sample_000_panel.png` through `sample_003_panel.png`

Each panel contains true solution, CNN prediction, signed error, absolute
error, the flux PDE residual, and the forcing term `f`. Use `--show-plots` if
you also want Matplotlib to display the figures interactively after saving
them.

The committed demo results in this branch were generated with the demo code in
this branch.

## Implementation Method

The model takes six input channels: `[f, boundary_g, boundary_mask, k, x, y]`.
The U-Net predicts a full-grid solution and then applies an exact Dirichlet
projection, so boundary values are copied from the manufactured boundary data.

Training uses manufactured nonlinear Poisson problems. The forcing `f` comes
from the continuous PDE using analytic derivatives of `u_exact`. The only PDE
residual used by the loss is the conservative flux-form finite-difference
stencil with shared face fluxes:

```text
R_flux(u_pred) = -div_h(u_pred^k grad_h u_pred) - f
```

Prediction and plotting reuse the same checkpoint loader, manufactured test
data path, and shared prediction output pipeline. For each sample, the code
computes these evaluation-only reported metrics:

- `error = u_pred - u_exact`
- `abs_error = |u_pred - u_exact|`
- interior `relL2`, `rmse`, `mae`, and `linf`
- boundary maximum error
- CNN flux residual RMSE in `pred_flux_resRMSE`
- oracle flux residual RMSE from `u_exact` in `exact_flux_resRMSE`

These `u_pred` versus `u_exact` comparisons are evaluation outputs, not the
default backward training objective.

Figures are saved with Matplotlib `imshow` panels and colorbars using
`plt.savefig(..., dpi=200, bbox_inches="tight")`, followed by `plt.close(fig)`.

## Notes

- Full grid size is `(N + 2) x (N + 2)`, including boundary nodes.
- Input channels are `[f, boundary_g, boundary_mask, k, x, y]`.
- Boundary values are imposed exactly by projection, so the boundary error should be near floating-point roundoff.
- The manufactured RHS is always continuous analytic forcing.
- The training loss is always the flux-form PDE residual; `u_exact` comparisons are evaluation only.
