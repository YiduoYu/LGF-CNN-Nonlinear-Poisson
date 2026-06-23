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

Run the discrete residual self-check:

```bash
python3 Baseline.py --check-residual --N 32 --device cpu
```

Train a small baseline:

```bash
python3 Baseline.py --N 32 --epochs 200 --pretrain-epochs 50 --batch-size 8 --rhs-mode continuous --save-path checkpoints/unet_n32.pt
```

Pure physics-only training:

```bash
python3 Baseline.py --N 32 --epochs 200 --pure-physics --pretrain-epochs 0 --rhs-mode continuous
```

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

Training uses manufactured nonlinear Poisson problems. The only PDE residual is
the conservative flux-form finite-difference stencil with shared face fluxes:

```text
R_flux(u_pred) = -div_h(u_pred^k grad_h u_pred) - f
```

For `rhs_mode=discrete`, the manufactured RHS is generated with the same flux
stencil:

```text
f = -div_h(u_exact^k grad_h u_exact)
```

Prediction and plotting reuse the same checkpoint loader and manufactured test
data path as normal prediction. For each sample, the code computes:

- `error = u_pred - u_exact`
- `abs_error = |u_pred - u_exact|`
- interior `relL2`, `rmse`, `mae`, and `linf`
- boundary maximum error
- CNN flux residual RMSE in `pred_flux_resRMSE`
- oracle flux residual RMSE from `u_exact` in `exact_flux_resRMSE`

Figures are saved with Matplotlib `imshow` panels and colorbars using
`plt.savefig(..., dpi=200, bbox_inches="tight")`, followed by `plt.close(fig)`.

## Notes

- Full grid size is `(N + 2) x (N + 2)`, including boundary nodes.
- Input channels are `[f, boundary_g, boundary_mask, k, x, y]`.
- Boundary values are imposed exactly by projection, so the boundary error should be near floating-point roundoff.
- `rhs_mode=discrete` is useful for debugging because the manufactured solution satisfies the same finite-difference stencil used by the loss.
- `rhs_mode=continuous` computes the manufactured RHS from the continuous PDE by automatic differentiation.
