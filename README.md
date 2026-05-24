# Nonlinear Poisson CNN Baseline

This repository contains a PyTorch baseline for a nonlinear Poisson equation in flux form:

```text
-div(a(u) grad u) = f,  a(u) = u**k,  u = g on boundary
```

The baseline uses a grid-to-grid U-Net, exact Dirichlet boundary enforcement, and a conservative finite-difference flux residual on the interior nodes.

## Quick Start

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

## Notes

- Full grid size is `(N + 2) x (N + 2)`, including boundary nodes.
- Input channels are `[f, boundary_g, boundary_mask, k, x, y]`.
- Boundary values are imposed exactly by projection, so the boundary error should be near floating-point roundoff.
- `rhs_mode=discrete` is useful for debugging because the manufactured solution satisfies the same finite-difference stencil used by the loss.
- `rhs_mode=continuous` computes the manufactured RHS from the continuous PDE by automatic differentiation.
