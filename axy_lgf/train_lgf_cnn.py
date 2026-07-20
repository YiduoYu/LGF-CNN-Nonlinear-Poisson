#!/usr/bin/env python3
"""Train / evaluate LGF-CNN on rectangle a(x,y) problems.

Usage:
    source ~/myenv/bin/activate
    cd /Users/qing/Dropbox/Research/LGFNN
    python -m axy_lgf.train_lgf_cnn
    python -m axy_lgf.train_lgf_cnn --epochs 100 --n 31 --K 6
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .data import make_batch
from .grid import SquareGrid
from .lgf_cnn import LGFCNN, LGFCNNTrainConfig
from .manufactured import make_problem
from .solvers import solve_axy_pcg, solve_axy_richardson
from .operators import apply_div_a_grad, l2_norm


def train(cfg: LGFCNNTrainConfig, save_path: Path | None = None) -> LGFCNN:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = torch.device(cfg.device)
    grid = SquareGrid(n=cfg.n)

    model = LGFCNN(
        grid,
        J=cfg.J,
        K=cfg.K,
        hidden=cfg.hidden,
        eta_max=cfg.eta_max,
        share_weights=True,
        dtype=torch.float64,
        device=device,
    )
    # Mixers on device; bank buffers follow model.device via constructor device=
    model.mixers.to(device)
    model.bank.to(device)

    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
    )

    print(
        f"[train] n={cfg.n}  J={cfg.J}  K={cfg.K}  epochs={cfg.epochs}  "
        f"batch={cfg.batch_size}  device={cfg.device}"
    )
    print(f"        trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    t0 = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_res = 0.0
        epoch_uerr = 0.0
        nbatches = max(1, cfg.samples_per_epoch // cfg.batch_size)

        for _ in range(nbatches):
            a, f, u_exact = make_batch(
                grid,
                model.ops,
                batch_size=cfg.batch_size,
                contrast=cfg.contrast,
                rng=rng,
                device=device,
                dtype=torch.float64,
            )
            u_pred = model(a, f)
            res = model.residual(u_pred, a, f)

            f_norm = model.ops.l2_norm(f).clamp_min(1e-14)
            u_norm = model.ops.l2_norm(u_exact).clamp_min(1e-14)
            rel_r = model.ops.l2_norm(res) / f_norm
            rel_u = model.ops.l2_norm(u_pred - u_exact) / u_norm

            loss = cfg.lambda_r * torch.mean(rel_r**2) + cfg.lambda_u * torch.mean(rel_u**2)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            epoch_loss += float(loss.detach())
            epoch_res += float(rel_r.mean().detach())
            epoch_uerr += float(rel_u.mean().detach())

        epoch_loss /= nbatches
        epoch_res /= nbatches
        epoch_uerr /= nbatches

        if epoch == 1 or epoch % 20 == 0 or epoch == cfg.epochs:
            print(
                f"  epoch {epoch:4d}  loss={epoch_loss:.3e}  "
                f"rel_res={epoch_res:.3e}  rel_u={epoch_uerr:.3e}"
            )

    elapsed = time.perf_counter() - t0
    print(f"[train] done in {elapsed:.1f}s")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": cfg.__dict__,
                "state_dict": model.state_dict(),
                "lambdas": model.bank.lambdas.detach().cpu().numpy(),
            },
            save_path,
        )
        print(f"[train] saved {save_path}")

    return model


@torch.no_grad()
def evaluate(
    model: LGFCNN,
    n_eval: int = 16,
    contrast: float = 0.5,
    seed: int = 123,
    richardson_steps: int | None = None,
) -> None:
    """Compare LGF-CNN vs Richardson (same K) and PCG on held-out batches."""
    device = model.bank.lambdas.device
    grid = model.grid
    rng = np.random.default_rng(seed)
    K = model.K
    if richardson_steps is None:
        richardson_steps = K

    model.eval()
    a, f, u_exact = make_batch(
        grid,
        model.ops,
        batch_size=n_eval,
        contrast=contrast,
        rng=rng,
        device=device,
        dtype=torch.float64,
    )
    u_nn = model(a, f)
    res_nn = model.residual(u_nn, a, f)
    rel_r_nn = (model.ops.l2_norm(res_nn) / model.ops.l2_norm(f).clamp_min(1e-14)).mean()
    rel_u_nn = (
        model.ops.l2_norm(u_nn - u_exact) / model.ops.l2_norm(u_exact).clamp_min(1e-14)
    ).mean()

    # Classical baselines on the same batch (numpy path)
    rel_r_rich = []
    rel_u_rich = []
    rel_r_pcg = []
    rel_u_pcg = []
    iters_pcg = []
    for b in range(n_eval):
        aa = a[b].cpu().numpy()
        ff = f[b].cpu().numpy()
        ue = u_exact[b].cpu().numpy()

        rich = solve_axy_richardson(
            aa, ff, grid, omega=1.0, tol=0.0, maxiter=richardson_steps
        )
        r_rich = ff - apply_div_a_grad(rich.u, aa, grid)
        rel_r_rich.append(l2_norm(r_rich, grid) / l2_norm(ff, grid))
        rel_u_rich.append(l2_norm(rich.u - ue, grid) / l2_norm(ue, grid))

        pcg = solve_axy_pcg(aa, ff, grid, tol=1e-10, maxiter=500)
        rel_r_pcg.append(pcg.residual_history[-1])
        rel_u_pcg.append(l2_norm(pcg.u - ue, grid) / l2_norm(ue, grid))
        iters_pcg.append(pcg.iterations)

    print("[eval] held-out random batch")
    print(f"  LGF-CNN (K={K}):          rel_res={rel_r_nn:.3e}  rel_u={rel_u_nn:.3e}")
    print(
        f"  Richardson (K={richardson_steps}):  rel_res={np.mean(rel_r_rich):.3e}  "
        f"rel_u={np.mean(rel_u_rich):.3e}"
    )
    print(
        f"  PCG (to 1e-10):         rel_res={np.mean(rel_r_pcg):.3e}  "
        f"rel_u={np.mean(rel_u_pcg):.3e}  mean_iters={np.mean(iters_pcg):.1f}"
    )

    # Fixed manufactured problem
    prob = make_problem("smooth_sine", n=grid.n, forcing="discrete")
    a_t = torch.as_tensor(prob.a[None], dtype=torch.float64, device=device)
    f_t = torch.as_tensor(prob.f[None], dtype=torch.float64, device=device)
    u_t = torch.as_tensor(prob.u_exact[None], dtype=torch.float64, device=device)
    u_nn = model(a_t, f_t)
    rel_u = float(
        (
            model.ops.l2_norm(u_nn - u_t) / model.ops.l2_norm(u_t).clamp_min(1e-14)
        ).mean()
    )
    rel_r = float(
        (
            model.ops.l2_norm(model.residual(u_nn, a_t, f_t))
            / model.ops.l2_norm(f_t).clamp_min(1e-14)
        ).mean()
    )
    rich = solve_axy_richardson(
        prob.a, prob.f, grid, omega=1.0, tol=0.0, maxiter=K
    )
    r_rich = prob.f - apply_div_a_grad(rich.u, prob.a, grid)
    print("[eval] manufactured smooth_sine (discrete f)")
    print(f"  LGF-CNN:     rel_res={rel_r:.3e}  rel_u={rel_u:.3e}")
    print(
        f"  Richardson:  rel_res={l2_norm(r_rich, grid) / l2_norm(prob.f, grid):.3e}  "
        f"rel_u={l2_norm(rich.u - prob.u_exact, grid) / l2_norm(prob.u_exact, grid):.3e}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=31)
    p.add_argument("--J", type=int, default=6)
    p.add_argument("--K", type=int, default=6)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--samples-per-epoch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--contrast", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--save",
        type=Path,
        default=Path("axy_lgf/checkpoints/lgf_cnn_axy.pt"),
    )
    p.add_argument("--eval-only", type=Path, default=None, help="load checkpoint and eval")
    args = p.parse_args()

    cfg = LGFCNNTrainConfig(
        n=args.n,
        J=args.J,
        K=args.K,
        hidden=args.hidden,
        epochs=args.epochs,
        batch_size=args.batch_size,
        samples_per_epoch=args.samples_per_epoch,
        lr=args.lr,
        contrast=args.contrast,
        seed=args.seed,
        device=args.device,
    )

    if args.eval_only is not None:
        ckpt = torch.load(args.eval_only, map_location=args.device, weights_only=False)
        grid = SquareGrid(n=ckpt["config"]["n"])
        model = LGFCNN(
            grid,
            J=ckpt["config"]["J"],
            K=ckpt["config"]["K"],
            hidden=ckpt["config"]["hidden"],
            dtype=torch.float64,
            device=torch.device(args.device),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.mixers.to(args.device)
        model.bank.to(args.device)
        evaluate(model, contrast=ckpt["config"].get("contrast", 0.5))
        return

    model = train(cfg, save_path=args.save)
    evaluate(model, contrast=cfg.contrast, seed=cfg.seed + 1)


if __name__ == "__main__":
    main()
