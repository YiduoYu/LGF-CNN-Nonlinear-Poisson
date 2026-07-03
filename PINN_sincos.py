"""
PINN solver for Poisson PDE: -laplacian(u) = f  on [-1,1]^2.
Dirichlet BC: u = g on boundary.
Exact reference: u(x,y) = sin(pi*x)*cos(pi*y)  →  f = 2*pi^2*sin(pi*x)*cos(pi*y).

Follows the Raissi-Perdikaris-Karniadakis architecture:
  - tanh-activated MLP, Xavier init
  - Adam warmup → L-BFGS fine-tuning
  - physics residual computed via autograd

Usage:
  python PINN_sincos.py                      # train + evaluate
  python PINN_sincos.py --predict            # load and evaluate
  python PINN_sincos.py --plot               # save figures
"""

import argparse, math, os, random, tempfile, csv
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Exact solution
# ---------------------------------------------------------------------------

def exact_u(x, y):
    """u(x,y) = sin(pi*x) * cos(pi*y)."""
    return torch.sin(math.pi * x) * torch.cos(math.pi * y)


def source_f(x, y):
    """f = -laplacian(u) = 2*pi^2 * sin(pi*x) * cos(pi*y)."""
    return 2.0 * math.pi ** 2 * torch.sin(math.pi * x) * torch.cos(math.pi * y)


# ---------------------------------------------------------------------------
# PINN model
# ---------------------------------------------------------------------------

class PINN(nn.Module):
    """Fully-connected network with tanh activations, per Raissi et al."""

    def __init__(self, layers):
        """
        Args:
            layers: list of ints, e.g. [2, 50, 50, 50, 50, 1].
        """
        super().__init__()
        self.lb = nn.Parameter(torch.tensor([-1., -1.]), requires_grad=False)
        self.ub = nn.Parameter(torch.tensor([1., 1.]), requires_grad=False)

        weights = []
        biases = []
        for i in range(len(layers) - 1):
            # Xavier uniform
            w = nn.Parameter(
                torch.empty(layers[i], layers[i + 1]))
            nn.init.xavier_uniform_(w)
            b = nn.Parameter(torch.zeros(1, layers[i + 1]))
            weights.append(w)
            biases.append(b)
        self.weights = nn.ParameterList(weights)
        self.biases = nn.ParameterList(biases)

    def forward(self, x, y):
        """Forward pass: (x, y) → u.  Inputs scaled to [-1, 1]."""
        X = torch.cat([x, y], dim=1)                         # (N, 2)
        # Scale to [-1, 1]
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for i, (w, b) in enumerate(zip(self.weights[:-1], self.biases[:-1])):
            H = torch.tanh(H @ w + b)
        return H @ self.weights[-1] + self.biases[-1]


def pinn_residual(model, x, y):
    """Compute physics residual: r = -laplacian(u_pred) - f(x,y)."""
    x_var = x.detach().requires_grad_(True)
    y_var = y.detach().requires_grad_(True)

    u = model(x_var, y_var)

    u_x = torch.autograd.grad(u, x_var, grad_outputs=torch.ones_like(u),
                              create_graph=True)[0]
    u_y = torch.autograd.grad(u, y_var, grad_outputs=torch.ones_like(u),
                              create_graph=True)[0]

    u_xx = torch.autograd.grad(u_x, x_var, grad_outputs=torch.ones_like(u_x),
                               create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y_var, grad_outputs=torch.ones_like(u_y),
                               create_graph=True)[0]

    laplacian_u = u_xx + u_yy
    f = source_f(x_var, y_var)
    return -laplacian_u - f


# ---------------------------------------------------------------------------
# Training data: sample collocation + boundary points
# ---------------------------------------------------------------------------

def latin_hypercube(n, d, lb, ub):
    """Simple LHS sampling in d dimensions."""
    pts = np.random.uniform(0, 1, (n, d))
    for j in range(d):
        pts[:, j] = (pts[:, j] + np.random.permutation(np.arange(n))) / n
    return lb + (ub - lb) * pts


def sample_points(N_f, N_b):
    """
    Returns:
        x_f, y_f: interior collocation points  (N_f, 1)
        x_b, y_b: boundary points              (N_b, 1)
        g_b:      exact u on boundary          (N_b, 1)
    """
    lb = np.array([-1., -1.])
    ub = np.array([1., 1.])

    # Interior collocation points via LHS
    interior = latin_hypercube(N_f, 2, lb, ub)
    x_f = torch.tensor(interior[:, 0:1], dtype=torch.float32)
    y_f = torch.tensor(interior[:, 1:2], dtype=torch.float32)

    # Boundary points: grid on four edges
    N_side = N_b // 4
    r = N_b % 4

    pts = []
    # bottom: y = -1, x in [-1, 1]
    pts.append(np.column_stack([
        np.linspace(-1, 1, N_side + (1 if r > 0 else 0)),
        np.full(N_side + (1 if r > 0 else 0), -1.)
    ]))
    # top: y = 1
    pts.append(np.column_stack([
        np.linspace(-1, 1, N_side + (1 if r > 1 else 0)),
        np.full(N_side + (1 if r > 1 else 0), 1.)
    ]))
    # left: x = -1
    pts.append(np.column_stack([
        np.full(N_side + (1 if r > 2 else 0), -1.),
        np.linspace(-1, 1, N_side + (1 if r > 2 else 0))
    ]))
    # right: x = 1
    pts.append(np.column_stack([
        np.full(N_side, 1.),
        np.linspace(-1, 1, N_side)
    ]))

    bnd_pts = np.vstack(pts)
    np.random.shuffle(bnd_pts)

    x_b = torch.tensor(bnd_pts[:, 0:1], dtype=torch.float32)
    y_b = torch.tensor(bnd_pts[:, 1:2], dtype=torch.float32)
    g_b = exact_u(x_b, y_b)

    return x_f, y_f, x_b, y_b, g_b


# ---------------------------------------------------------------------------
# Grid evaluation (for metrics, per Baseline.py)
# ---------------------------------------------------------------------------

def make_eval_grid(N):
    """Return (x, y, interior_mask) on a uniform (N+2)x(N+2) grid."""
    xs = torch.linspace(-1., 1., N + 2)
    ys = torch.linspace(-1., 1., N + 2)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    x_f = xx.reshape(-1, 1)
    y_f = yy.reshape(-1, 1)
    interior = torch.ones(N + 2, N + 2, dtype=torch.bool)
    interior[0, :] = interior[-1, :] = interior[:, 0] = interior[:, -1] = False
    return x_f, y_f, interior.reshape(-1)


@torch.no_grad()
def evaluate(model, N=32):
    """Evaluate on a grid: relL2, linf, resRMSE (per Baseline.py)."""
    x_f, y_f, interior_mask = make_eval_grid(N)
    H, W = N + 2, N + 2

    # Reshape to (N+2, N+2) for central differences
    u_pred = model(x_f, y_f).reshape(H, W)
    u_exact = exact_u(x_f, y_f).reshape(H, W)
    interior = interior_mask.reshape(H, W).float()

    err = (u_pred - u_exact) * interior
    relL2 = float(err.norm() / u_exact.mul(interior).norm().clamp_min(1e-12))

    linf = float(err.abs().max())

    # PDE residual on interior via central FD (matches Baseline.py)
    h = 2.0 / (N + 1)
    uc = u_pred[1:-1, 1:-1]
    ue_ = u_pred[1:-1, 2:]
    uw = u_pred[1:-1, :-2]
    un = u_pred[2:, 1:-1]
    us = u_pred[:-2, 1:-1]
    laplacian_fd = (ue_ + uw + un + us - 4.0 * uc) / (h * h)

    f_grid = source_f(x_f.reshape(H, W), y_f.reshape(H, W))[1:-1, 1:-1]
    res = -laplacian_fd - f_grid
    ni = interior_mask.sum()
    resRMSE = float(res.pow(2).mean().sqrt())

    # Gradient relative L2
    du = lambda v: (v[1:-1, 2:] - v[1:-1, :-2]) / (2.0 * h)
    dv = lambda v: (v[2:, 1:-1] - v[:-2, 1:-1]) / (2.0 * h)
    pg = torch.stack([du(u_pred), dv(u_pred)], dim=0)
    tg = torch.stack([du(u_exact), dv(u_exact)], dim=0)
    gradRelL2 = float(
        (pg - tg).norm() / tg.norm().clamp_min(1e-12))

    return {"relL2": relL2, "linf": linf, "resRMSE": resRMSE,
            "gradRelL2": gradRelL2, "bmax": 0.0}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    # Sample points
    x_f, y_f, x_b, y_b, g_b = sample_points(args.N_f, args.N_b)
    x_f, y_f = x_f.to(device), y_f.to(device)
    x_b, y_b, g_b = x_b.to(device), y_b.to(device), g_b.to(device)

    # Build network
    layers = [2] + [args.hidden_dim] * args.n_layers + [1]
    model = PINN(layers).to(device)
    print(f"layers = {layers}  |  N_f = {args.N_f}  N_b = {args.N_b}")

    ckpt_dir = Path(args.save_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _save_ckpt(name, state, extra=None):
        d = {
            "model_state_dict": state,
            "layers": layers,
            "args": vars(args),
        }
        if extra:
            d.update(extra)
        torch.save(d, ckpt_dir / name)

    # --- Phase 1: Adam warmup ---
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_loss = math.inf
    best_state = None

    for epoch in range(args.adam_epochs):
        opt.zero_grad()

        res = pinn_residual(model, x_f, y_f)
        loss_pde = res.pow(2).mean()
        u_b = model(x_b, y_b)
        loss_bc = (u_b - g_b).pow(2).mean()
        loss = loss_pde + args.bc_weight * loss_bc
        loss.backward()
        opt.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % max(1, args.adam_epochs // 10) == 0 or epoch == args.adam_epochs - 1:
            print(f"adam {epoch:5d}  |  loss={loss.item():.4e}  "
                  f"pde={loss_pde.item():.4e}  bc={loss_bc.item():.4e}")

        # Periodic checkpoint
        if (epoch + 1) % args.checkpoint_every == 0:
            _save_ckpt(f"adam_epoch{epoch+1:05d}.pt", model.state_dict(),
                       {"epoch": epoch + 1, "loss": loss.item()})

    if best_state:
        model.load_state_dict(best_state)
    _save_ckpt("adam_best.pt", best_state, {"loss": best_loss})
    print("saved adam_best.pt")

    # --- Phase 2: L-BFGS refinement ---
    print("running L-BFGS ...")
    opt_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=0.8, max_iter=args.lbfgs_max_iter,
        tolerance_grad=1e-12, tolerance_change=1e-14,
        history_size=50, line_search_fn="strong_wolfe",
    )

    def closure():
        opt_lbfgs.zero_grad()
        res = pinn_residual(model, x_f, y_f)
        loss_pde = res.pow(2).mean()
        u_b = model(x_b, y_b)
        loss_bc = (u_b - g_b).pow(2).mean()
        loss = loss_pde + args.bc_weight * loss_bc
        loss.backward()
        return loss

    opt_lbfgs.step(closure)

    # --- Evaluation ---
    print("evaluating ...")
    with torch.no_grad():
        val_m = {
            "train": evaluate(model, args.N),
            "fine": evaluate(model, args.N_fine or args.N),
        }
    for k, m in val_m.items():
        print(f"[{k}]  relL2={m['relL2']:.4e}  linf={m['linf']:.4e}  "
              f"resRMSE={m['resRMSE']:.4e}  gradRelL2={m['gradRelL2']:.4e}")

    _save_ckpt("lbfgs_final.pt", model.state_dict(), {"val": val_m})
    print(f"saved → {ckpt_dir / 'lbfgs_final.pt'}")

    # CSV log
    csv_path = Path(args.save_dir) / "history.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["N", "N_f", "N_b", "hidden_dim",
                                          "n_layers", "adam_epochs",
                                          "relL2", "linf", "resRMSE", "gradRelL2"])
        w.writeheader()
        w.writerow({"N": args.N, "N_f": args.N_f, "N_b": args.N_b,
                    "hidden_dim": args.hidden_dim, "n_layers": args.n_layers,
                    "adam_epochs": args.adam_epochs,
                    "relL2": val_m["fine"]["relL2"],
                    "linf": val_m["fine"]["linf"],
                    "resRMSE": val_m["fine"]["resRMSE"],
                    "gradRelL2": val_m["fine"]["gradRelL2"]})

    return model, val_m


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(model, N, out_dir):
    os.environ.setdefault("MPLCONFIGDIR",
                          str(Path(tempfile.gettempdir()) / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    x_f, y_f, _ = make_eval_grid(N)
    H = W = N + 2
    u_pred = model(x_f, y_f).reshape(H, W).detach()
    u_exact_ = exact_u(x_f, y_f).reshape(H, W)

    def sym(v):
        mx = max(float(v.abs().max()), 1e-12)
        return -mx, mx

    umin = min(float(u_pred.min()), float(u_exact_.min()))
    umax = max(float(u_pred.max()), float(u_exact_.max()))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    fig.suptitle(r"PINN: $-\nabla^2 u = f,\ u=\sin(\pi x)\cos(\pi y)$")

    panels = [
        (0, u_exact_, "Exact u", "viridis", umin, umax),
        (1, u_pred,  "PINN prediction", "viridis", umin, umax),
        (2, u_pred - u_exact_, "Error", "coolwarm", *sym(u_pred - u_exact_)),
    ]
    for idx, v, label, cmap, v0, v1 in panels:
        ax = axes[idx]
        im = ax.imshow(v.numpy(), origin="lower",
                       extent=[-1, 1, -1, 1], cmap=cmap, vmin=v0, vmax=v1)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title(label)
        fig.colorbar(im, ax=ax, fraction=.046, pad=.04)

    fig.savefig(out / "pinn_sincos.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {out / 'pinn_sincos.png'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=32,
                   help="grid resolution for evaluation (N+2)^2 points")
    p.add_argument("--N-f", type=int, default=10000,
                   help="interior collocation points")
    p.add_argument("--N-b", type=int, default=400,
                   help="boundary points")
    p.add_argument("--hidden-dim", type=int, default=50,
                   help="neurons per hidden layer")
    p.add_argument("--n-layers", type=int, default=4,
                   help="number of hidden layers")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Adam learning rate")
    p.add_argument("--adam-epochs", type=int, default=5000,
                   help="Adam iterations")
    p.add_argument("--lbfgs-max-iter", type=int, default=5000,
                   help="L-BFGS max iterations")
    p.add_argument("--bc-weight", type=float, default=1.0,
                   help="weight on boundary loss")
    p.add_argument("--checkpoint-every", type=int, default=500,
                   help="save checkpoint every N Adam epochs")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--predict", action="store_true",
                   help="load checkpoint and evaluate")
    p.add_argument("--load-path", type=str,
                   default="results/pinn_sincos/pinn_sincos.pt")
    p.add_argument("--save-dir", type=str, default="results/pinn_sincos")
    p.add_argument("--N-fine", type=int, default=None,
                   help="fine-grid N for evaluation (default: same as --N)")
    p.add_argument("--plot", action="store_true",
                   help="save comparison figure")
    p.add_argument("--plot-dir", type=str, default="results/pinn_sincos")
    args = p.parse_args()

    if args.predict:
        ckpt = torch.load(args.load_path, map_location="cpu", weights_only=False)
        layers = ckpt["layers"]
        model = PINN(layers)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        print(f"loaded {args.load_path}")
        N = args.N_fine or args.N
        m = evaluate(model, N)
        print(f"N={N}  relL2={m['relL2']:.4e}  linf={m['linf']:.4e}  "
              f"resRMSE={m['resRMSE']:.4e}  gradRelL2={m['gradRelL2']:.4e}")
        if args.plot:
            plot_results(model, N, args.plot_dir)
        return

    model, val_m = train(args)
    if args.plot:
        plot_results(model, args.N_fine or args.N, args.plot_dir)


if __name__ == "__main__":
    main()
