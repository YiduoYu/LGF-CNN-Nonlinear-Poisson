"""
CNN solver for nonlinear flux-form Poisson equation: -div(u^k grad u) = f on [-1,1]^2.
Dirichlet BC: u = g on boundary.

ROUTE A: manufactured-solution (MMS) sanity check.
  - Generates positive exact solutions u_exact(x,y).
  - Computes f = -div_h(u_exact^k grad_h u_exact) via discrete FD operator.
  - Verifies residual near zero for u_exact.
  - Trains CNN to overfit on a tiny dataset.

  Equation: -div(u^k grad u) = f,  u > 0,  k > 0.
  No external a(x,y) field — the diffusivity is u^k itself.
"""

import argparse, csv, math, os, random, tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F


# -- grid -------------------------------------------------------------------

def make_grid(N, device="cpu", dtype=torch.float32):
    H = W = N + 2
    h = 2.0 / (N + 1)
    xs = torch.linspace(-1., 1., W, device=device, dtype=dtype)
    ys = torch.linspace(-1., 1., H, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    x = xx[None, None, :, :]
    y = yy[None, None, :, :]
    bnd = torch.zeros(1, 1, H, W, device=device, dtype=dtype)
    bnd[:, :,  0, :] = 1.
    bnd[:, :, -1, :] = 1.
    bnd[:, :,  :,  0] = 1.
    bnd[:, :,  :, -1] = 1.
    return x, y, bnd, 1. - bnd, h


def enforce_dirichlet(u_raw, g, bnd):
    """u = g on boundary, u_raw on interior."""
    return bnd * g + (1. - bnd) * u_raw


# -- nonlinear flux-form operator -------------------------------------------

def power_coefficient(u, k, eps=1e-6):
    """Compute a = u^k.  k may be scalar or shape [B].  u assumed positive."""
    if isinstance(k, (int, float)):
        k = torch.tensor([k], device=u.device)
    if k.ndim == 1:
        k = k.view(-1, 1, 1, 1)
    return torch.clamp(u, min=eps).pow(k)


def nonlinear_flux_divergence(u, k, h, eps=1e-6):
    """5-point flux-form FD: -div(u^k grad u).
    Interface coefficient = arithmetic average of neighbour values.
    Args:
        u: [B, 1, H, W]  solution field
        k: [B] or scalar   exponent
        h: grid spacing
    Returns:
        [B, 1, H-2, W-2]  -div_h(u^k grad_h u) on interior
    """
    a = power_coefficient(u, k, eps)                    # a = u^k, [B,1,H,W]

    uc = u[:, :, 1:-1, 1:-1]
    ac = a[:, :, 1:-1, 1:-1]

    ae = 0.5 * (ac + a[:, :, 1:-1, 2:])
    aw = 0.5 * (ac + a[:, :, 1:-1, :-2])
    an = 0.5 * (ac + a[:, :, 2:, 1:-1])
    as_ = 0.5 * (ac + a[:, :, :-2, 1:-1])

    fe = ae * (u[:, :, 1:-1, 2:] - uc) / h
    fw = aw * (uc - u[:, :, 1:-1, :-2]) / h
    fn = an * (u[:, :, 2:, 1:-1] - uc) / h
    fs = as_ * (uc - u[:, :, :-2, 1:-1]) / h

    return -(fe - fw + fn - fs) / h


def nonlinear_pde_residual(u, f, k, h, eps=1e-6):
    """R(u) = -div_h(u^k grad_h u) - f  on interior nodes.  [B,1,N,N]."""
    return nonlinear_flux_divergence(u, k, h, eps) - f[:, :, 1:-1, 1:-1]


# -- CNN --------------------------------------------------------------------

class SimpleCNN(nn.Module):
    def __init__(self, in_ch=6, out_ch=1, hidden_ch=32, n_layers=6):
        super().__init__()
        layers = []
        for i in range(n_layers):
            c_in = in_ch if i == 0 else hidden_ch
            g = min(8, hidden_ch)
            while hidden_ch % g:
                g -= 1
            layers += [
                nn.Conv2d(c_in, hidden_ch, 3, padding=1, padding_mode="replicate"),
                nn.GroupNorm(g, hidden_ch),
                nn.GELU(),
            ]
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(hidden_ch, out_ch, 1)

    def forward(self, x):
        return self.head(self.body(x))


# -- manufactured solution --------------------------------------------------

def _smooth_field(N, seed=None):
    """Random smooth field on (N+2)x(N+2) grid for building MMS solutions."""
    if seed is not None:
        np.random.seed(seed)
    xs = np.linspace(-1, 1, N + 2)
    ys = np.linspace(-1, 1, N + 2)
    xx, yy = np.meshgrid(xs, ys)

    field = np.zeros((N + 2, N + 2))
    for _ in range(np.random.randint(2, 5)):
        kx = np.random.randint(1, 4)
        ky = np.random.randint(1, 4)
        amp = np.random.uniform(-1, 1)
        px = np.random.uniform(0, 2 * np.pi)
        py = np.random.uniform(0, 2 * np.pi)
        field += amp * np.sin(kx * np.pi * xx + px) * np.sin(ky * np.pi * yy + py)

    return field / max(np.abs(field).max(), 1e-12)


def manufactured_u(x, y):
    """Generate a positive exact solution u(x,y) > 0 on [-1,1]^2.

    Formula:
        phi = (1 - x^2)(1 - y^2)              # zero on boundary
        boundary_part = 1.0 + c1*x + c2*y + c3*x*y
        interior_part = phi * smooth_sine
        u = boundary_part + 0.15 * interior_part

    Random coefficients c1,c2,c3 per call ensure variety.
    Returns u with min(u) > 0.3.
    """
    x_np = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
    y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else y
    N = x_np.shape[-1] - 2

    phi = (1. - x_np ** 2) * (1. - y_np ** 2)           # zero on boundary

    c1 = np.random.uniform(-0.5, 0.5)
    c2 = np.random.uniform(-0.5, 0.5)
    c3 = np.random.uniform(-0.5, 0.5)

    boundary_part = 1.0 + c1 * x_np + c2 * y_np + c3 * x_np * y_np

    interior_field = _smooth_field(N)
    interior_part = phi * interior_field

    u = boundary_part + 0.15 * interior_part

    # Safety clip to guarantee positivity
    u = np.maximum(u, 0.3)

    if isinstance(x, torch.Tensor):
        return torch.from_numpy(u.astype(np.float32)).to(device=x.device)
    return u


# -- MMS dataset ------------------------------------------------------------


def build_input(f_norm, g, bnd, k_norm, x, y):
    """Build 6-channel CNN input from normalized fields.

    Channels: [f_norm, g, boundary_mask, k_channel, x, y]
    """
    B = f_norm.shape[0]
    if bnd.shape[0] == 1:
        bnd = bnd.expand(B, -1, -1, -1)
    if x.shape[0] == 1:
        x = x.expand(B, -1, -1, -1)
        y = y.expand(B, -1, -1, -1)
    # k_norm: [B] or scalar -> [B, 1, 1, 1] broadcast to [B, 1, H, W]
    if k_norm.ndim == 1:
        k_ch = k_norm.view(-1, 1, 1, 1).expand(B, 1, *bnd.shape[-2:])
    else:
        k_ch = torch.full(
            (B, 1, *bnd.shape[-2:]), float(k_norm),
            device=bnd.device, dtype=bnd.dtype)
    return torch.cat([f_norm, g, bnd, k_ch, x, y], dim=1)


def make_mms_dataset(N, n_samples, device, args):
    """Build MMS dataset: u_exact manufactured, f computed via discrete FD.

    Equation: -div(u^k grad u) = f,  u = g on boundary.
    """
    x_t, y_t, bnd, interior, h = make_grid(N, device)
    H = W = N + 2

    f_list, g_list, k_list, u_list = [], [], [], []

    for i in range(n_samples):
        # Sample k
        if getattr(args, "random_k", False):
            k_i = float(np.random.uniform(args.k_min, args.k_max))
        else:
            k_i = float(args.k)

        # Generate positive exact solution
        u_t = manufactured_u(x_t[0, 0], y_t[0, 0])
        u_t = u_t.unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)

        # Boundary data
        g_t = bnd * u_t                        # exact on boundary, 0 interior

        # Compute f via discrete operator
        k_t = torch.tensor([k_i], device=device)
        f_t = torch.zeros_like(u_t)
        f_t[:, :, 1:-1, 1:-1] = nonlinear_flux_divergence(u_t, k_t, h)

        f_list.append(f_t)
        g_list.append(g_t)
        k_list.append(k_t)
        u_list.append(u_t)

    return {
        "f": torch.cat(f_list),                # [B, 1, H, W]
        "g": torch.cat(g_list),                # [B, 1, H, W]
        "k": torch.cat(k_list),                # [B]
        "u_exact": torch.cat(u_list),           # [B, 1, H, W]
        "mask": bnd,
        "interior_mask": interior,
        "x": x_t,
        "y": y_t,
        "h": h,
    }


def compute_stats(data):
    """Compute z-score stats for f (interior) and k."""
    fi = data["f"][:, :, 1:-1, 1:-1]           # [B, 1, N, N]
    k = data["k"]                                # [B]
    return (
        float(fi.mean()), float(fi.std().clamp_min(1e-8)),
        float(k.mean()), float(k.std().clamp_min(1e-8)),
    )


# -- training ---------------------------------------------------------------

def _take_batch(data, idx):
    batch = {k: data[k][idx] for k in ("f", "g", "k", "u_exact")}
    batch.update({k: data[k][:1] for k in ("mask", "interior_mask", "x", "y")})
    batch["h"] = data["h"]
    return batch


def _forward(model, batch, stats):
    f_mean, f_std, k_mean, k_std = stats
    fn = (batch["f"] - f_mean) / f_std
    kn = (batch["k"] - k_mean) / k_std
    inp = build_input(fn, batch["g"], batch["mask"], kn, batch["x"], batch["y"])
    return enforce_dirichlet(model(inp), batch["g"], batch["mask"])


def _select_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(args, best_path, last_path, history_path, convergence_path, model=None):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = _select_device(args.device)

    print(f"Equation: -div(u^k grad u) = f  |  k = {args.k}")
    if getattr(args, "random_k", False):
        print(f"  random k in [{args.k_min}, {args.k_max}]")

    print(f"generating {args.train_samples} MMS samples (N={args.N})")
    train_data = make_mms_dataset(args.N, args.train_samples, device, args)
    val_data = make_mms_dataset(args.N, args.val_samples, device, args)
    stats = compute_stats(train_data)
    print(f"f stats: mean={stats[0]:.4e}  std={stats[1]:.4e}  "
          f"k stats: mean={stats[2]:.4e}  std={stats[3]:.4e}")

    # Verify MMS residual
    with torch.no_grad():
        exact_res = float(nonlinear_pde_residual(
            val_data["u_exact"], val_data["f"], val_data["k"],
            val_data["h"]).pow(2).mean().sqrt())
        u_min = float(val_data["u_exact"].min())
        u_max = float(val_data["u_exact"].max())
        f_vals = val_data["f"][:, :, 1:-1, 1:-1]
        bnd_err = float(
            ((val_data["u_exact"] - val_data["g"]).abs()
             * val_data["mask"].expand_as(val_data["u_exact"])).max())
        print(f"MMS exact residual RMSE = {exact_res:.4e}  "
              f"(should be near machine precision)")
        print(f"u_exact range: [{u_min:.4f}, {u_max:.4f}]")
        print(f"f range: [{float(f_vals.min()):.4e}, {float(f_vals.max()):.4e}]")
        print(f"boundary error = {bnd_err:.2e}  (should be zero)")
        oracle = exact_res

    n_layers = getattr(args, "n_layers", 6)
    if model is None:
        model = SimpleCNN(hidden_ch=args.base_ch, n_layers=n_layers).to(device)
    else:
        model = model.to(device)

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if n.endswith(".bias") or p.ndim <= 1 else decay).append(p)

    opt = torch.optim.AdamW([
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.},
    ], lr=args.lr)

    steps_per_ep = math.ceil(args.train_samples / args.batch_size)
    total_steps = steps_per_ep * args.epochs
    warmup = max(1, int(0.05 * total_steps))

    def lr_fn(s):
        if s < warmup:
            return s / warmup
        return 0.5 * (1. + math.cos(
            math.pi * (s - warmup) / max(1, total_steps - warmup)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

    print(f"Training tiny MMS dataset for overfit sanity check")
    print(f"device={device}  oracle={oracle:.4e}  "
          f"lambda_sup={args.lambda_sup}  lambda_pde={args.lambda_pde}  "
          f"steps/epoch={steps_per_ep}  total={total_steps}")

    best_res = math.inf
    eval_bs = args.eval_batch_size or args.batch_size
    history = []

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(args.train_samples, device=device)

        for start in range(0, args.train_samples, args.batch_size):
            batch = _take_batch(train_data, perm[start:start + args.batch_size])
            u_pred = _forward(model, batch, stats)

            interior = batch["interior_mask"].expand_as(u_pred)
            err = (u_pred - batch["u_exact"]) * interior
            loss_sup = err.pow(2).mean()

            res = nonlinear_pde_residual(
                u_pred, batch["f"], batch["k"], batch["h"])
            h = batch["h"]
            loss_pde = (h * h * res).pow(2).mean()

            loss = args.lambda_sup * loss_sup + args.lambda_pde * loss_pde

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            val_m = evaluate(model, val_data, stats, eval_bs, args)
            if val_m["resRMSE"] < best_res:
                best_res = val_m["resRMSE"]
                best_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "stats": stats,
                    "args": vars(args),
                    "epoch": epoch,
                    "best_val_resRMSE": best_res,
                }, best_path)

            print(f"epoch {epoch:03d}  loss={loss.detach():.3e}  "
                  f"sup={loss_sup.detach():.3e}  pde={loss_pde.detach():.3e}  "
                  f"resRMSE={val_m['resRMSE']:.4f}  "
                  f"relL2={val_m['relL2']:.4e}  linf={val_m['linf']:.4e}  "
                  f"gradRelL2={val_m['gradRelL2']:.4e}")

            history.append({
                "epoch": epoch,
                "loss": loss.detach().item(),
                "loss_sup": loss_sup.detach().item(),
                "loss_pde": loss_pde.detach().item(),
                "val_relL2": val_m["relL2"],
                "val_resRMSE": val_m["resRMSE"],
                "val_linf": val_m["linf"],
                "val_gradRelL2": val_m["gradRelL2"],
                "val_exactResRMSE": val_m.get("exactResRMSE", 0),
            })

    for p, d in [
        (last_path, {"model_state_dict": model.state_dict(),
                      "stats": stats, "args": vars(args)}),
        (history_path, None),
        (convergence_path, None),
    ]:
        p.parent.mkdir(parents=True, exist_ok=True)
        if d:
            torch.save(d, p)

    _write_csv(history, history_path)
    _save_convergence_plot(history, convergence_path, oracle)
    return model, oracle


# -- evaluation -------------------------------------------------------------

@torch.no_grad()
def evaluate(model, data, stats, batch_size, args):
    model.eval()
    n = data["f"].shape[0]
    dev = data["f"].device
    H, W = data["f"].shape[-2], data["f"].shape[-1]
    ac = {"rl2": 0., "li": 0., "rs": 0., "rli": 0., "bm": 0., "gs": 0.,
          "ers": 0., "c": 0}

    for start in range(0, n, batch_size):
        idx = torch.arange(start, min(start + batch_size, n), device=dev)
        b = _take_batch(data, idx)

        up = _forward(model, b, stats)
        ue = b["u_exact"]
        interior = b["interior_mask"].expand_as(up)
        h = float(b["h"])

        err = (up - ue) * interior
        res = nonlinear_pde_residual(up, b["f"], b["k"], b["h"])
        res_e = nonlinear_pde_residual(ue, b["f"], b["k"], b["h"])

        m = up.shape[0]
        ac["c"] += m

        ac["rl2"] += float(
            (err.flatten(1).norm(dim=1)
             / ue.mul(interior).flatten(1).norm(dim=1).clamp_min(1e-12)).sum())
        ac["li"] = max(ac["li"], float(err.abs().max()))
        ac["rs"] += float(res.pow(2).sum())
        ac["ers"] += float(res_e.pow(2).sum())
        ac["rli"] = max(ac["rli"], float(res.abs().max()))
        ac["bm"] = max(ac["bm"],
                       float(((up - b["g"]).abs()
                              * b["mask"].expand_as(up)).max()))

        du = lambda v: (v[:, :, 1:-1, 2:] - v[:, :, 1:-1, :-2]) / (2. * h)
        dv = lambda v: (v[:, :, 2:, 1:-1] - v[:, :, :-2, 1:-1]) / (2. * h)
        pg = torch.cat([du(up), dv(up)], 1)
        tg = torch.cat([du(ue), dv(ue)], 1)
        ac["gs"] += float(
            ((pg - tg).flatten(1).norm(dim=1)
             / tg.flatten(1).norm(dim=1).clamp_min(1e-12)).sum())

    c = ac["c"]
    ni = c * (H - 2) * (W - 2)
    return {
        "relL2": ac["rl2"] / c,
        "linf": ac["li"],
        "resRMSE": math.sqrt(ac["rs"] / ni),
        "exactResRMSE": math.sqrt(ac["ers"] / ni),
        "resLinf": ac["rli"],
        "bmax": ac["bm"],
        "gradRelL2": ac["gs"] / c,
    }


def per_sample_metrics(u_pred, batch):
    ue = batch["u_exact"]
    interior = batch["interior_mask"].expand_as(u_pred)
    err = (u_pred - ue) * interior

    res = nonlinear_pde_residual(u_pred, batch["f"], batch["k"], batch["h"])
    res_e = nonlinear_pde_residual(ue, batch["f"], batch["k"], batch["h"])

    ie = err[:, :, 1:-1, 1:-1].flatten(1)
    iu = ue[:, :, 1:-1, 1:-1].flatten(1)
    rf = res.flatten(1)
    ref = res_e.flatten(1)
    fi = batch["f"][:, :, 1:-1, 1:-1].flatten(1)
    rl2 = ie.norm(dim=1) / iu.norm(dim=1).clamp_min(1e-12)

    rows = []
    for i in range(u_pred.shape[0]):
        k_i = batch["k"][i].item()
        rows.append({
            "sample_id": i,
            "k": float(k_i),
            "u_min": float(ue[i].min()),
            "u_max": float(ue[i].max()),
            "relL2": float(rl2[i]),
            "RMSE": float(ie[i].pow(2).mean().sqrt()),
            "MAE": float(ie[i].abs().mean()),
            "Linf": float(ie[i].abs().max()),
            "pred_flux_resRMSE": float(rf[i].pow(2).mean().sqrt()),
            "ref_flux_resRMSE": float(ref[i].pow(2).mean().sqrt()),
            "f_RMSE": float(fi[i].pow(2).mean().sqrt()),
        })
    return rows


# -- visualization ----------------------------------------------------------

def _write_csv(rows, path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _save_panel(ue, up, err, fm, rp, re, path, title, fext, rext, dpi=200):
    os.environ.setdefault("MPLCONFIGDIR",
                          str(Path(tempfile.gettempdir()) / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def sym(v):
        mx = max(float(v.abs().max().cpu()), 1e-12)
        return -mx, mx

    umin = min(float(ue.min()), float(up.min()))
    umax = max(float(ue.max()), float(up.max()))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    fig.suptitle(title)

    panels = [
        (0, 0, ue,  "reference u",            "viridis",  fext, umin, umax),
        (0, 1, up,  "CNN prediction",          "viridis",  fext, umin, umax),
        (0, 2, err, "signed error",            "coolwarm", fext, *sym(err)),
        (1, 0, fm,  "forcing f",               "coolwarm", fext, *sym(fm)),
        (1, 1, rp,  "pred PDE residual",       "coolwarm", rext, *sym(rp)),
        (1, 2, re,  "FD residual of ref u",    "coolwarm", rext, *sym(re)),
    ]
    for r, c, v, label, cmap, ext, v0, v1 in panels:
        ax = axes[r][c]
        im = ax.imshow(v.detach().cpu().numpy(), origin="lower", extent=ext,
                       cmap=cmap, vmin=v0, vmax=v1, aspect="equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(label)
        fig.colorbar(im, ax=ax, fraction=.046, pad=.04)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _save_convergence_plot(history, path, oracle=0.):
    if len(history) < 2:
        return

    os.environ.setdefault("MPLCONFIGDIR",
                          str(Path(tempfile.gettempdir()) / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs_list = [h["epoch"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    fig.suptitle("Training convergence — Nonlinear MMS")

    ax1 = axes[0]
    ax1.plot(epochs_list, [h["loss"] for h in history], "b-", label="total")
    ax1.plot(epochs_list, [h["loss_sup"] for h in history], "r--", label="sup")
    ax1.plot(epochs_list, [h["loss_pde"] for h in history], "g--", label="pde")
    ax1.set_xlabel("epoch")
    ax1.set_yscale("log")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=.3)
    ax1.set_title("Training loss")

    ax2 = axes[1]
    ax2b = ax2.twinx()
    l3, = ax2.plot(epochs_list, [h["val_relL2"] for h in history],
                   "g-", label="val relL2")
    l4, = ax2b.plot(epochs_list, [h["val_resRMSE"] for h in history],
                    "m--", label="val resRMSE")
    handles = [l3, l4]
    if oracle > 0:
        l5 = ax2b.axhline(oracle, color="gray", ls=":",
                          label=f"oracle ({oracle:.4f})")
        handles.append(l5)
    ax2.set_xlabel("epoch")
    ax2.legend(handles=handles, loc="upper right")
    ax2.grid(True, alpha=.3)
    ax2.set_title("Validation metrics")

    ax3 = axes[2]
    ax3.plot(epochs_list, [h["val_linf"] for h in history], "c-", label="linf")
    ax3.set_xlabel("epoch")
    ax3.set_yscale("log")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=.3)
    ax3.set_title("Max error")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -- prediction -------------------------------------------------------------

def run_prediction(args):
    if not args.load_path:
        raise ValueError("--load-path required")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = _select_device(args.device)

    ckpt = torch.load(args.load_path, map_location=device, weights_only=False)
    ckpt_args = argparse.Namespace(**ckpt["args"])
    model = SimpleCNN(
        hidden_ch=ckpt_args.base_ch,
        n_layers=getattr(ckpt_args, "n_layers", 6),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    stats = ckpt["stats"]

    N = args.predict_N or ckpt_args.N
    print(f"prediction (MMS): N={N}  samples={args.predict_samples}")

    random.seed(args.test_seed)
    np.random.seed(args.test_seed)
    batch = make_mms_dataset(N, args.predict_samples, device, args)

    with torch.no_grad():
        u_pred = _forward(model, batch, stats)

    vm = evaluate(model, batch, stats, args.predict_samples, args)
    print(f"relL2={vm['relL2']:.4e}  linf={vm['linf']:.4e}  "
          f"resRMSE={vm['resRMSE']:.4f}  exactResRMSE={vm['exactResRMSE']:.4e}  "
          f"gradRelL2={vm['gradRelL2']:.4e}")

    rows = per_sample_metrics(u_pred, batch)

    if args.plot:
        out = Path(args.plot_dir)
        out.mkdir(parents=True, exist_ok=True)

        h = float(batch["h"])
        fext = [-1., 1., -1., 1.]
        res_extent = [-1. + h, 1. - h, -1. + h, 1. - h]

        for i, row in enumerate(rows):
            ui = batch["u_exact"][i, 0]
            up = u_pred[i, 0]

            pr = nonlinear_pde_residual(
                u_pred[i:i+1], batch["f"][i:i+1],
                batch["k"][i:i+1], h)[0, 0]
            er = nonlinear_pde_residual(
                batch["u_exact"][i:i+1], batch["f"][i:i+1],
                batch["k"][i:i+1], h)[0, 0]

            title = (f"sample {row['sample_id']}  k={row['k']:.2f}  "
                     f"relL2={row['relL2']:.3e}  "
                     f"pred_res={row['pred_flux_resRMSE']:.3e}")
            _save_panel(ui, up, up - ui, batch["f"][i, 0],
                        pr, er,
                        out / f"sample_{row['sample_id']:03d}_panel.png",
                        title, fext, res_extent)

        _write_csv(rows, out / "metrics.csv")
        print(f"saved -> {out}")


# -- CLI --------------------------------------------------------------------

def ckpt_key(args):
    key = f"N{args.N}_e{args.epochs}"
    if getattr(args, "base_ch", 32) != 32:
        key += f"_c{args.base_ch}"
    if getattr(args, "n_layers", 6) != 6:
        key += f"_L{args.n_layers}"
    if getattr(args, "random_k", False):
        key += f"_nonlinear_k{args.k_min:g}_{args.k_max:g}_tr{args.train_samples}"
    else:
        key += f"_nonlinear_k{args.k:g}_tr{args.train_samples}"
    return key


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=32)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--train-samples", type=int, default=32)
    p.add_argument("--val-samples", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=1234)

    # Nonlinear PDE
    p.add_argument("--k", type=float, default=2.0,
                   help="exponent in u^k (diffusivity)")
    p.add_argument("--random-k", action="store_true",
                   help="sample k uniformly from [k_min, k_max]")
    p.add_argument("--k-min", type=float, default=0.5)
    p.add_argument("--k-max", type=float, default=3.0)

    # Loss weights
    p.add_argument("--lambda-sup", type=float, default=1.0,
                   help="weight on supervised loss")
    p.add_argument("--lambda-pde", type=float, default=0.1,
                   help="weight on PDE residual loss")

    # MMS sanity
    p.add_argument("--mms-sanity", action="store_true",
                   help="run MMS sanity check (print exact residual, then train)")

    p.add_argument("--retrain", action="store_true")
    p.add_argument("--load-path", default="")
    p.add_argument("--predict", action="store_true")
    p.add_argument("--predict-samples", type=int, default=8)
    p.add_argument("--predict-save", default="")
    p.add_argument("--predict-N", type=int, default=None)
    p.add_argument("--test-seed", type=int, default=1234)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--plot-dir", default="results/figures")
    return p.parse_args()


def main():
    args = parse_args()

    if args.mms_sanity:
        print("=" * 60)
        print("MMS SANITY CHECK")
        print("=" * 60)
        print(f"Equation: -div(u^k grad u) = f  |  k = {args.k}")
        print(f"Grid: N={args.N}  |  samples: {args.train_samples}")

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        device = _select_device(args.device)
        data = make_mms_dataset(args.N, args.train_samples, device, args)

        with torch.no_grad():
            res = nonlinear_pde_residual(
                data["u_exact"], data["f"], data["k"], data["h"])
            exact_rmse = float(res.pow(2).mean().sqrt())
            u_ex = data["u_exact"]
            bnd_err = float(
                ((u_ex - data["g"]).abs()
                 * data["mask"].expand_as(u_ex)).max())
            u_min, u_max = float(u_ex.min()), float(u_ex.max())
            fi = data["f"][:, :, 1:-1, 1:-1]
            f_min, f_max = float(fi.min()), float(fi.max())

        print(f"  exact_resRMSE   = {exact_rmse:.4e}  "
              f"(should be near machine precision)")
        print(f"  boundary error  = {bnd_err:.2e}  (should be zero)")
        print(f"  u_exact range   = [{u_min:.4f}, {u_max:.4f}]  "
              f"(should be > 0.3)")
        print(f"  f range         = [{f_min:.4e}, {f_max:.4e}]")

        if exact_rmse > 1e-8:
            print(f"  WARNING: exact residual RMSE = {exact_rmse:.4e} > 1e-8, "
                  f"check FD operator!")
        else:
            print(f"  OK — MMS residual is near machine precision.")

        print("=" * 60)

    if args.predict or args.load_path:
        return run_prediction(args)

    key = ckpt_key(args)
    ckpt_dir = Path("checkpoints") / "cnn" / key
    best_path = ckpt_dir / "best.pt"

    if best_path.exists() and not args.retrain:
        print(f"found {best_path}, use --retrain to retrain")
    else:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"training {key}")
        train(args, best_path,
              ckpt_dir / "last.pt",
              ckpt_dir / "history.csv",
              ckpt_dir / "convergence.png")

    if args.plot:
        pa = argparse.Namespace(**vars(args))
        pa.load_path = str(best_path)
        pa.predict_save = ""
        pa.plot = True
        pa.plot_dir = str(Path("results") / "cnn" / key / "figures")
        pa.predict_N = None
        pa.test_seed = args.seed
        run_prediction(pa)

    print(f"done: checkpoints/cnn/{key}")


if __name__ == "__main__":
    main()
