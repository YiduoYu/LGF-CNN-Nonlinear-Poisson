"""
CNN solver for Darcy-like PDE: -div(a(x,y) grad u) = f on [-1,1]^2.
Dirichlet BC: u = g on boundary.
Data: random a, f, g -> FD solver -> u_ref.
Training: physics-only, normalized flux-form PDE residual.
"""

import argparse, csv, math, os, random, tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# -- grid & FD residual (Darcy) -----------------------------------------------

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


def build_input(f_norm, g, bnd, a_norm, x, y):
    B = f_norm.shape[0]
    if bnd.shape[0] == 1:
        bnd = bnd.expand(B, -1, -1, -1)
    if x.shape[0] == 1:
        x = x.expand(B, -1, -1, -1)
        y = y.expand(B, -1, -1, -1)
    return torch.cat([f_norm, g, bnd, a_norm, x, y], dim=1)


def enforce_dirichlet(u_raw, g, bnd):
    """u = g on boundary, u_raw on interior (no positivity constraint)."""
    return bnd * g + (1. - bnd) * u_raw


def flux_divergence(u, a, h):
    """5-point flux-form FD: -div(a grad u).  a is a positive coefficient field."""
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


def pde_residual(u, f, a, h):
    return flux_divergence(u, a, h) - f[:, :, 1:-1, 1:-1]


# -- CNN ----------------------------------------------------------------------

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


# -- data generation ----------------------------------------------------------

def _smooth_field(N):
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


def _random_a(N, args):
    raw = _smooth_field(N)
    a = np.maximum(args.a_base + args.a_scale * raw, args.a_min)
    a[0, :] = a[-1, :] = a[:, 0] = a[:, -1] = args.a_min
    return a


def _random_f(N, f_scale):
    f = _smooth_field(N) * f_scale
    f[0, :] = f[-1, :] = f[:, 0] = f[:, -1] = 0.
    return f


def _random_g(N, g_scale):
    raw = _smooth_field(N)
    g = g_scale * raw
    g[1:-1, 1:-1] = 0.
    return g


def _assemble_darcy_system(a, g, f, h):
    """Sparse 5-point flux-form system for -div(a grad u) = f, Dirichlet BC u=g."""
    N = a.shape[0] - 2
    n = N * N

    A = sp.lil_matrix((n, n))
    b = np.zeros(n)
    h2 = 1.0 / (h * h)

    for i in range(1, N + 1):
        for j in range(1, N + 1):
            row = (i - 1) * N + (j - 1)
            ac = a[i, j]

            for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ni, nj = i + di, j + dj
                coeff = 0.5 * (ac + a[ni, nj])
                A[row, row] += coeff * h2
                if 1 <= ni <= N and 1 <= nj <= N:
                    A[row, (ni - 1) * N + (nj - 1)] -= coeff * h2
                else:
                    b[row] += coeff * h2 * g[ni, nj]

            b[row] += f[i, j]

    return A.tocsr(), b


def solve_darcy_fd(f, g, a, h):
    """Direct solve: -div(a grad u) = f, Dirichlet BC u = g."""
    A, b = _assemble_darcy_system(a, g, f, h)
    u_int = spla.spsolve(A, b)
    u = g.copy()
    u[1:-1, 1:-1] = u_int.reshape(a.shape[0] - 2, a.shape[1] - 2)
    return u


def _make_one_problem(N, device, args):
    a_np = _random_a(N, args)
    f_np = _random_f(N, args.f_scale)
    g_np = _random_g(N, args.g_scale)
    h = 2.0 / (N + 1)

    u_np = solve_darcy_fd(f_np, g_np, a_np, h)

    a_t = torch.tensor(a_np, dtype=torch.float32, device=device)
    f_t = torch.tensor(f_np, dtype=torch.float32, device=device)
    g_t = torch.tensor(g_np, dtype=torch.float32, device=device)
    u_t = torch.tensor(u_np, dtype=torch.float32, device=device)

    return (f_t.unsqueeze(0).unsqueeze(0),
            g_t.unsqueeze(0).unsqueeze(0),
            a_t.unsqueeze(0).unsqueeze(0),
            u_t.unsqueeze(0).unsqueeze(0),
            h)


def generate_dataset(N, n_samples, device, args):
    probs = [_make_one_problem(N, device, args) for _ in range(n_samples)]

    x, y, bnd, interior, _ = make_grid(N, device)
    return {
        "f": torch.cat([p[0] for p in probs]),
        "g": torch.cat([p[1] for p in probs]),
        "a": torch.cat([p[2] for p in probs]),
        "u_exact": torch.cat([p[3] for p in probs]),
        "mask": bnd,
        "interior_mask": interior,
        "x": x,
        "y": y,
        "h": probs[0][4],
    }


def compute_stats(data):
    fi = data["f"][:, :, 1:-1, 1:-1]
    ai = data["a"][:, :, 1:-1, 1:-1]
    return (
        float(fi.mean()), float(fi.std().clamp_min(1e-8)),
        float(ai.mean()), float(ai.std().clamp_min(1e-8)),
    )


# -- training -----------------------------------------------------------------

def _take_batch(data, idx):
    batch = {k: data[k][idx] for k in ("f", "g", "a", "u_exact")}
    batch.update({k: data[k][:1] for k in ("mask", "interior_mask", "x", "y")})
    batch["h"] = data["h"]
    return batch


def _forward(model, batch, stats):
    f_mean, f_std, a_mean, a_std = stats
    fn = (batch["f"] - f_mean) / f_std
    an = (batch["a"] - a_mean) / a_std
    inp = build_input(fn, batch["g"], batch["mask"], an, batch["x"], batch["y"])
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

    print(f"generating {args.train_samples} PDEs (N={args.N})")
    train_data = generate_dataset(args.N, args.train_samples, device, args)
    val_data = generate_dataset(args.N, args.val_samples, device, args)
    stats = compute_stats(train_data)

    with torch.no_grad():
        oracle = float(pde_residual(
            val_data["u_exact"], val_data["f"], val_data["a"],
            val_data["h"]).pow(2).mean().sqrt())
        print(f"val ref residual RMSE = {oracle:.4e}")

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

    print(f"device={device}  oracle={oracle:.4e}  "
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

            res = pde_residual(u_pred, batch["f"], batch["a"], batch["h"])
            denom = (batch["f"][:, :, 1:-1, 1:-1]
                     .pow(2).mean(dim=(1, 2, 3)).sqrt().clamp_min(1e-3))
            loss = (res / denom.view(-1, 1, 1, 1)).pow(2).mean()

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
                  f"resRMSE={val_m['resRMSE']:.4f}  "
                  f"relL2={val_m['relL2']:.4e}  linf={val_m['linf']:.4e}  "
                  f"gradRelL2={val_m['gradRelL2']:.4e}")

            history.append({
                "epoch": epoch,
                "loss": loss.detach().item(),
                "val_relL2": val_m["relL2"],
                "val_resRMSE": val_m["resRMSE"],
                "val_linf": val_m["linf"],
                "val_gradRelL2": val_m["gradRelL2"],
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


# -- evaluation ---------------------------------------------------------------

@torch.no_grad()
def evaluate(model, data, stats, batch_size, args):
    model.eval()
    n = data["f"].shape[0]
    dev = data["f"].device
    H, W = data["f"].shape[-2], data["f"].shape[-1]
    ac = {"rl2": 0., "li": 0., "rs": 0., "rli": 0., "bm": 0., "gs": 0., "c": 0}

    for start in range(0, n, batch_size):
        idx = torch.arange(start, min(start + batch_size, n), device=dev)
        b = _take_batch(data, idx)

        up = _forward(model, b, stats)
        ue = b["u_exact"]
        interior = b["interior_mask"].expand_as(up)
        h = float(b["h"])

        err = (up - ue) * interior
        res = pde_residual(up, b["f"], b["a"], b["h"])

        m = up.shape[0]
        ac["c"] += m

        ac["rl2"] += float(
            (err.flatten(1).norm(dim=1)
             / ue.mul(interior).flatten(1).norm(dim=1).clamp_min(1e-12)).sum())
        ac["li"] = max(ac["li"], float(err.abs().max()))
        ac["rs"] += float(res.pow(2).sum())
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
        "resLinf": ac["rli"],
        "bmax": ac["bm"],
        "gradRelL2": ac["gs"] / c,
    }


def per_sample_metrics(u_pred, batch):
    ue = batch["u_exact"]
    interior = batch["interior_mask"].expand_as(u_pred)
    err = (u_pred - ue) * interior

    res = pde_residual(u_pred, batch["f"], batch["a"], batch["h"])
    res_e = pde_residual(ue, batch["f"], batch["a"], batch["h"])

    ie = err[:, :, 1:-1, 1:-1].flatten(1)
    iu = ue[:, :, 1:-1, 1:-1].flatten(1)
    rf = res.flatten(1)
    ref = res_e.flatten(1)
    fi = batch["f"][:, :, 1:-1, 1:-1].flatten(1)
    ai = batch["a"][:, :, 1:-1, 1:-1].flatten(1)
    rl2 = ie.norm(dim=1) / iu.norm(dim=1).clamp_min(1e-12)

    rows = []
    for i in range(u_pred.shape[0]):
        rows.append({
            "sample_id": i,
            "a_mean": float(ai[i].mean()),
            "a_min": float(ai[i].min()),
            "a_max": float(ai[i].max()),
            "relL2": float(rl2[i]),
            "RMSE": float(ie[i].pow(2).mean().sqrt()),
            "MAE": float(ie[i].abs().mean()),
            "Linf": float(ie[i].abs().max()),
            "pred_flux_resRMSE": float(rf[i].pow(2).mean().sqrt()),
            "ref_flux_resRMSE": float(ref[i].pow(2).mean().sqrt()),
            "f_RMSE": float(fi[i].pow(2).mean().sqrt()),
        })
    return rows


# -- visualization ------------------------------------------------------------

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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.suptitle("Training convergence")

    ax1.plot(epochs_list, [h["loss"] for h in history], "b-", label="loss")
    ax1.set_xlabel("epoch")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=.3)
    ax1.set_title("Training loss")

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

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -- sincos test equation -----------------------------------------------------

def _sincos_u_ref(x, y):
    return torch.sin(math.pi * x) * torch.cos(math.pi * y)


# -- prediction ---------------------------------------------------------------

def _make_sincos_dataset(N, n_samples, device, args):
    """Build test PDEs with u_ref = sin(pi*x)*cos(pi*y), f from Darcy FD."""
    x_t, y_t, bnd, interior, h = make_grid(N, device)

    samples = []
    for _ in range(n_samples):
        # constant positive coefficient
        a_np = np.full((N + 2, N + 2), 1.0, dtype=np.float32)
        a_t = torch.tensor(a_np, dtype=torch.float32, device=device)

        u_ref = _sincos_u_ref(x_t[0, 0], y_t[0, 0])
        u_ref = u_ref.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        g = bnd * u_ref
        f = torch.zeros_like(u_ref)
        f[:, :, 1:-1, 1:-1] = flux_divergence(u_ref, a_t.unsqueeze(0).unsqueeze(0), h)

        samples.append((f, g, a_t.unsqueeze(0).unsqueeze(0), u_ref))

    return {
        "f": torch.cat([s[0] for s in samples]),
        "g": torch.cat([s[1] for s in samples]),
        "a": torch.cat([s[2] for s in samples]),
        "u_exact": torch.cat([s[3] for s in samples]),
        "mask": bnd,
        "interior_mask": interior,
        "x": x_t,
        "y": y_t,
        "h": h,
    }


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
    predict_case = getattr(args, "predict_case", "random")

    if predict_case == "sincos":
        print(f"prediction (sincos): N={N}  samples={args.predict_samples}  "
              f"u_ref = sin(pi*x)*cos(pi*y)")
        random.seed(args.test_seed)
        np.random.seed(args.test_seed)
        batch = _make_sincos_dataset(N, args.predict_samples, device, args)
    else:
        print(f"prediction (random): N={N}  samples={args.predict_samples}")
        random.seed(args.test_seed)
        np.random.seed(args.test_seed)
        batch = generate_dataset(N, args.predict_samples, device, args)

    with torch.no_grad():
        u_pred = _forward(model, batch, stats)

    vm = evaluate(model, batch, stats, args.predict_samples, args)
    print(f"relL2={vm['relL2']:.4e}  linf={vm['linf']:.4e}  "
          f"resRMSE={vm['resRMSE']:.4f}  gradRelL2={vm['gradRelL2']:.4e}")

    rows = per_sample_metrics(u_pred, batch)

    if args.predict_save:
        p = Path(args.predict_save)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "f": batch["f"].cpu(),
            "g": batch["g"].cpu(),
            "a": batch["a"].cpu(),
            "u_exact": batch["u_exact"].cpu(),
            "u_pred": u_pred.cpu(),
            "h": float(batch["h"]),
            "metrics": vm,
            "config": {"N": N},
        }, p)

    if args.plot:
        out = Path(args.plot_dir)
        out.mkdir(parents=True, exist_ok=True)

        h = float(batch["h"])
        fext = [-1., 1., -1., 1.]
        res_extent = [-1. + h, 1. - h, -1. + h, 1. - h]

        for i, row in enumerate(rows):
            ui = batch["u_exact"][i, 0]
            up = u_pred[i, 0]

            pr = pde_residual(
                u_pred[i:i+1], batch["f"][i:i+1],
                batch["a"][i:i+1], h)[0, 0]
            er = pde_residual(
                batch["u_exact"][i:i+1], batch["f"][i:i+1],
                batch["a"][i:i+1], h)[0, 0]

            title = (f"sample {row['sample_id']}  "
                     f"relL2={row['relL2']:.3e}  "
                     f"pred_res={row['pred_flux_resRMSE']:.3e}")
            _save_panel(ui, up, up - ui, batch["f"][i, 0],
                        pr, er,
                        out / f"sample_{row['sample_id']:03d}_panel.png",
                        title, fext, res_extent)

        _write_csv(rows, out / "metrics.csv")
        print(f"saved -> {out}")


# -- CLI ----------------------------------------------------------------------

def ckpt_key(args):
    key = f"N{args.N}_e{args.epochs}"
    if getattr(args, "base_ch", 32) != 32:
        key += f"_c{args.base_ch}"
    if getattr(args, "n_layers", 6) != 6:
        key += f"_L{args.n_layers}"
    key += (f"_darcy_ab{args.a_base:g}_as{args.a_scale:g}_am{args.a_min:g}"
            f"_fs{args.f_scale:g}_gs{args.g_scale:g}"
            f"_tr{args.train_samples}")
    return key


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=32)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--train-samples", type=int, default=2048)
    p.add_argument("--val-samples", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--a-base", type=float, default=1.5)
    p.add_argument("--a-scale", type=float, default=0.5)
    p.add_argument("--a-min", type=float, default=0.1)
    p.add_argument("--f-scale", type=float, default=3.0)
    p.add_argument("--g-scale", type=float, default=0.5)
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--load-path", default="")
    p.add_argument("--predict", action="store_true")
    p.add_argument("--predict-samples", type=int, default=8)
    p.add_argument("--predict-save", default="")
    p.add_argument("--predict-N", type=int, default=None)
    p.add_argument("--test-seed", type=int, default=1234)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--plot-dir", default="results/figures")
    p.add_argument("--predict-case", default="random",
                   choices=["random", "sincos"])
    return p.parse_args()


def main():
    args = parse_args()
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
