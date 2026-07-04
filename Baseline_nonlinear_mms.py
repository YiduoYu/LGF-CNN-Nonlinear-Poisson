"""
CNN solver for nonlinear flux-form Poisson equation: -div(u^k grad u) = f on [-1,1]^2.
Dirichlet BC: u = g on boundary.

ROUTE A: manufactured-solution (MMS) sanity check.
  - Generates positive exact solutions u_ref(x,y).
  - Computes f = -div_h(u_ref^k grad_h u_ref) via discrete FD operator.
  - Verifies residual near zero (oracle = 0).
  - Trains CNN to overfit on a tiny dataset.
"""

import argparse, csv, json, math, os, random, tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch, torch.nn as nn


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
    return bnd * g + (1. - bnd) * u_raw


# -- nonlinear flux-form operator -------------------------------------------

def nonlinear_flux_divergence(u, k, h, eps=1e-6):
    if isinstance(k, (int, float)):
        k = torch.tensor([k], device=u.device)
    if k.ndim == 1:
        k = k.view(-1, 1, 1, 1)
    a = torch.clamp(u, min=eps).pow(k)

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


# -- smooth field generator -------------------------------------------------

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


# -- manufactured solution --------------------------------------------------

def manufactured_u(x, y):
    x_np = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
    y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else y
    N = x_np.shape[-1] - 2

    phi = (1. - x_np ** 2) * (1. - y_np ** 2)

    c1 = np.random.uniform(-0.5, 0.5)
    c2 = np.random.uniform(-0.5, 0.5)
    c3 = np.random.uniform(-0.5, 0.5)

    boundary_part = 1.0 + c1 * x_np + c2 * y_np + c3 * x_np * y_np
    interior_field = _smooth_field(N)
    interior_part = phi * interior_field

    u = np.maximum(boundary_part + 0.15 * interior_part, 0.3)

    if isinstance(x, torch.Tensor):
        return torch.from_numpy(u.astype(np.float32)).to(device=x.device)
    return u


# -- MMS dataset ------------------------------------------------------------

def make_mms_dataset(N, n_samples, device, args):
    x_t, y_t, bnd, interior, h = make_grid(N, device)
    f_list, g_list, k_list, u_list = [], [], [], []

    eps = getattr(args, "coeff_eps", 1e-4)
    for _ in range(n_samples):
        if getattr(args, "random_k", False):
            k_i = float(np.random.uniform(args.k_min, args.k_max))
        else:
            k_i = float(args.k)

        u_t = manufactured_u(x_t[0, 0], y_t[0, 0])
        u_t = u_t.unsqueeze(0).unsqueeze(0)

        g_t = bnd * u_t
        k_t = torch.tensor([k_i], device=device)
        f_t = torch.zeros_like(u_t)
        f_t[:, :, 1:-1, 1:-1] = nonlinear_flux_divergence(u_t, k_t, h, eps)

        f_list.append(f_t)
        g_list.append(g_t)
        k_list.append(k_t)
        u_list.append(u_t)

    return {
        "f": torch.cat(f_list), "g": torch.cat(g_list),
        "k": torch.cat(k_list), "u_ref": torch.cat(u_list),
        "mask": bnd, "interior_mask": interior,
        "x": x_t, "y": y_t, "h": h,
    }


# -- input / stats / batching ------------------------------------------------

def build_input(f_norm, g, bnd, k_norm, x, y):
    B = f_norm.shape[0]
    bnd = bnd.expand(B, -1, -1, -1) if bnd.shape[0] == 1 else bnd
    x = x.expand(B, -1, -1, -1) if x.shape[0] == 1 else x
    y = y.expand(B, -1, -1, -1) if y.shape[0] == 1 else y
    k_ch = k_norm.view(B, 1, 1, 1).expand(-1, 1, *bnd.shape[-2:])
    return torch.cat([f_norm, g, bnd, k_ch, x, y], dim=1)


def compute_stats(data):
    fi = data["f"][:, :, 1:-1, 1:-1]
    k = data["k"]
    return (fi.mean().item(), fi.std(unbiased=False).clamp_min(1e-8).item(),
            k.mean().item(), k.std(unbiased=False).clamp_min(1e-8).item())


def _take_batch(data, idx):
    return {"h": data["h"],
            **{k: data[k][idx] for k in ("f", "g", "k", "u_ref")},
            **{k: data[k][:1] for k in ("mask", "interior_mask", "x", "y")}}


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


# -- training ---------------------------------------------------------------

def train(args, best_path, last_path, history_path, convergence_path, model=None):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = _select_device(args.device)

    print(f"Equation: -div(u^k grad u) = f  |  k = {args.k}  |  data=mms")
    if getattr(args, "random_k", False):
        print(f"  random k in [{args.k_min}, {args.k_max}]")

    print(f"generating {args.train_samples} MMS samples (N={args.N})")
    train_data = make_mms_dataset(args.N, args.train_samples, device, args)
    val_data = make_mms_dataset(args.N, args.val_samples, device, args)
    stats = compute_stats(train_data)
    print(f"f stats: mean={stats[0]:.4e}  std={stats[1]:.4e}  "
          f"k stats: mean={stats[2]:.4e}  std={stats[3]:.4e}")

    eps = getattr(args, "coeff_eps", 1e-4)
    with torch.no_grad():
        ref_res = float(nonlinear_pde_residual(
            val_data["u_ref"], val_data["f"], val_data["k"],
            val_data["h"], eps=eps).pow(2).mean().sqrt())
        u_min = float(val_data["u_ref"].min())
        u_max = float(val_data["u_ref"].max())
        f_vals = val_data["f"][:, :, 1:-1, 1:-1]
        bnd_err = float(
            ((val_data["u_ref"] - val_data["g"]).abs()
             * val_data["mask"].expand_as(val_data["u_ref"])).max())
        print(f"MMS residual RMSE = {ref_res:.4e}  (should be near 0)")
        print(f"u_ref range: [{u_min:.4f}, {u_max:.4f}]  "
              f"boundary error = {bnd_err:.2e}")
        oracle = ref_res

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

    n_train = train_data["f"].shape[0]
    n_val = val_data["f"].shape[0]
    print(f"device={device}  oracle={oracle:.4e}  "
          f"steps/epoch={steps_per_ep}  total={total_steps}  "
          f"n_train={n_train}  n_val={n_val}")

    best_metric = math.inf
    eval_bs = args.eval_batch_size or args.batch_size
    history = []

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)

        for start in range(0, n_train, args.batch_size):
            batch = _take_batch(train_data, perm[start:start + args.batch_size])
            u_pred = _forward(model, batch, stats)

            if args.loss == "supervised":
                interior = batch["interior_mask"].expand_as(u_pred)
                err = (u_pred - batch["u_ref"]) * interior
                loss = err.pow(2).mean()
            else:  # physics
                res = nonlinear_pde_residual(
                    u_pred, batch["f"], batch["k"], batch["h"], eps=eps)
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
            val_key = "relL2" if args.loss == "supervised" else "resRMSE"
            if val_m[val_key] < best_metric:
                best_metric = val_m[val_key]
                best_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "stats": stats,
                    "args": vars(args),
                    "epoch": epoch,
                    f"best_val_{val_key}": best_metric,
                }, best_path)

            print(f"epoch {epoch:03d}  loss={loss.detach():.3e}  "
                  f"resRMSE={val_m['resRMSE']:.4f}  "
                  f"relL2={val_m['relL2']:.4e}  linf={val_m['linf']:.4e}  "
                  f"gradRelL2={val_m['gradRelL2']:.4e}")

            history.append({
                "epoch": epoch,
                "loss": loss.detach().item(),
                "val_resRMSE": val_m["resRMSE"],
                "val_relL2": val_m["relL2"],
                "val_linf": val_m["linf"],
                "val_gradRelL2": val_m["gradRelL2"],
                "val_refResRMSE": val_m["refResRMSE"],
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
    _save_convergence_plot(history, convergence_path, oracle, args.loss)
    return model, oracle, best_metric


# -- evaluation -------------------------------------------------------------

@torch.no_grad()
def evaluate(model, data, stats, batch_size, args):
    model.eval()
    n = data["f"].shape[0]
    dev = data["f"].device
    H, W = data["f"].shape[-2:]
    eps = getattr(args, "coeff_eps", 1e-4)
    sum_rl2 = sum_rs = sum_ers = sum_gs = 0.
    max_li = max_rli = max_bm = 0.
    count = 0

    for start in range(0, n, batch_size):
        idx = torch.arange(start, min(start + batch_size, n), device=dev)
        b = _take_batch(data, idx)
        up = _forward(model, b, stats)
        ue = b["u_ref"]
        interior = b["interior_mask"].expand_as(up)
        h = float(b["h"])

        err = (up - ue) * interior
        res = nonlinear_pde_residual(up, b["f"], b["k"], b["h"], eps=eps)
        res_e = nonlinear_pde_residual(ue, b["f"], b["k"], b["h"], eps=eps)

        count += up.shape[0]
        sum_rl2 += (err.flatten(1).norm(dim=1)
                    / ue.mul(interior).flatten(1).norm(dim=1).clamp_min(1e-12)).sum().item()
        max_li = max(max_li, err.abs().max().item())
        sum_rs += res.pow(2).sum().item()
        sum_ers += res_e.pow(2).sum().item()
        max_rli = max(max_rli, res.abs().max().item())
        max_bm = max(max_bm, ((up - b["g"]).abs() * b["mask"].expand_as(up)).max().item())

        du = lambda v: (v[:, :, 1:-1, 2:] - v[:, :, 1:-1, :-2]) / (2. * h)
        dv = lambda v: (v[:, :, 2:, 1:-1] - v[:, :, :-2, 1:-1]) / (2. * h)
        pg = torch.cat([du(up), dv(up)], 1)
        tg = torch.cat([du(ue), dv(ue)], 1)
        sum_gs += ((pg - tg).flatten(1).norm(dim=1)
                   / tg.flatten(1).norm(dim=1).clamp_min(1e-12)).sum().item()

    ni = count * (H - 2) * (W - 2)
    return {"relL2": sum_rl2 / count, "linf": max_li,
            "resRMSE": math.sqrt(sum_rs / ni), "refResRMSE": math.sqrt(sum_ers / ni),
            "resLinf": max_rli, "bmax": max_bm, "gradRelL2": sum_gs / count}


def per_sample_metrics(u_pred, batch, eps):
    ue = batch["u_ref"]
    err = (u_pred - ue) * batch["interior_mask"].expand_as(u_pred)
    res = nonlinear_pde_residual(u_pred, batch["f"], batch["k"], batch["h"], eps=eps)
    res_e = nonlinear_pde_residual(ue, batch["f"], batch["k"], batch["h"], eps=eps)

    ie = err[:, :, 1:-1, 1:-1].flatten(1)
    iu = ue[:, :, 1:-1, 1:-1].flatten(1)
    rf, ref = res.flatten(1), res_e.flatten(1)
    fi = batch["f"][:, :, 1:-1, 1:-1].flatten(1)
    rl2 = ie.norm(dim=1) / iu.norm(dim=1).clamp_min(1e-12)

    return [{"sample_id": i, "k": batch["k"][i].item(),
             "relL2": rl2[i].item(), "RMSE": ie[i].pow(2).mean().sqrt().item(),
             "MAE": ie[i].abs().mean().item(), "Linf": ie[i].abs().max().item(),
             "pred_flux_resRMSE": rf[i].pow(2).mean().sqrt().item(),
             "ref_flux_resRMSE": ref[i].pow(2).mean().sqrt().item(),
             "f_RMSE": fi[i].pow(2).mean().sqrt().item()}
            for i in range(u_pred.shape[0])]


# -- visualization ----------------------------------------------------------

def _write_csv(rows, path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _save_panel(ue, up, err, fm, rp, re, path, title, fext, rext, dpi=200):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def sym(v):
        m = max(v.abs().max().cpu().item(), 1e-12)
        return -m, m

    umin = min(ue.min().item(), up.min().item())
    umax = max(ue.max().item(), up.max().item())

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    fig.suptitle(title)

    panels = [
        (0, 0, ue,  "u_ref",          "viridis",  fext, umin, umax),
        (0, 1, up,  "CNN prediction",  "viridis",  fext, umin, umax),
        (0, 2, err, "signed error",    "coolwarm", fext, *sym(err)),
        (1, 0, fm,  "forcing f",       "coolwarm", fext, *sym(fm)),
        (1, 1, rp,  "pred PDE res",    "coolwarm", rext, *sym(rp)),
        (1, 2, re,  "ref PDE res",     "coolwarm", rext, *sym(re)),
    ]
    for r, c, v, label, cmap, ext, v0, v1 in panels:
        ax = axes[r][c]
        im = ax.imshow(v.detach().cpu().numpy(), origin="lower", extent=ext,
                       cmap=cmap, vmin=v0, vmax=v1, aspect="equal")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title(label)
        fig.colorbar(im, ax=ax, fraction=.046, pad=.04)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _save_convergence_plot(history, path, oracle=0., loss_type="physics"):
    if len(history) < 2:
        return

    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs_list = [h["epoch"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.suptitle(f"Training convergence — MMS ({loss_type})")

    ax1.plot(epochs_list, [h["loss"] for h in history], "b-")
    ax1.set_xlabel("epoch"); ax1.set_yscale("log")
    ax1.set_title("Loss"); ax1.grid(True, alpha=.3)

    ax2b = ax2.twinx()
    l1, = ax2.plot(epochs_list, [h["val_relL2"] for h in history], "g-", label="relL2")
    l2, = ax2b.plot(epochs_list, [h["val_resRMSE"] for h in history], "m--", label="resRMSE")
    handles = [l1, l2]
    if oracle > 0:
        handles.append(ax2b.axhline(oracle, color="gray", ls=":", label=f"oracle {oracle:.1e}"))
    ax2.set_xlabel("epoch"); ax2.legend(handles=handles, loc="upper right")
    ax2.grid(True, alpha=.3); ax2.set_title("Validation")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -- prediction -------------------------------------------------------------

def run_prediction(args):
    if not args.predict_from:
        raise ValueError("--predict-from required")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = _select_device(args.device)

    ckpt = torch.load(args.predict_from, map_location=device, weights_only=False)
    ckpt_args = argparse.Namespace(**ckpt["args"])
    model = SimpleCNN(
        hidden_ch=ckpt_args.base_ch,
        n_layers=getattr(ckpt_args, "n_layers", 6),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    stats = ckpt["stats"]

    N = args.predict_N or ckpt_args.N
    data_args = argparse.Namespace(**vars(ckpt_args))
    data_args.device = args.device
    data_args.seed = args.test_seed

    random.seed(args.test_seed)
    np.random.seed(args.test_seed)

    print(f"prediction (MMS): N={N}  samples={args.predict_samples}")
    batch = make_mms_dataset(N, args.predict_samples, device, data_args)

    with torch.no_grad():
        u_pred = _forward(model, batch, stats)

    vm = evaluate(model, batch, stats, args.predict_samples, data_args)
    print(f"relL2={vm['relL2']:.4e}  linf={vm['linf']:.4e}  "
          f"resRMSE={vm['resRMSE']:.4f}  refRMSE={vm['refResRMSE']:.4e}  "
          f"gradRelL2={vm['gradRelL2']:.4e}")

    if args.predict_save:
        p = Path(args.predict_save)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "f": batch["f"].cpu(), "g": batch["g"].cpu(),
            "k": batch["k"].cpu(), "u_ref": batch["u_ref"].cpu(),
            "u_pred": u_pred.cpu(), "h": float(batch["h"]), "metrics": vm,
        }, p)
        print(f"saved predictions → {p}")

    eps = getattr(data_args, "coeff_eps", 1e-4)
    rows = per_sample_metrics(u_pred, batch, eps=eps)
    if args.plot:
        out = Path(args.plot_dir)
        out.mkdir(parents=True, exist_ok=True)

        h = float(batch["h"])
        fext = [-1., 1., -1., 1.]
        rext = [-1. + h, 1. - h, -1. + h, 1. - h]

        for i, row in enumerate(rows):
            ui = batch["u_ref"][i, 0]
            up = u_pred[i, 0]

            pr = nonlinear_pde_residual(
                u_pred[i:i+1], batch["f"][i:i+1],
                batch["k"][i:i+1], h, eps=eps)[0, 0]
            er = nonlinear_pde_residual(
                batch["u_ref"][i:i+1], batch["f"][i:i+1],
                batch["k"][i:i+1], h, eps=eps)[0, 0]

            _save_panel(ui, up, up - ui, batch["f"][i, 0], pr, er,
                        out / f"sample_{i:03d}_panel.png",
                        f"sample {i}  k={row['k']:.2f}  relL2={row['relL2']:.3e}",
                        fext, rext)

        _write_csv(rows, out / "metrics.csv")
        print(f"saved → {out}")


# -- CLI --------------------------------------------------------------------

def ckpt_key(args):
    if getattr(args, "random_k", False):
        k_str = f"k{args.k_min:g}-{args.k_max:g}"
    else:
        k_str = f"k{args.k:g}"
    n_layers = getattr(args, "n_layers", 6)
    loss_suffix = "_sup" if getattr(args, "loss", "physics") == "supervised" else ""
    return (f"N{args.N}_e{args.epochs}_tr{args.train_samples}"
            f"_{k_str}_ch{args.base_ch}_L{n_layers}{loss_suffix}_mms")


def _write_run_info(path, args, best_metric, oracle):
    k_str = (f"{args.k_min:.4g}–{args.k_max:.4g}" if getattr(args, "random_k", False)
             else f"{args.k:.4g}")
    metric_name = "relL2" if args.loss == "supervised" else "resRMSE"
    path.write_text(
        f"Experiment: CNN Nonlinear Poisson (MMS, loss={args.loss})\n"
        f"Key: {ckpt_key(args)}\n"
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"N: {args.N}  Epochs: {args.epochs}  k: {k_str}  Seed: {args.seed}\n"
        f"Train: {args.train_samples}  Val: {args.val_samples}\n"
        f"ch={args.base_ch}  L={getattr(args, 'n_layers', 6)}\n"
        f"Best val {metric_name}: {best_metric:.4e}  Oracle: {oracle:.4e}\n")


def _update_registry(base_dir, key, args, best_metric, oracle):
    reg_path = base_dir / "registry.json"
    try:
        reg = json.loads(reg_path.read_text()) if reg_path.exists() else {}
    except json.JSONDecodeError:
        reg = {}
    metric_key = "best_val_relL2" if args.loss == "supervised" else "best_val_resRMSE"
    reg.setdefault("cnn_mms", {})[key] = {
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "N": args.N, "epochs": args.epochs,
        "k": [args.k_min, args.k_max] if getattr(args, "random_k", False) else args.k,
        "train_samples": args.train_samples, "loss": args.loss,
        metric_key: best_metric, "oracle": oracle,
    }
    reg_path.write_text(json.dumps(reg, indent=2, ensure_ascii=False) + "\n")


def _link_latest(base_dir, key):
    latest = base_dir / "cnn_mms" / "latest"
    latest.unlink(missing_ok=True)
    latest.symlink_to(key)


def _archive_old_figures(figures_dir):
    if not figures_dir.exists():
        return
    old = [p for p in figures_dir.iterdir() if p.name != "history"]
    if not old:
        return
    archive = figures_dir / "history" / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    archive.mkdir(parents=True, exist_ok=True)
    for p in old:
        p.rename(archive / p.name)
    print(f"  archived {len(old)} files → {archive}")


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
    p.add_argument("--k", type=float, default=2.0)
    p.add_argument("--random-k", action="store_true")
    p.add_argument("--k-min", type=float, default=0.5)
    p.add_argument("--k-max", type=float, default=3.0)
    p.add_argument("--coeff-eps", type=float, default=1e-4)

    # Loss
    p.add_argument("--loss", default="physics",
                   choices=["physics", "supervised"],
                   help="physics: PDE residual only | supervised: |u_pred - u_ref|^2")

    # MMS sanity
    p.add_argument("--mms-sanity", action="store_true",
                   help="verify MMS residual, then exit")

    # Training control
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--predict-from", default="")
    p.add_argument("--predict", action="store_true")
    p.add_argument("--predict-samples", type=int, default=8)
    p.add_argument("--predict-save", default="")
    p.add_argument("--predict-N", type=int, default=None)
    p.add_argument("--test-seed", type=int, default=1234)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--plot-dir", default="results/figures")
    return p.parse_args()


# -- main -------------------------------------------------------------------

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

        eps = getattr(args, "coeff_eps", 1e-4)
        with torch.no_grad():
            res = nonlinear_pde_residual(
                data["u_ref"], data["f"], data["k"], data["h"], eps=eps)
            exact_rmse = float(res.pow(2).mean().sqrt())
            u_ex = data["u_ref"]
            bnd_err = float(
                ((u_ex - data["g"]).abs()
                 * data["mask"].expand_as(u_ex)).max())
            u_min, u_max = float(u_ex.min()), float(u_ex.max())
            fi = data["f"][:, :, 1:-1, 1:-1]

        print(f"  exact_resRMSE   = {exact_rmse:.4e}  (should be near 0)")
        print(f"  boundary error  = {bnd_err:.2e}  (should be 0)")
        print(f"  u_ref range     = [{u_min:.4f}, {u_max:.4f}]  (should be > 0.3)")
        print(f"  f range         = [{float(fi.min()):.4e}, {float(fi.max()):.4e}]")

        if exact_rmse > 1e-8:
            print(f"  WARNING: residual = {exact_rmse:.4e} > 1e-8, check FD!")
        else:
            print(f"  OK — MMS residual is near machine precision.")
        print("=" * 60)
        return

    if args.predict:
        return run_prediction(args)

    if args.epochs == 0:
        print("epochs=0, nothing to train.")
        return

    key = ckpt_key(args)
    base_dir = Path("checkpoints")
    ckpt_dir = base_dir / "cnn_mms" / key
    best_path = ckpt_dir / "best.pt"

    if best_path.exists() and not args.retrain:
        print(f"found {best_path}, use --retrain to retrain")
    else:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"training {key}")
        _, oracle, best_metric = train(args, best_path,
                                       ckpt_dir / "last.pt",
                                       ckpt_dir / "history.csv",
                                       ckpt_dir / "convergence.png")
        _write_run_info(ckpt_dir / "run_info.txt", args, best_metric, oracle)
        _update_registry(base_dir, key, args, best_metric, oracle)
        _link_latest(base_dir, key)
        print(f"saved → {ckpt_dir}")

    if args.plot:
        figures_dir = Path("results") / "cnn_mms" / key / "figures"
        _archive_old_figures(figures_dir)
        pa = argparse.Namespace(**vars(args))
        pa.predict_from = str(best_path)
        pa.predict_save = ""
        pa.plot = True
        pa.plot_dir = str(figures_dir)
        pa.predict_N = None
        pa.test_seed = args.seed
        run_prediction(pa)

    print(f"done: checkpoints/cnn_mms/{key}")


if __name__ == "__main__":
    main()
