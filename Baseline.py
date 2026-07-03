"""
CNN solver for -div(u^k grad u) = f on [-1,1]^2 with Dirichlet BC.

Training objective: normalized flux-form PDE residual only.
    R = -div_h(u_pred^k grad_h u_pred) - f
    loss = mean((R / RMS(f))^2)

u_exact is used ONLY for generating f, g and for evaluation metrics.
It never participates in the training loss.

Each training sample is a different random PDE (operator-learning style).

Usage:
    python Baseline.py                              # demo: N=32, 40 epochs
    python Baseline.py --N 32 --epochs 40 --retrain
    python Baseline.py --rhs-mode discrete          # debug mode (R_h=0)
"""

from __future__ import annotations

import argparse, csv, json, math, os, random, shutil, sys, tempfile, time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch, torch.nn as nn, torch.nn.functional as F

K_RANGE = (0.5, 2.5)
K_DEFAULT = K_RANGE  # backward compat for TransUNet

# ==============================================================================
# 1. Utilities
# ==============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def select_device(name: str) -> torch.device:
    if name != "auto": return torch.device(name)
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def run_id() -> str: return datetime.now().strftime("%Y%m%d_%H%M%S")
get_run_id = run_id  # backward compat for TransUNet

def _write_run_info(path: Path, info: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"Run: {info.get('run_id','unknown')}", f"Model: {info.get('model','unknown')}",
             f"Date: {datetime.now():%Y-%m-%d %H:%M:%S}"]
    for k in ("N","epochs","train_samples","k"):
        if k in info: lines.append(f"{k} = {info[k]}")
    path.write_text("\n".join(lines))

def _update_latest_symlink(link_dir: Path, target: str) -> None:
    link = link_dir / "latest"; link_dir.mkdir(parents=True, exist_ok=True)
    try:
        if link.is_symlink() or link.exists(): link.unlink()
        link.symlink_to(target, target_is_directory=True)
    except OSError: link.with_suffix(".txt").write_text(target)

# ==============================================================================
# 2. Config-key storage
# ==============================================================================

def config_key(N: int, epochs: int, k: float, base_ch: int = 32,
               n_layers: int = 6, train_mode: str = "physics",
               rhs_mode: str = "continuous") -> str:
    k_str = "krand" if isinstance(k, tuple) else f"k{k:.1f}".rstrip("0").rstrip(".")
    key = f"N{N}_e{epochs}_{k_str}"
    if base_ch != 32: key += f"_c{base_ch}"
    if n_layers != 6: key += f"_L{n_layers}"
    if train_mode != "physics": key += f"_{train_mode}"
    if rhs_mode != "continuous": key += f"_{rhs_mode}"
    return key

def resolve_paths(N: int, epochs: int, k: float, base_ch: int = 32,
                  n_layers: int = 6, train_mode: str = "physics",
                  rhs_mode: str = "continuous", model_type: str = "cnn") -> dict:
    ckey = config_key(N, epochs, k, base_ch, n_layers, train_mode, rhs_mode)
    ckpt = Path("checkpoints") / model_type / ckey
    res = Path("results") / model_type / ckey
    return {"ckpt_dir": ckpt, "best_path": ckpt / "best.pt", "last_path": ckpt / "last.pt",
            "history_path": ckpt / "history.csv", "convergence_path": ckpt / "convergence.png",
            "run_info_path": ckpt / "run_info.txt", "figure_dir": res / "figures",
            "results_dir": res, "config_name": f"{model_type}/{ckey}"}

def find_existing(ckpt_dir: Path, args: argparse.Namespace) -> Path | None:
    best = ckpt_dir / "best.pt"
    if not best.exists(): return None
    try:
        ckpt = torch.load(best, map_location="cpu", weights_only=False)
        a = ckpt.get("args", {}); a = vars(a) if isinstance(a, argparse.Namespace) else a
        keys = ["N","epochs","base_ch","n_layers","rhs_mode","train_mode",
                "pretrain_epochs","k_min","k_max","train_samples"]
        for k in keys:
            if a.get(k) != getattr(args, k, a.get(k)): return None
        return best
    except Exception: pass
    return None

def update_registry(model_type: str, ckey: str, info: dict) -> None:
    reg = Path("checkpoints") / "registry.json"; reg.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if reg.exists():
        try: data = json.loads(reg.read_text())
        except (json.JSONDecodeError, ValueError): pass
    data.setdefault(model_type, {})[ckey] = {"updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **info}
    reg.write_text(json.dumps(data, indent=2, ensure_ascii=False))

def migrate_legacy_checkpoint(ckpt_dir: Path) -> None:
    if (ckpt_dir / "best.pt").exists(): return
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for old, new in [("demo_best.pt","best.pt"),("demo.csv","history.csv"),("demo.png","convergence.png")]:
        src = Path("checkpoints") / old
        if src.exists(): src.rename(ckpt_dir / new)

# ==============================================================================
# 3. Grid, inputs, boundary
# ==============================================================================

def make_grid(N: int, device: torch.device = "cpu", dtype: torch.dtype = torch.float32):
    H = W = N + 2; h = 2.0 / (N + 1)
    xs = torch.linspace(-1., 1., W, device=device, dtype=dtype)
    ys = torch.linspace(-1., 1., H, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    x, y = xx[None, None, :, :], yy[None, None, :, :]
    boundary = torch.zeros(1, 1, H, W, device=device, dtype=dtype)
    boundary[:, :,  0, :] = boundary[:, :, -1, :] = 1.
    boundary[:, :,  :,  0] = boundary[:, :,  :, -1] = 1.
    return x, y, boundary, 1. - boundary, h

def build_input(f_norm, g, boundary, k_norm, x, y):
    B = f_norm.shape[0]
    if boundary.shape[0] == 1: boundary = boundary.expand(B, -1, -1, -1)
    if x.shape[0] == 1: x = x.expand(B, -1, -1, -1); y = y.expand(B, -1, -1, -1)
    k_ch = k_norm.view(B, 1, 1, 1).expand(B, 1, *f_norm.shape[-2:])
    return torch.cat([f_norm, g, boundary, k_ch, x, y], dim=1)

def enforce_dirichlet(u_raw, g, boundary, u_floor=1e-4):
    u_int = u_floor + F.softplus(u_raw)
    return boundary * g + (1. - boundary) * u_int

# ==============================================================================
# 4. Flux-form FD residual
# ==============================================================================

def _safe_pow(u, k, eps=1e-6):
    k = k.to(device=u.device, dtype=u.dtype)
    while k.ndim < 4: k = k.unsqueeze(-1)
    if torch.allclose(k, k.round(), atol=1e-6): return torch.pow(u, k), k * torch.pow(u, k - 1.)
    return torch.pow(u.clamp(min=eps), k), k * torch.pow(u.clamp(min=eps), k - 1.)

def power_coefficient(u, k, eps=1e-6):
    a, _ = _safe_pow(u, k, eps); return a

def flux_divergence(u, k, h, eps=1e-6):
    a = power_coefficient(u, k, eps)
    uc, ac = u[:,:,1:-1,1:-1], a[:,:,1:-1,1:-1]
    ae = .5 * (ac + a[:,:,1:-1,2:]);  aw = .5 * (ac + a[:,:,1:-1,:-2])
    an = .5 * (ac + a[:,:,2:,1:-1]);  as_ = .5 * (ac + a[:,:,:-2,1:-1])
    fe = ae * (u[:,:,1:-1,2:] - uc) / h;  fw = aw * (uc - u[:,:,1:-1,:-2]) / h
    fn = an * (u[:,:,2:,1:-1] - uc) / h;  fs = as_ * (uc - u[:,:,:-2,1:-1]) / h
    return -(fe - fw + fn - fs) / h

def pde_residual(u, f, k, h, eps=1e-6):
    return flux_divergence(u, k, h, eps) - f[:,:,1:-1,1:-1]

# ==============================================================================
# 5. Simple CNN
# ==============================================================================

class SimpleCNN(nn.Module):
    """Pure conv stack: n_layers × (Conv3×3 + GroupNorm + GELU), then 1×1 output."""
    def __init__(self, in_ch=6, out_ch=1, hidden_ch=32, n_layers=6):
        super().__init__()
        layers = []
        for i in range(n_layers):
            c_in = in_ch if i == 0 else hidden_ch
            g = min(8, hidden_ch)
            while hidden_ch % g: g -= 1
            layers += [nn.Conv2d(c_in, hidden_ch, 3, padding=1, padding_mode="replicate"),
                       nn.GroupNorm(g, hidden_ch), nn.GELU()]
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(hidden_ch, out_ch, 1)

    def forward(self, x): return self.head(self.body(x))

# ==============================================================================
# 6. Random PDE family dataset
# ==============================================================================

def _random_u_params(device, k_range=(0.5, 2.5)):
    """Return (params, k) for one random manufactured solution."""
    p = {"c0": torch.empty(1, device=device).uniform_(1.2, 1.8),
         "c1": torch.empty(1, device=device).uniform_(-.15, .15),
         "c2": torch.empty(1, device=device).uniform_(-.15, .15),
         "c3": torch.empty(1, device=device).uniform_(-.10, .10),
         "a1": torch.empty(1, device=device).uniform_(-.20, .20),
         "a2": torch.empty(1, device=device).uniform_(-.20, .20)}
    k = torch.empty(1, device=device).uniform_(k_range[0], k_range[1])
    return p, k

def _manufactured_u(x, y, p):
    """u = b + (1-x²)(1-y²)·s.  b = boundary poly, s = random Fourier modes."""
    phi = (1. - x**2) * (1. - y**2)
    b = p["c0"] + p["c1"]*x + p["c2"]*y + p["c3"]*x*y
    s = p["a1"] * torch.sin(math.pi*x) * torch.sin(math.pi*y) \
      + p["a2"] * torch.cos(2.*math.pi*x) * torch.cos(math.pi*y)
    return b + phi * s

def _continuous_rhs(u, x, y, k):
    """f = -div(u^k grad u) via autograd."""
    k4 = k.view(1,1,1,1) if k.ndim <= 1 else k
    while k4.ndim < 4: k4 = k4.unsqueeze(-1)
    a = torch.pow(u, k4); ones = torch.ones_like(u)
    ux = torch.autograd.grad(u, x, ones, create_graph=True, retain_graph=True)[0]
    uy = torch.autograd.grad(u, y, ones, create_graph=True, retain_graph=True)[0]
    qx = a * ux; qy = a * uy
    qxx = torch.autograd.grad(qx, x, ones, create_graph=False, retain_graph=True)[0]
    qyy = torch.autograd.grad(qy, y, ones, create_graph=False, retain_graph=False)[0]
    return -(qxx + qyy)

def _discrete_rhs(u, k, h):
    """f such that R_h(u) = 0 (for code verification)."""
    f = torch.zeros_like(u); f[:,:,1:-1,1:-1] = flux_divergence(u, k, h); return f

def make_one_problem(N, device, rhs_mode="continuous", k_range=(0.5, 2.5)):
    x0, y0, boundary, interior, h = make_grid(N, device)
    need_grad = rhs_mode == "continuous"
    x = x0.clone().detach().requires_grad_(need_grad)
    y = y0.clone().detach().requires_grad_(need_grad)
    p, k = _random_u_params(device, k_range)
    u = _manufactured_u(x, y, p)
    # ensure positivity
    margin = .2 - u.detach().min()
    if margin > 0: u = _manufactured_u(x, y, {**p, "c0": p["c0"] + margin})
    f = _continuous_rhs(u, x, y, k) if rhs_mode == "continuous" else _discrete_rhs(u.detach(), k, h)
    u, f = u.detach(), f.detach()
    return {"f": f, "g": boundary * u, "k": k.detach(), "u_exact": u, "h": h}

def generate_random_dataset(N, n_samples, device, rhs_mode="continuous",
                            k_range=None):
    kr = k_range if k_range is not None else (0.5, 2.5)
    probs = [make_one_problem(N, device, rhs_mode, kr) for _ in range(n_samples)]
    # shared geometry (same for all samples — store once)
    x, y, boundary, interior, _ = make_grid(N, device)
    return {"f": torch.cat([p["f"] for p in probs], dim=0),
            "g": torch.cat([p["g"] for p in probs], dim=0),
            "k": torch.cat([p["k"] for p in probs], dim=0),
            "u_exact": torch.cat([p["u_exact"] for p in probs], dim=0),
            "mask": boundary, "interior_mask": interior,
            "x": x, "y": y, "h": probs[0]["h"]}

@dataclass
class InputStats:
    f_mean: float; f_std: float; k_mean: float; k_std: float

def compute_stats(data):
    fi = data["f"][:,:,1:-1,1:-1]; k = data["k"]
    return InputStats(float(fi.mean()), float(fi.std().clamp_min(1e-8)),
                      float(k.mean()), float(k.std(unbiased=False).clamp_min(1e-8)))

# ==============================================================================
# 7. Training
# ==============================================================================

def _take_batch(data, idx):
    return {**{k: data[k][idx] for k in ("f","g","u_exact","k")},
            **{k: data[k][:1] for k in ("mask","interior_mask","x","y")}, "h": data["h"]}

def _forward(model, batch, stats, norm_k):
    fn = (batch["f"] - stats.f_mean) / stats.f_std
    ki = ((batch["k"] - stats.k_mean) / stats.k_std) if norm_k else batch["k"]
    inp = build_input(fn, batch["g"], batch["mask"], ki, batch["x"], batch["y"])
    return enforce_dirichlet(model(inp), batch["g"], batch["mask"])

def train(args, best_path, last_path, history_path, convergence_path, model=None):
    set_seed(args.seed); device = select_device(args.device); t0 = time.time()

    rhs_mode = getattr(args, "rhs_mode", "continuous")
    kr = (getattr(args, "k_min", 0.5), getattr(args, "k_max", 2.5))

    print(f"generating {args.train_samples} random PDEs (N={args.N}, k~[{kr[0]:.1f},{kr[1]:.1f}], rhs={rhs_mode})")
    train_data = generate_random_dataset(args.N, args.train_samples, device, rhs_mode, kr)
    print(f"generating {args.val_samples} random PDEs")
    val_data = generate_random_dataset(args.N, args.val_samples, device, rhs_mode, kr)
    stats = compute_stats(train_data)

    with torch.no_grad():
        oracle = float(pde_residual(val_data["u_exact"], val_data["f"], val_data["k"],
                                     val_data["h"]).pow(2).mean().sqrt())

    n_layers = getattr(args, "n_layers", 6)
    if model is None:
        model = SimpleCNN(hidden_ch=args.base_ch, n_layers=n_layers).to(device)
    else:
        model = model.to(device)

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        (no_decay if n.endswith(".bias") or p.ndim <= 1 else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                              {"params": no_decay, "weight_decay": 0.}], lr=args.lr)

    steps_per_ep = math.ceil(args.train_samples / args.batch_size)
    total_steps = steps_per_ep * args.epochs
    warmup = max(1, int(.05 * total_steps))
    def lr_fn(s):
        if s < warmup: return s / warmup
        return .5 * (1. + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

    best_res = math.inf; eval_bs = args.eval_batch_size or args.batch_size

    print(f"device={device}  N={args.N}  oracle={oracle:.4f}  rhs={rhs_mode}")
    print(f"training objective = normalized flux-form PDE residual only")
    print(f"u_exact is evaluation-only (no supervised MSE in training)")
    print(f"steps/epoch={steps_per_ep}  total={total_steps}")

    history, step = [], 0
    for epoch in range(args.epochs):
        model.train(); perm = torch.randperm(args.train_samples, device=device)
        for start in range(0, args.train_samples, args.batch_size):
            batch = _take_batch(train_data, perm[start:start + args.batch_size])
            u_pred = _forward(model, batch, stats, args.normalize_k)

            res = pde_residual(u_pred, batch["f"], batch["k"], batch["h"])
            denom = batch["f"][:,:,1:-1,1:-1].pow(2).mean(dim=(1,2,3)).sqrt().clamp_min(1e-3)
            loss = (res / denom.view(-1,1,1,1)).pow(2).mean()

            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step(); sched.step(); step += 1

        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            val_m = evaluate(model, val_data, stats, eval_bs, args.normalize_k)
            if val_m["resRMSE"] < best_res:
                best_res = val_m["resRMSE"]
                best_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model_state_dict": model.state_dict(), "input_stats": vars(stats),
                            "args": vars(args), "epoch": epoch, "best_val_resRMSE": best_res}, best_path)
            print(f"epoch={epoch:03d}  loss={loss.detach():.3e}  resRMSE={val_m['resRMSE']:.4f} | "
                  f"val relL2={val_m['relL2']:.4e}  linf={val_m['linf']:.4e}  "
                  f"gradRelL2={val_m['gradRelL2']:.4e}")
            history.append({"epoch": epoch, "loss": loss.detach().item(),
                           "val_relL2": val_m["relL2"], "val_resRMSE": val_m["resRMSE"],
                           "val_linf": val_m["linf"], "val_gradRelL2": val_m["gradRelL2"]})

    for p, data in [(last_path, {"model_state_dict": model.state_dict(),
                                  "input_stats": vars(stats), "args": vars(args)}),
                    (history_path, None), (convergence_path, None)]:
        p.parent.mkdir(parents=True, exist_ok=True)
        if data: torch.save(data, p)
    _write_csv(history, history_path)
    _save_convergence_plot(history, convergence_path, oracle)
    return model, oracle

# ==============================================================================
# 8. Evaluation
# ==============================================================================

def _error_tensors(u_pred, batch):
    ue = batch["u_exact"]; interior = batch["interior_mask"].expand_as(u_pred)
    bnd = batch["mask"]; bnd = bnd.expand_as(u_pred) if bnd.shape[0] == 1 else bnd
    err = (u_pred - ue) * interior
    rp = pde_residual(u_pred, batch["f"], batch["k"], batch["h"])
    re = pde_residual(ue, batch["f"], batch["k"], batch["h"])
    return err, rp, re, (u_pred - batch["g"]).abs() * bnd, interior, ue, float(batch["h"])

@torch.no_grad()
def evaluate(model, data, stats, batch_size, norm_k):
    model.eval(); n = data["f"].shape[0]; dev = data["f"].device
    s = {"rl2": 0., "li": 0., "rs": 0., "rli": 0., "bm": 0., "gs": 0., "c": 0}
    for start in range(0, n, batch_size):
        idx = torch.arange(start, min(start+batch_size, n), device=dev)
        b = _take_batch(data, idx); up = _forward(model, b, stats, norm_k)
        ue, interior, h = b["u_exact"], b["interior_mask"].expand_as(up), float(b["h"])
        err = (up - ue) * interior
        res = pde_residual(up, b["f"], b["k"], b["h"])
        m = up.shape[0]; s["c"] += m
        s["rl2"] += float((err.flatten(1).norm(dim=1) / ue.mul(interior).flatten(1).norm(dim=1).clamp_min(1e-12)).sum())
        s["li"] = max(s["li"], float(err.abs().max()))
        s["rs"] += float(res.pow(2).sum()); s["rli"] = max(s["rli"], float(res.abs().max()))
        s["bm"] = max(s["bm"], float(((up - b["g"]).abs() * b["mask"].expand_as(up)).max()))
        du = lambda v: (v[:,:,1:-1,2:] - v[:,:,1:-1,:-2]) / (2.*h)
        dv = lambda v: (v[:,:,2:,1:-1] - v[:,:,:-2,1:-1]) / (2.*h)
        pg = torch.cat([du(up), dv(up)], 1); tg = torch.cat([du(ue), dv(ue)], 1)
        s["gs"] += float(((pg-tg).flatten(1).norm(dim=1) / tg.flatten(1).norm(dim=1).clamp_min(1e-12)).sum())
    c = s["c"]; rn = c * (data["f"].shape[-2]-2) * (data["f"].shape[-1]-2)
    return {"relL2": s["rl2"]/c, "linf": s["li"], "resRMSE": math.sqrt(s["rs"]/rn),
            "resLinf": s["rli"], "bmax": s["bm"], "gradRelL2": s["gs"]/c}

def per_sample_metrics(u_pred, batch):
    ue = batch["u_exact"]; interior = batch["interior_mask"].expand_as(u_pred)
    err = (u_pred - ue) * interior
    res = pde_residual(u_pred, batch["f"], batch["k"], batch["h"])
    res_e = pde_residual(ue, batch["f"], batch["k"], batch["h"])
    ie = err[:,:,1:-1,1:-1].flatten(1); iu = ue[:,:,1:-1,1:-1].flatten(1)
    rf = res.flatten(1); ref = res_e.flatten(1)
    rl2 = ie.norm(dim=1) / iu.norm(dim=1).clamp_min(1e-12)
    return [{"sample_id": i, "k": float(batch["k"][i]), "relL2": float(rl2[i]),
             "RMSE": float(ie[i].pow(2).mean().sqrt()), "MAE": float(ie[i].abs().mean()),
             "Linf": float(ie[i].abs().max()),
             "pred_flux_resRMSE": float(rf[i].pow(2).mean().sqrt()),
             "exact_flux_resRMSE": float(ref[i].pow(2).mean().sqrt()),
             "f_RMSE": float(batch["f"].flatten(1)[i].pow(2).mean().sqrt())}
            for i in range(u_pred.shape[0])]

# ==============================================================================
# 9. Prediction & visualization
# ==============================================================================

def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    a = ckpt.get("args", {}); a = vars(a) if isinstance(a, argparse.Namespace) else a
    model = SimpleCNN(hidden_ch=a.get("base_ch",32), n_layers=a.get("n_layers",6)).to(device)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    s = ckpt["input_stats"]
    return model, InputStats(**s), argparse.Namespace(**a)

def _get_plt():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; return plt

def _save_panel(ue, up, err, fm, rp, re, path, title, fext, rext, dpi=200):
    plt = _get_plt()
    def sym(v): mx = max(float(v.abs().max().cpu()), 1e-12); return -mx, mx
    umin = min(float(ue.min()), float(up.min())); umax = max(float(ue.max()), float(up.max()))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    fig.suptitle(title)
    for r, c, v, l, cm, ex, v0, v1 in [
        (0,0,ue,"true u","viridis",fext,umin,umax), (0,1,up,"CNN prediction","viridis",fext,umin,umax),
        (0,2,err,"signed error","coolwarm",fext,*sym(err)),
        (1,0,fm,"forcing f","coolwarm",fext,*sym(fm)),
        (1,1,rp,"pred PDE residual","coolwarm",rext,*sym(rp)),
        (1,2,re,"exact PDE residual (floor)","coolwarm",rext,*sym(re))]:
        ax = axes[r][c]; im = ax.imshow(v.detach().cpu().numpy(), origin="lower", extent=ex, cmap=cm, vmin=v0, vmax=v1, aspect="equal")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title(l); fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(str(path), dpi=dpi, bbox_inches="tight"); plt.close(fig)

def _save_convergence_plot(history, path, oracle=0.):
    if len(history) < 2: return
    plt = _get_plt(); eps = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.suptitle("Training convergence")
    ax1b = ax1.twinx(); l1, = ax1.plot(eps, [h["loss"] for h in history], "b-", label="loss")
    l2, = ax1b.plot(eps, [h.get("loss_pde",0) for h in history], "r--", label="loss_pde")
    ax1.set_xlabel("epoch"); ax1.legend(handles=[l1,l2], loc="upper right"); ax1.grid(True, alpha=.3)
    ax1.set_title("Training loss")
    ax2b = ax2.twinx(); l3, = ax2.plot(eps, [h["val_relL2"] for h in history], "g-", label="val relL2")
    l4, = ax2b.plot(eps, [h["val_resRMSE"] for h in history], "m--", label="val resRMSE")
    if oracle > 0: ax2b.axhline(oracle, color="gray", ls=":", label=f"oracle ({oracle:.4f})")
    ax2.set_xlabel("epoch"); ax2.legend(handles=[l3,l4], loc="upper right"); ax2.grid(True, alpha=.3)
    ax2.set_title("Validation metrics")
    path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(str(path), dpi=150, bbox_inches="tight"); plt.close(fig)

def _write_csv(rows, path):
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def archive_existing_results(out_dir):
    existing = list(out_dir.glob("*.png")) + list(out_dir.glob("*.csv"))
    if not existing: return None
    ts = run_id(); ad = out_dir / "history" / ts; ad.mkdir(parents=True, exist_ok=True)
    for f in existing: shutil.move(str(f), str(ad / f.name))
    print(f"Archived previous results -> {ad}"); return ad

def run_prediction(args):
    if not args.load_path: raise ValueError("--load-path required")
    set_seed(args.seed); device = select_device(args.device)
    model, stats, ckpt_args = load_checkpoint(args.load_path, device)
    N = args.predict_N or getattr(ckpt_args, "N", 32)
    norm_k = getattr(ckpt_args, "normalize_k", True)

    print(f"prediction: N={N}  samples={args.predict_samples}")
    set_seed(args.test_seed)
    batch = generate_random_dataset(N, args.predict_samples, device, rhs_mode="continuous")
    model.eval()
    with torch.no_grad(): u_pred = _forward(model, batch, stats, norm_k)

    vm = evaluate(model, batch, stats, args.predict_samples, norm_k)
    print(f"relL2={vm['relL2']:.4e}  linf={vm['linf']:.4e}  resRMSE={vm['resRMSE']:.4f}  "
          f"gradRelL2={vm['gradRelL2']:.4e}")
    rows = per_sample_metrics(u_pred, batch)

    if args.predict_save:
        p = Path(args.predict_save); p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"f": batch["f"].cpu(), "g": batch["g"].cpu(), "k": batch["k"].cpu(),
                     "u_exact": batch["u_exact"].cpu(), "u_pred": u_pred.cpu(),
                     "h": float(batch["h"]), "metrics": vm, "config": {"N": N}}, p)

    if args.plot:
        out = Path(args.plot_dir); archive_existing_results(out)
        h = float(batch["h"]); fe = [-1., 1., -1., 1.]; re = [-1.+h, 1.-h, -1.+h, 1.-h]
        for i, row in enumerate(rows):
            ui = batch["u_exact"][i,0]; up = u_pred[i,0]
            rp = pde_residual(u_pred[i:i+1], batch["f"][i:i+1], batch["k"][i:i+1], h)[0,0]
            re = pde_residual(batch["u_exact"][i:i+1], batch["f"][i:i+1], batch["k"][i:i+1], h)[0,0]
            _save_panel(ui, up, up-ui, batch["f"][i,0], rp, re,
                        out / f"sample_{row['sample_id']:03d}_panel.png",
                        f"sample {row['sample_id']} | k={row['k']:.4g}  "
                        f"relL2={row['relL2']:.3e}  pred_res={row['pred_flux_resRMSE']:.3e}",
                        fe, re)
        _write_csv(rows, out / "metrics.csv"); print(f"saved panels + metrics: {out}")

# ==============================================================================
# 10. CLI & main
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=32)
    p.add_argument("--train-samples", type=int, default=2048)
    p.add_argument("--val-samples", type=int, default=128)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--base-ch", type=int, default=32, help="hidden channels")
    p.add_argument("--n-layers", type=int, default=6, help="conv layers")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.)
    p.add_argument("--normalize-k", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--k-min", type=float, default=0.5)
    p.add_argument("--k-max", type=float, default=2.5)
    p.add_argument("--rhs-mode", default="continuous", choices=["continuous","discrete"])
    p.add_argument("--device", default="auto"); p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--retrain", action="store_true"); p.add_argument("--load-path", default="")
    p.add_argument("--predict", action="store_true"); p.add_argument("--predict-samples", type=int, default=8)
    p.add_argument("--predict-save", default=""); p.add_argument("--predict-N", type=int, default=None)
    p.add_argument("--test-seed", type=int, default=1234)
    p.add_argument("--plot", action="store_true"); p.add_argument("--plot-dir", default="results/figures")
    return p.parse_args()

def main():
    args = parse_args()
    if args.predict or args.load_path: return run_prediction(args)

    if len(sys.argv) == 1:
        args.N = 32; args.epochs = 40; args.train_samples = 2048; args.val_samples = 128
        args.batch_size = 16; args.eval_batch_size = 0; args.eval_every = 1

    mt = "cnn"
    paths = resolve_paths(args.N, args.epochs, K_RANGE, args.base_ch,
                          getattr(args,"n_layers",6), "physics",
                          getattr(args,"rhs_mode","continuous"), mt)
    ckey = config_key(args.N, args.epochs, K_RANGE, args.base_ch,
                      getattr(args,"n_layers",6), "physics",
                      getattr(args,"rhs_mode","continuous"))

    migrate_legacy_checkpoint(paths["ckpt_dir"])
    existing = find_existing(paths["ckpt_dir"], args)

    if existing and not args.retrain:
        print("=" * 60); print(f"  Config:  {paths['config_name']}")
        print(f"  Found trained model: {existing}"); print(f"  Use --retrain to retrain."); print("=" * 60)
        update_registry(mt, ckey, {"N": args.N, "epochs": args.epochs, "k": K_RANGE})
    else:
        if args.retrain and existing: print(f"Retraining {paths['config_name']}")
        print("=" * 60); print(f"  Config:  {paths['config_name']}"); print("=" * 60)
        ta = argparse.Namespace(**vars(args))
        if not hasattr(ta, "rhs_mode"): ta.rhs_mode = "continuous"
        model, oracle = train(ta, paths["best_path"], paths["last_path"], paths["history_path"], paths["convergence_path"])
        existing = paths["best_path"]
        update_registry(mt, ckey, {"N": args.N, "epochs": args.epochs, "k": K_RANGE, "oracle_floor": oracle})

    if getattr(args, "demo", False) or args.plot:
        pred_ckpt = existing or paths["best_path"]
        pa = argparse.Namespace(**vars(args)); pa.load_path = str(pred_ckpt)
        pa.predict_samples = 1 if getattr(args, "demo", False) else args.predict_samples
        pa.predict_save = ""; pa.plot = True; pa.plot_dir = str(paths["figure_dir"])
        pa.predict_N = None; pa.test_seed = args.seed; run_prediction(pa)

    _write_run_info(paths["run_info_path"], {"run_id": run_id(), "model": "SimpleCNN",
                   "N": args.N, "epochs": args.epochs, "train_samples": args.train_samples, "k": K_RANGE,
                   "training": "normalized flux-form PDE residual only, u_exact evaluation-only"})
    _update_latest_symlink(Path("checkpoints")/mt, ckey); _update_latest_symlink(Path("results")/mt, ckey)
    print(f"\nDone. Config: {paths['config_name']}")

if __name__ == "__main__": main()
