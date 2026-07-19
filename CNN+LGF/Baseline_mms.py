"""
CNN solver for -div(a(x,y) grad u) = f on [-1,1]^2, Dirichlet BC u=g.
a(x,y) = exp(sigma*psi) is a given smooth positive coefficient field
(sigma controls contrast).  MMS data: u_ref -> f = -div_h(a grad_h u_ref).
u may take either sign. Supports --loss {supervised, physics}.

Shared grid/operator/model/metric components live in Baseline_core.py; this
script owns the MMS data generation, the analytic test suite, and the
cnn_mms checkpoint/output layout.
"""

import argparse, hashlib, json, math
from dataclasses import astuple
from pathlib import Path

import numpy as np
import torch

# Full shared surface is imported explicitly (even names this script does not
# call directly) so pre-refactor usage like `import Baseline_mms;
# Baseline_mms.SimpleCNN(...)` keeps working.
from Baseline_core import (
    LGF_BANK_DIR,
    LGF_LOG_A_RANGE,
    LGFNet,
    LGFPhi,
    METRIC_KEYS,
    MultiKappaLGFBlock,
    NormalizationStats,
    SimpleCNN,
    _aggregate_metrics,
    _apply_kernel_bank,
    _assemble_linear_system,
    _expand_to_batch,
    _forward,
    _interp_kernel_response,
    _lgf_const_coeff_sanity_check,
    _linear_baseline_rl2,
    _make_groupnorm,
    _print_res_trace,
    _save_centerline,
    _save_convergence_plot,
    _save_panel,
    _save_sample_figures,
    _select_device,
    _select_kernel_weights,
    _set_seed,
    _setup_plot,
    _sync,
    _take_batch,
    _time_forward,
    _uref_range_and_bnd_err,
    _write_csv,
    build_input,
    build_lgf_kernel_bank,
    compute_linear_baselines,
    compute_loss,
    compute_stats,
    create_model,
    create_optimizer,
    enforce_dirichlet,
    ensure_linear_baselines,
    evaluate,
    flux_divergence,
    make_grid,
    pde_residual,
    per_sample_metrics,
    random_fourier_field,
    random_positive_field,
    ref_residual_rmse64,
    solve_linear_fd,
    summarize_rows,
    train_one_epoch,
    trivial_linear_baseline,
)

# -- manufactured solution --------------------------------------------------

def manufactured_u(x, y, M, rng=np.random):
    """u = b(x,y) + (1-x^2)(1-y^2)*q(x,y) (signed).

    q(x,y) is a normalized 2D Fourier series (interior perturbation).
    x, y are torch tensors; returns a CPU float64 tensor."""
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()
    N = x_np.shape[-1] - 2

    phi = (1. - x_np ** 2) * (1. - y_np ** 2)

    # Boundary polynomial: b(x,y) = b0 + b1*x + b2*y + b3*x*y
    b0 = rng.uniform(-0.25, 0.25)
    b1, b2 = rng.uniform(-0.15, 0.15, 2)
    b3 = rng.uniform(-0.1, 0.1)

    boundary_part = b0 + b1 * x_np + b2 * y_np + b3 * x_np * y_np

    fourier_part = random_fourier_field(N, M, rng, normalize=True)

    u_signed = boundary_part + 0.36 * phi * fourier_part
    return torch.from_numpy(u_signed)

# -- MMS dataset ------------------------------------------------------------

def make_mms_dataset(N, n_samples, device, args, seed):
    """Build MMS dataset: u_ref, a(sigma) -> f = -div_h(a grad_h u_ref) in
    float64, store float32."""
    rng = np.random.RandomState(seed)
    x_t, y_t, bnd, interior, h = make_grid(N, device)
    x64, y64, *_ = make_grid(N, "cpu", torch.float64)
    f_list, g_list, a_list, sigma_list, u_list = [], [], [], [], []
    res64_sq = 0.

    M = getattr(args, "fourier_m", 6)
    for _ in range(n_samples):
        sigma_i = (rng.uniform(args.sigma_min, args.sigma_max)
                   if getattr(args, "random_sigma", False) else args.sigma)

        a64 = torch.from_numpy(random_positive_field(N, sigma_i, M, rng))[None, None]
        u64 = manufactured_u(x64[0, 0], y64[0, 0], M, rng)
        u64 = u64.unsqueeze(0).unsqueeze(0)          # CPU float64
        f64 = torch.zeros_like(u64)
        f64[:, :, 1:-1, 1:-1] = flux_divergence(u64, a64, h)
        g64 = u64.clone()
        g64[:, :, 1:-1, 1:-1] = 0.
        res64_sq += float(pde_residual(
            u64, f64, a64, h).pow(2).sum())

        # accumulate on CPU; one bulk .to(device) after the cat below, instead
        # of one small host->device transfer per sample
        f_list.append(f64.float())
        g_list.append(g64.float())
        a_list.append(a64.float())
        sigma_list.append(torch.tensor([sigma_i], dtype=torch.float32))
        u_list.append(u64.float())

    result = {
        "f": torch.cat(f_list).to(device), "g": torch.cat(g_list).to(device),
        "a": torch.cat(a_list).to(device), "sigma": torch.cat(sigma_list).to(device),
        "u_ref": torch.cat(u_list).to(device),
        "mask": bnd, "interior_mask": interior,
        "x": x_t, "y": y_t, "h": h,
        "ref_res64": math.sqrt(res64_sq / (n_samples * N * N)),
    }
    return result

# -- training ---------------------------------------------------------------

def validate_and_checkpoint(model, val_data, stats, args, epoch, epoch_loss,
                            best_metric, best_path, history):
    eval_bs = args.eval_batch_size or args.batch_size
    val_m = evaluate(model, val_data, stats, eval_bs)
    val_key = "relL2" if args.loss == "supervised" else "resRMSE"
    if val_m[val_key] < best_metric:
        best_metric = val_m[val_key]
        best_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_type": "cnn_mms",
            "model_state_dict": model.state_dict(),
            "stats": astuple(stats),   # plain tuple: robust to future
            "args": vars(args),        # NormalizationStats renames/moves
            "epoch": epoch,
            f"best_val_{val_key}": best_metric,
        }, best_path)

    print(f"epoch {epoch:03d}  loss={epoch_loss:.3e}  "
          f"resRMSE={val_m['resRMSE']:.4f}  "
          f"relL2={val_m['relL2']:.4e}  maxErr={val_m['maxError']:.4e}  "
          f"gradRelL2={val_m['gradRelL2']:.4e}  "
          f"linBL={val_m['linearBaselineRelL2']:.4e}")

    history.append({"epoch": epoch, "loss": epoch_loss,
                    **{f"val_{key}": val_m[key] for key in (
                        "resRMSE", "relL2", "maxError", "gradRelL2",
                        "meanMaxError", "meanBoundaryError",
                        "maxBoundaryError", "meanGradLinf",
                        "maxGradLinf", "refResRMSE",
                        "linearBaselineRelL2")}})
    return best_metric

def train(args, best_path, last_path, history_path, convergence_path, model=None):
    _set_seed(args.seed)
    device = _select_device(args.device)

    print(f"Equation: -div(a(x,y) grad u) = f  |  sigma = {args.sigma}  |  data=mms")
    if args.random_sigma:
        print(f"  random sigma in [{args.sigma_min}, {args.sigma_max}]")

    print(f"generating {args.train_samples} MMS samples (N={args.N})")
    train_data = make_mms_dataset(args.N, args.train_samples, device, args,
                                  seed=args.seed)
    val_data = make_mms_dataset(args.N, args.val_samples, device, args,
                                seed=args.seed + 1)
    stats = compute_stats(train_data)
    print(f"f stats: mean={stats.f_mean:.4e}  std={stats.f_std:.4e}  "
          f"log(a) stats: mean={stats.loga_mean:.4e}  std={stats.loga_std:.4e}")

    # Reference residual, reported in float64 (float32 rounding floor
    # ~h^-2*eps32 must not be mistaken for solver accuracy)
    ref_res = val_data["ref_res64"]
    u_min, u_max, bnd_err = _uref_range_and_bnd_err(val_data)
    print(f"MMS residual RMSE (float64) = {ref_res:.4e}  (should be near 0)")
    print(f"u_ref range: [{u_min:.4f}, {u_max:.4f}]  "
          f"boundary error = {bnd_err:.2e}")

    triv_rl2 = trivial_linear_baseline(val_data)
    print(f"trivial baseline (a≡1 linear Poisson) relL2 = {triv_rl2:.4e}  "
          f"(CNN must beat this to have learned a(x,y))")

    model = create_model(args, device) if model is None else model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model={args.model}  parameters={n_params:,}")

    opt = create_optimizer(model, args)

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
    print(f"device={device}  ref_res={ref_res:.4e}  "
          f"steps/epoch={steps_per_ep}  total={total_steps}  "
          f"n_train={n_train}  n_val={n_val}")

    best_metric = math.inf
    history = []

    for epoch in range(args.epochs):
        epoch_loss = train_one_epoch(model, train_data, stats, opt, sched, args, device)

        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            best_metric = validate_and_checkpoint(
                model, val_data, stats, args, epoch, epoch_loss,
                best_metric, best_path, history)

    if isinstance(model, LGFNet):
        assert torch.equal(model.kernel_bank.cpu(), model._bank_snapshot.cpu()), (
            "LGF kernel bank changed during training (should be frozen)")

    last_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_type": "cnn_mms",
                 "model_state_dict": model.state_dict(),
                 "stats": astuple(stats), "args": vars(args)}, last_path)
    _write_csv(history, history_path)
    _save_convergence_plot(history, convergence_path, args.loss, dataset_label="MMS")
    return model, best_metric

# -- analytic test suite ----------------------------------------------------

# Analytic formulas (signed) -- used directly, no shift/rescale.
ANALYTIC_CASES = {
    "u1": lambda x, y: np.sin(np.pi * x) * np.cos(np.pi * y),
    "u2": lambda x, y: (1. - x ** 2) * (1. - y ** 2),
    "u3": lambda x, y: np.cos(np.pi * x) * np.cos(2. * np.pi * y),
    "u4": lambda x, y: x ** 2 * (1. - y) + np.sin(np.pi * x) * y,
}

def make_analytic_dataset(N, case, sigma, device, test_rng, M=6):
    """Build a test dataset from an analytic u(x,y) (signed) against a
    freshly-drawn test coefficient field a(x,y) of contrast sigma."""
    x_t, y_t, bnd, interior, h = make_grid(N, device)
    x_np = x_t[0, 0].cpu().numpy()
    y_np = y_t[0, 0].cpu().numpy()
    u_func = ANALYTIC_CASES[case]

    u_np = u_func(x_np, y_np).astype(np.float64)
    a_np = random_positive_field(N, sigma, M, test_rng)
    u64 = torch.from_numpy(u_np)[None, None]          # CPU float64
    a64 = torch.from_numpy(a_np)[None, None]
    f64 = torch.zeros_like(u64)
    f64[:, :, 1:-1, 1:-1] = flux_divergence(u64, a64, h)
    g_np = u_np.copy()
    g_np[1:-1, 1:-1] = 0.

    res64 = float(pde_residual(u64, f64, a64, h).pow(2).mean().sqrt())

    result = {
        "f": f64.float().to(device), "g": torch.from_numpy(g_np).float()[None, None].to(device),
        "a": a64.float().to(device),
        "sigma": torch.tensor([sigma], dtype=torch.float32, device=device),
        "u_ref": u64.float().to(device),
        "mask": bnd, "interior_mask": interior,
        "x": x_t, "y": y_t, "h": h,
        "ref_res64": res64,   # float64 residual of (u_ref, f): machine precision by construction
    }
    return result

def run_analytic_suite(model, stats, device, args, ckpt_args, out_dir):
    """Run all analytic test cases, save metrics + figures."""
    N = ckpt_args.N
    all_rows = []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sigma_test = float(args.predict_sigma)
    M = getattr(ckpt_args, "fourier_m", 6)
    test_rng = np.random.RandomState(args.test_seed)
    cnn_ms_all = []
    for case_name in ANALYTIC_CASES:
        print(f"\n  --- analytic case: {case_name}  (sigma={sigma_test:g}) ---")
        batch = make_analytic_dataset(N, case_name, sigma_test, device, test_rng, M)
        ensure_linear_baselines(batch)
        u_pred, _, cnn_ms_sample = _time_forward(model, batch, stats)
        cnn_ms_all.append(cnn_ms_sample)
        rows = per_sample_metrics(u_pred, batch)
        for r in rows:
            r["case"] = case_name
            r["cnn_ms_per_sample"] = cnn_ms_sample
            r["fd_solver_ms_per_sample"] = float("nan")   # no FD solve in MMS pipeline
        all_rows.extend(rows)

        print(f"    relL2={rows[0]['relL2']:.4e}  maxErr={rows[0]['maxError']:.4e}  "
              f"resRMSE={rows[0]['predResRMSE']:.4e}  "
              f"linBL={rows[0]['linearBaselineRelL2']:.4e}")
        print(f"    ref residual (float64) = {batch['ref_res64']:.4e}")

        # analytic suite always saves figures (only reached with predict_case=analytic)
        _save_sample_figures(batch, u_pred, 0, out_dir, f"analytic_{case_name}",
                             f"analytic {case_name}  sigma={rows[0]['sigma']:.2f}  "
                             f"relL2={rows[0]['relL2']:.3e}")

    print(f"\n  timing: cnn_ms_per_sample={np.mean(cnn_ms_all):.3f}  "
          f"fd_solver_ms_per_sample=nan (no FD solve in MMS pipeline)")
    _write_csv(all_rows, out_dir / "analytic_metrics.csv")
    metric_keys = ["relL2", "maxError", "predResRMSE", "refResRMSE",
                   "bndErr", "gradRelL2", "constBaselineRelL2", "linearBaselineRelL2"]
    summary_all = summarize_rows(all_rows, metric_keys)
    with open(out_dir / "analytic_summary.json", "w") as f:
        json.dump(summary_all, f, indent=2, ensure_ascii=False)
    print(f"  analytic suite saved → {out_dir}")
    return all_rows

# -- prediction -------------------------------------------------------------

def run_prediction(args):
    if not args.predict_from:
        raise ValueError("--predict-from required")

    _set_seed(args.seed)
    device = _select_device(args.device)

    ckpt = torch.load(args.predict_from, map_location=device, weights_only=False)
    ckpt_type = ckpt.get("model_type")
    if ckpt_type and ckpt_type != "cnn_mms":
        raise ValueError(
            f"checkpoint type mismatch: expected 'cnn_mms', got '{ckpt_type}'. "
            f"Use the correct prediction script (Baseline_mms.py for "
            f"'cnn_mms', Baseline_solver.py for 'cnn').")
    if "sigma" not in ckpt["args"]:
        raise ValueError(
            "checkpoint predates the a(x,y)/sigma refactor (has 'k', not "
            "'sigma') — retrain with the current script.")
    ckpt_args = argparse.Namespace(**ckpt["args"])
    model = create_model(ckpt_args, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    stats = ckpt["stats"]
    if not isinstance(stats, NormalizationStats):
        stats = NormalizationStats(*stats)   # pre-refactor checkpoint: plain tuple

    N = ckpt_args.N
    data_args = argparse.Namespace(**vars(ckpt_args))
    data_args.device = args.device
    data_args.seed = args.test_seed
    data_args.random_sigma = False
    data_args.sigma = args.predict_sigma

    if args.predict_case == "analytic":
        out = (Path(args.plot_dir) if args.plot_dir
               else Path("results") / "predict_mms" / "analytic")
        run_analytic_suite(model, stats, device, args, ckpt_args, out)
        return

    assert args.test_seed not in (ckpt_args.seed, ckpt_args.seed + 1), (
        "test seed collides with a train/val seed! prediction data would "
        "overlap the training run (evaluation pollution)."
    )

    print(f"prediction (MMS): N={N}  samples={args.predict_samples}  "
          f"sigma={args.predict_sigma:g} (fixed)")

    batch = make_mms_dataset(N, args.predict_samples, device, data_args,
                             seed=args.test_seed)
    ensure_linear_baselines(batch)

    want_trace = isinstance(model, LGFNet)
    timing = _time_forward(model, batch, stats, return_diagnostics=want_trace)
    if want_trace:
        u_pred, cnn_ms_batch, cnn_ms_sample, res_trace = timing
        _print_res_trace(res_trace)
    else:
        u_pred, cnn_ms_batch, cnn_ms_sample = timing

    rows = per_sample_metrics(u_pred, batch)
    vm = _aggregate_metrics(rows)
    print(f"relL2={vm['relL2']:.4e}  maxErr={vm['maxError']:.4e}  "
          f"resRMSE={vm['resRMSE']:.4f}  refRMSE={vm['refResRMSE']:.4e}  "
          f"gradRelL2={vm['gradRelL2']:.4e}")
    print(f"meanMaxError={vm['meanMaxError']:.4e}")
    print(f"meanBoundaryError={vm['meanBoundaryError']:.2e}  maxBoundaryError={vm['maxBoundaryError']:.2e}")
    print(f"meanGradLinf={vm['meanGradLinf']:.4e}  maxGradLinf={vm['maxGradLinf']:.4e}")
    print(f"reference residual RMSE (float64) = {batch['ref_res64']:.4e}")
    print(f"linear baseline (a≡1) relL2 = {vm['linearBaselineRelL2']:.4e}  "
          f"(CNN must beat this to have learned a(x,y))  "
          f"const baseline relL2 = {vm['constBaselineRelL2']:.4e}")
    diag_note = " (incl. per-step residual diagnostics)" if want_trace else ""
    print(f"timing: cnn={cnn_ms_batch:.2f} ms/batch  "
          f"cnn_ms_per_sample={cnn_ms_sample:.3f}{diag_note}  "
          f"fd_solver_ms_per_sample=nan (no FD solve in MMS pipeline)")

    if args.predict_save:
        p = Path(args.predict_save)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "f": batch["f"].cpu(), "g": batch["g"].cpu(),
            "a": batch["a"].cpu(), "u_ref": batch["u_ref"].cpu(),
            "u_pred": u_pred.cpu(), "h": float(batch["h"]), "metrics": vm,
        }, p)
        print(f"saved predictions → {p}")

    for r in rows:
        r["cnn_ms_per_sample"] = cnn_ms_sample
        r["fd_solver_ms_per_sample"] = float("nan")   # no FD solve in MMS pipeline
    summary = summarize_rows(rows, METRIC_KEYS)

    # Always save metrics — use save dir parent, plot-dir, or default
    if args.predict_save:
        out = Path(args.predict_save).parent
    elif args.plot_dir:
        out = Path(args.plot_dir)
    else:
        out = Path("results") / "predict_mms"
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, out / "metrics.csv")
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if args.plot:
        for i, row in enumerate(rows):
            _save_sample_figures(batch, u_pred, i, out, f"sample_{i:03d}",
                                 f"sample {i}  sigma={row['sigma']:.2f}  "
                                 f"relL2={row['relL2']:.3e}")
        print(f"saved → {out}")

# -- CLI --------------------------------------------------------------------

def ckpt_key(args):
    """Readable experiment key + short hash of the remaining result-affecting
    training args (seed, lr, weight decay, batch size, val samples, eval
    cadence, grad clip), so changing any of them can't silently reuse another
    run's checkpoint directory."""
    if getattr(args, "random_sigma", False):
        sig_str = f"sig{args.sigma_min:g}-{args.sigma_max:g}"
    else:
        sig_str = f"sig{args.sigma:g}"
    n_layers = getattr(args, "n_layers", 6)
    loss_suffix = "_sup" if getattr(args, "loss", "physics") == "supervised" else ""
    model_name = getattr(args, "model", "cnn")
    model_suffix = ""
    if model_name == "lgf":
        model_suffix = f"_lgf_T{args.lgf_T}_M{args.lgf_M}_L{args.lgf_L}"
    extra = (args.seed, args.lr, args.weight_decay, args.batch_size,
             args.val_samples, args.eval_every, args.grad_clip)
    cfg = hashlib.md5(repr(extra).encode()).hexdigest()[:8]
    return (f"N{args.N}_e{args.epochs}_tr{args.train_samples}"
            f"_{sig_str}_M{getattr(args, 'fourier_m', 6)}"
            f"_ch{args.base_ch}_L{n_layers}{loss_suffix}{model_suffix}_{cfg}_mms")

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

    # Model selection
    p.add_argument("--model", default="cnn", choices=["cnn", "lgf"],
                   help="cnn: SimpleCNN one-shot map | lgf: LGFNet multi-kappa "
                        "LGF unrolled iteration")
    p.add_argument("--lgf-T", type=int, default=12,
                   help="LGFNet: number of unrolled iteration steps")
    p.add_argument("--lgf-M", type=int, default=8,
                   help="LGFNet: number of kernel-bank nodes")
    p.add_argument("--lgf-L", type=int, default=4,
                   help="LGFNet: number of multi-kappa LGF blocks in Phi_theta")

    # Linear variable coefficient a(x,y) = exp(sigma * psi)
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--random-sigma", action=argparse.BooleanOptionalAction, default=False,
                   help="sample sigma uniformly from [sigma_min, sigma_max] "
                        "(default: False; use --random-sigma to enable)")
    p.add_argument("--sigma-min", type=float, default=0.5)
    p.add_argument("--sigma-max", type=float, default=1.5)
    p.add_argument("--fourier-m", type=int, default=6,
                   help="number of Fourier modes per dimension for random fields")

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
    p.add_argument("--test-seed", type=int, default=9999)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--predict-case", default="data",
                   choices=["data", "analytic"],
                   help="'data': same MMS family | 'analytic': 4 signed analytic test cases")
    p.add_argument("--predict-sigma", type=float, default=0.5,
                   help="fixed sigma (test coefficient-field contrast) for prediction")
    p.add_argument("--plot-dir", default="",
                   help="directory for output figures and metrics CSVs")
    return p.parse_args()

# -- main -------------------------------------------------------------------

def main():
    args = parse_args()

    if args.mms_sanity:
        print("=" * 60)
        print("MMS SANITY CHECK")
        print("=" * 60)
        print(f"Equation: -div(a(x,y) grad u) = f  |  sigma = {args.sigma}")
        print(f"Grid: N={args.N}  |  samples: {args.train_samples}")

        _set_seed(args.seed)
        device = _select_device(args.device)
        data = make_mms_dataset(args.N, args.train_samples, device, args,
                                seed=args.seed)

        exact_rmse = data["ref_res64"]
        ref32in64 = ref_residual_rmse64(data)
        u_min, u_max, bnd_err = _uref_range_and_bnd_err(data)
        fi = data["f"][:, :, 1:-1, 1:-1]

        print(f"  exact_resRMSE (float64) = {exact_rmse:.4e}  (should be near 0)")
        print(f"  float32 data residual   = {ref32in64:.4e}  "
              f"(~h^-2·eps32 rounding floor)")
        print(f"  boundary error  = {bnd_err:.2e}  (should be 0)")
        print(f"  u_ref range     = [{u_min:.4f}, {u_max:.4f}]")
        print(f"  f range         = [{float(fi.min()):.4e}, {float(fi.max()):.4e}]")
        print(f"  trivial baseline (a≡1) relL2 = {trivial_linear_baseline(data):.4e}")

        if exact_rmse > 1e-8:
            print(f"  WARNING: residual = {exact_rmse:.4e} > 1e-8, check FD!")
        else:
            print("  OK — MMS residual is near machine precision.")

        if args.model == "lgf":
            print("-" * 60)
            print(f"LGF kernel bank (M={args.lgf_M}) sanity checks:")
            bank, _ = build_lgf_kernel_bank(args.N, data["h"], args.lgf_M)
            assert torch.isfinite(bank).all()
            print(f"  bank finite: OK  (shape={tuple(bank.shape)})")
            _lgf_const_coeff_sanity_check(args.N, args.lgf_M)

        print("=" * 60)
        return

    if args.predict:
        return run_prediction(args)

    key = ckpt_key(args)
    ckpt_dir = Path("checkpoints") / "cnn_mms" / key
    best_path = ckpt_dir / "best.pt"

    if best_path.exists() and not args.retrain:
        print(f"found {best_path}, use --retrain to retrain")
    else:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"training {key}")
        _, best_metric = train(args, best_path,
                               ckpt_dir / "last.pt",
                               ckpt_dir / "history.csv",
                               ckpt_dir / "convergence.png")
        print(f"saved → {ckpt_dir}")

    if args.plot:
        figures_dir = Path("results") / "cnn_mms" / key / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        pa = argparse.Namespace(**vars(args))
        pa.predict_from = str(best_path)
        pa.predict_save = ""
        pa.plot = True
        pa.plot_dir = str(figures_dir)
        pa.test_seed = args.seed + 100000   # disjoint from train RNG stream
        run_prediction(pa)

    print(f"done: checkpoints/cnn_mms/{key}")

if __name__ == "__main__":
    main()
