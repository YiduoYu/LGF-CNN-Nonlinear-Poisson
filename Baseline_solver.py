"""
CNN solver for -div(a(x,y) grad u) = f on [-1,1]^2, Dirichlet BC u=g.
a(x,y) = exp(sigma*psi) is a given smooth positive coefficient field
(sigma controls contrast). Linear elliptic problem: forward solves reduce
to a single sparse linear solve; u may take either sign.

Data generation: random f,g,a(sigma) -> direct sparse linear solve -> u_ref.
Supports --loss {supervised, physics} and --model {cnn, lgf}.

Shared grid/operator/model/metric components live in Baseline_core.py; this
script owns the random-f,g solver data generation, the fixed-solver test
suite, and the cnn checkpoint/output layout.
"""

import argparse, hashlib, json, math, time
from dataclasses import astuple
from pathlib import Path

import numpy as np
import torch

# Full shared surface is imported explicitly (even names this script does not
# call directly) so pre-refactor usage like `import Baseline_solver;
# Baseline_solver.SimpleCNN(...)` keeps working.
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

# -- random f and boundary generators ----------------------------------------

def _random_f(N, f_scale, M, rng=np.random):
    """Random forcing f via full 2D Fourier series, boundary = 0."""
    f = random_fourier_field(N, M, rng) * f_scale
    f[0, :] = f[-1, :] = f[:, 0] = f[:, -1] = 0.
    return f

def _random_g(N, g_base=0.0, g_scale=0.6, M=6, rng=np.random):
    """Random Dirichlet boundary via Fourier series (g_base shifts it).
    Interior = 0."""
    raw = random_fourier_field(N, M, rng)
    g = np.zeros((N + 2, N + 2), dtype=np.float64)
    g[0, :] = g_base + g_scale * raw[0, :]
    g[-1, :] = g_base + g_scale * raw[-1, :]
    g[:, 0] = g_base + g_scale * raw[:, 0]
    g[:, -1] = g_base + g_scale * raw[:, -1]
    g[1:-1, 1:-1] = 0.
    return g

# -- forward solver dataset --------------------------------------------------

def make_solver_dataset(N, n_samples, device, args, seed):
    """Random f,g,a(sigma) -> single sparse linear FD solve -> u_ref."""
    rng = np.random.RandomState(seed)
    x_t, y_t, bnd, interior, h_grid = make_grid(N, device)
    M = getattr(args, "fourier_m", 6)

    f_list, g_list, a_list, sigma_list, u_list, info_list = [], [], [], [], [], []
    res64_sq = 0.

    print(f"generating {n_samples} solver samples (N={N})  "
          f"f_scale={args.f_scale}  g_base={args.g_base}  g_scale={args.g_scale}")

    for _ in range(n_samples):
        sigma_i = (rng.uniform(args.sigma_min, args.sigma_max)
                   if getattr(args, "random_sigma", False) else args.sigma)
        a_np = random_positive_field(N, sigma_i, M, rng)
        f_np = _random_f(N, args.f_scale, M, rng)
        g_np = _random_g(N, args.g_base, args.g_scale, M, rng)

        t0 = time.perf_counter()
        u_np = solve_linear_fd(f_np, g_np, a_np, h_grid)
        solve_ms = (time.perf_counter() - t0) * 1e3

        u64 = torch.from_numpy(u_np).double()[None, None]
        f64 = torch.from_numpy(f_np).double()[None, None]
        a64 = torch.from_numpy(a_np).double()[None, None]
        res64_sq += float(pde_residual(u64, f64, a64, h_grid).pow(2).sum())

        # accumulate on CPU; one bulk .to(device) after the cat below, instead
        # of one small host->device transfer per sample
        f_list.append(f64.float())
        g_list.append(torch.from_numpy(g_np).float().unsqueeze(0).unsqueeze(0))
        a_list.append(a64.float())
        sigma_list.append(torch.tensor([sigma_i], dtype=torch.float32))
        u_list.append(u64.float())
        info_list.append({"solve_time_ms": solve_ms})

    a_cat = torch.cat(a_list)
    sigma_cat = torch.cat(sigma_list)
    ref_res64 = math.sqrt(res64_sq / (n_samples * N * N))
    a_min, a_max = float(a_cat.min()), float(a_cat.max())
    print(f"  sigma range: [{float(sigma_cat.min()):.4f}, {float(sigma_cat.max()):.4f}]")
    print(f"  a range: [{a_min:.4f}, {a_max:.4f}]  "
          f"contrast a_max/a_min={a_max/a_min:.4f}")
    print(f"  ref residual (float64) = {ref_res64:.2e}")

    return {
        "f": torch.cat(f_list).to(device), "g": torch.cat(g_list).to(device),
        "a": a_cat.to(device), "sigma": sigma_cat.to(device),
        "u_ref": torch.cat(u_list).to(device),
        "mask": bnd, "interior_mask": interior,
        "x": x_t, "y": y_t, "h": h_grid,
        "ref_res64": ref_res64, "solver_info": info_list,
    }

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
            "model_type": "cnn",
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

    print(f"Equation: -div(a(x,y) grad u) = f  |  sigma = {args.sigma}  |  data=solver")
    if args.random_sigma:
        print(f"  random sigma in [{args.sigma_min}, {args.sigma_max}]")

    train_data = make_solver_dataset(args.N, args.train_samples, device,
                                      args, seed=args.seed)
    val_data = make_solver_dataset(args.N, args.val_samples, device,
                                    args, seed=args.seed + 1)
    stats = compute_stats(train_data)
    print(f"f stats: mean={stats.f_mean:.4e}  std={stats.f_std:.4e}  "
          f"log(a) stats: mean={stats.loga_mean:.4e}  std={stats.loga_std:.4e}")

    # Reference residual, reported in float64 (float32 rounding floor
    # ~h^-2*eps32 must not be mistaken for solver accuracy)
    ref_res = val_data["ref_res64"]
    u_min, u_max, bnd_err = _uref_range_and_bnd_err(val_data)
    print(f"ref residual RMSE (float64) = {ref_res:.4e}  (linear FD solver residual)")
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
    torch.save({"model_type": "cnn",
                 "model_state_dict": model.state_dict(),
                 "stats": astuple(stats), "args": vars(args)}, last_path)
    _write_csv(history, history_path)
    _save_convergence_plot(history, convergence_path, args.loss, dataset_label="solver")
    return model, best_metric

# -- fixed-solver test suite --------------------------------------------------
#
# predict-case semantics:
#   data:         random f,g,sigma -> linear FD solve -> u_ref (training family)
#   fixed-solver: manufactured DISCRETE test — u_target builds f and g through
#                 the discrete flux operator, then u_ref is recomputed by the
#                 sparse FD solver of that same discrete system. It checks that
#                 the torch flux operator and the scipy matrix assembly agree,
#                 that the sparse-solve residual is small, and how the CNN
#                 generalizes to manufactured analytic inputs. It does NOT
#                 measure the continuous-PDE discretization error (the forcing
#                 itself comes from the discrete operator): report it as a
#                 discrete consistency / manufactured discrete test, not as an
#                 independent continuous-accuracy validation. The analytic MMS
#                 cases live in Baseline_mms.py.

# Same raw formulas as the MMS ANALYTIC_CASES u1-u4 (signed, no shift).
FIXED_SOLVER_CASES = {
    "u1_sincos": lambda x, y: np.sin(np.pi * x) * np.cos(np.pi * y),
    "u2_poly":   lambda x, y: (1. - x ** 2) * (1. - y ** 2),
    "u3_cos":    lambda x, y: np.cos(np.pi * x) * np.cos(2. * np.pi * y),
    "u4_mixed":  lambda x, y: x ** 2 * (1. - y) + np.sin(np.pi * x) * y,
}

def make_fixed_solver_dataset(N, device, args, sigma_val, test_rng):
    """Deterministic solver test: u_target (signed) -> f = -div_h(a grad u),
    then u_ref = solve_linear_fd(f, g, a, h) (single linear solve) against
    a fresh test coefficient field per case."""
    x_t, y_t, bnd, interior, h = make_grid(N, device)
    x64, y64, *_ = make_grid(N, "cpu", torch.float64)
    x_np = x64[0, 0].numpy()
    y_np = y64[0, 0].numpy()
    M = getattr(args, "fourier_m", 6)

    f_list, g_list, a_list, sigma_list, u_list, t_list = [], [], [], [], [], []
    info_list, names = [], []
    res64_sq = 0.

    for case_name, u_func in FIXED_SOLVER_CASES.items():
        u_target = u_func(x_np, y_np).astype(np.float64)
        a_np = random_positive_field(N, sigma_val, M, test_rng)

        g_np = u_target.copy()
        g_np[1:-1, 1:-1] = 0.

        t64 = torch.from_numpy(u_target)[None, None]      # CPU float64
        a64 = torch.from_numpy(a_np)[None, None]
        f64 = torch.zeros_like(t64)
        f64[:, :, 1:-1, 1:-1] = flux_divergence(t64, a64, h)
        f_np = f64[0, 0].numpy()

        t0 = time.perf_counter()
        u_ref = solve_linear_fd(f_np, g_np, a_np, h)
        solve_ms = (time.perf_counter() - t0) * 1e3

        u64 = torch.from_numpy(u_ref).double()[None, None]
        res64 = pde_residual(u64, f64, a64, h).pow(2)
        case_res64 = float(res64.mean().sqrt())
        res64_sq += float(res64.sum())

        f_list.append(f64.float())
        g_list.append(torch.from_numpy(g_np).float()[None, None])
        a_list.append(a64.float())
        sigma_list.append(torch.tensor([sigma_val], dtype=torch.float32))
        u_list.append(u64.float())
        t_list.append(t64.float())
        info_list.append({"residual_rmse": case_res64, "solve_time_ms": solve_ms})
        names.append(case_name)

    return {
        "f": torch.cat(f_list).to(device), "g": torch.cat(g_list).to(device),
        "a": torch.cat(a_list).to(device), "sigma": torch.cat(sigma_list).to(device),
        "u_ref": torch.cat(u_list).to(device), "u_target": torch.cat(t_list).to(device),
        "case_names": names, "solver_info": info_list,
        "mask": bnd, "interior_mask": interior,
        "x": x_t, "y": y_t, "h": h,
        "ref_res64": math.sqrt(res64_sq / (len(names) * N * N)),
    }

def _save_target_comparison(u_target, u_solver, path, title, dpi=150):
    """1x3 panel: MMS u_target | solver u_ref | (solver - target)."""
    plt = _setup_plot()
    ut = u_target.detach().cpu().numpy()
    us = u_solver.detach().cpu().numpy()
    diff = us - ut
    vmin = min(ut.min(), us.min())
    vmax = max(ut.max(), us.max())
    dmax = max(abs(diff).max(), 1e-12)
    ext = [-1., 1., -1., 1.]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    fig.suptitle(title)
    panels = [(ut,   "u_target (manufactured)", "viridis",  vmin,  vmax),
              (us,   "u_solver_ref",     "viridis",  vmin,  vmax),
              (diff, "solver - target",  "coolwarm", -dmax, dmax)]
    for ax, (v, lbl, cmap, v0, v1) in zip(axes, panels):
        im = ax.imshow(v, origin="lower", extent=ext, cmap=cmap,
                       vmin=v0, vmax=v1, aspect="equal")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title(lbl)
        fig.colorbar(im, ax=ax, fraction=.046, pad=.04)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def run_fixed_solver_suite(model, stats, device, args, data_args, out_dir):
    """Run the fixed-solver (manufactured discrete) cases, save metrics
    CSV/JSON + figures.

    CNN metrics (via per_sample_metrics) compare u_pred against the solver
    reference; target_vs_solver_ref_* checks that the sparse solve reproduces
    the manufactured discrete target, i.e. that the torch flux operator and
    the scipy assembly are consistent (discrete consistency — see the section
    comment above; this is not a continuous-accuracy test)."""
    N = data_args.N
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"prediction (fixed-solver): N={N}  sigma={args.predict_sigma:g}  "
          f"{len(FIXED_SOLVER_CASES)} cases, u_ref from single linear FD solve")
    test_rng = np.random.RandomState(args.test_seed)
    batch = make_fixed_solver_dataset(N, device, data_args, args.predict_sigma, test_rng)
    ensure_linear_baselines(batch)
    print(f"  ref residual (float64) = {batch['ref_res64']:.4e}")

    u_pred, _, cnn_ms_sample = _time_forward(model, batch, stats)
    rows = per_sample_metrics(u_pred, batch)

    interior = batch["interior_mask"][0, 0].detach().cpu().numpy()
    ur_np = batch["u_ref"].detach().cpu().numpy()
    ut_np = batch["u_target"].detach().cpu().numpy()

    out_rows = []
    for i, (case, info) in enumerate(zip(batch["case_names"],
                                         batch["solver_info"])):
        diff = (ur_np[i, 0] - ut_np[i, 0]) * interior
        ref = ur_np[i, 0] * interior
        t_rl2 = float(np.linalg.norm(diff) / max(np.linalg.norm(ref), 1e-12))
        t_max = float(np.abs(diff).max())

        r = rows[i]
        out_rows.append({
            "case": case,
            "sigma": r["sigma"],
            "cnn_vs_solver_ref_relL2": r["relL2"],
            "cnn_vs_solver_ref_maxError": r["maxError"],
            "predResRMSE": r["predResRMSE"],
            "refResRMSE": r["refResRMSE"],
            "boundaryError": r["bndErr"],
            "gradRelL2": r["gradRelL2"],
            "constBaselineRelL2": r["constBaselineRelL2"],
            "linearBaselineRelL2": r["linearBaselineRelL2"],
            "target_vs_solver_ref_relL2": t_rl2,
            "target_vs_solver_ref_maxError": t_max,
            "solver_residual_rmse": info["residual_rmse"],
            "cnn_ms_per_sample": cnn_ms_sample,
            "fd_solver_ms_per_sample": info["solve_time_ms"],
        })

        print(f"\n  --- fixed-solver case: {case}  (sigma={r['sigma']:g}) ---")
        print(f"    relL2={r['relL2']:.4e}  maxErr={r['maxError']:.4e}  "
              f"resRMSE={r['predResRMSE']:.4e}  "
              f"linBL={r['linearBaselineRelL2']:.4e}")
        print(f"    target_vs_solver relL2={t_rl2:.2e}  maxErr={t_max:.2e}  "
              f"solver: res={info['residual_rmse']:.2e}")
        if t_rl2 > 1e-8:
            print(f"    WARNING: target_vs_solver_ref_relL2={t_rl2:.2e} — "
                  f"solver reference deviates from the manufactured discrete "
                  f"target; torch flux operator and scipy assembly disagree "
                  f"(discrete consistency broken)")

        if args.plot:
            _save_sample_figures(batch, u_pred, i, out_dir, f"fixed_solver_{case}",
                                 f"fixed-solver {case}  sigma={r['sigma']:.2f}  "
                                 f"relL2={r['relL2']:.3e}")
            _save_target_comparison(batch["u_target"][i, 0], batch["u_ref"][i, 0],
                                    out_dir / f"fixed_solver_{case}_target_vs_solver.png",
                                    f"{case}  sigma={r['sigma']:.2f}  "
                                    f"target_vs_solver relL2={t_rl2:.2e}")

    fd_ms_mean = float(np.mean([x["solve_time_ms"] for x in batch["solver_info"]]))
    print(f"\n  timing: cnn_ms_per_sample={cnn_ms_sample:.3f}  "
          f"fd_solver_ms_per_sample={fd_ms_mean:.3f}")
    _write_csv(out_rows, out_dir / "fixed_solver_metrics.csv")
    metric_keys = [key for key in out_rows[0] if key != "case"]
    summary = summarize_rows(out_rows, metric_keys)
    with open(out_dir / "fixed_solver_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  fixed-solver suite saved → {out_dir}")
    return out_rows

# -- prediction -------------------------------------------------------------

def run_prediction(args):
    if not args.predict_from:
        raise ValueError("--predict-from required")

    _set_seed(args.seed)
    device = _select_device(args.device)

    ckpt = torch.load(args.predict_from, map_location=device, weights_only=False)
    ckpt_type = ckpt.get("model_type")
    if ckpt_type and ckpt_type != "cnn":
        raise ValueError(
            f"checkpoint type mismatch: expected 'cnn', got '{ckpt_type}'. "
            f"Use the correct prediction script (Baseline_solver.py for "
            f"'cnn', Baseline_mms.py for 'cnn_mms').")
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

    if args.predict_case == "fixed-solver":
        out = (Path(args.plot_dir) if args.plot_dir
               else Path("results") / "predict" / "fixed_solver")
        run_fixed_solver_suite(model, stats, device, args, data_args, out)
        return

    assert args.test_seed not in (ckpt_args.seed, ckpt_args.seed + 1), (
        "test seed collides with a train/val seed! prediction data would "
        "overlap the training run (evaluation pollution)."
    )

    print(f"prediction (solver): N={N}  samples={args.predict_samples}  "
          f"sigma={args.predict_sigma:g} (fixed)")
    batch = make_solver_dataset(N, args.predict_samples, device,
                                data_args, seed=args.test_seed)
    ensure_linear_baselines(batch)

    want_trace = isinstance(model, LGFNet)
    timing = _time_forward(model, batch, stats, return_diagnostics=want_trace)
    if want_trace:
        u_pred, cnn_ms_batch, cnn_ms_sample, res_trace = timing
        _print_res_trace(res_trace)
    else:
        u_pred, cnn_ms_batch, cnn_ms_sample = timing
    fd_ms_sample = float(np.mean(
        [x["solve_time_ms"] for x in batch["solver_info"]]))

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
          f"fd_solver_ms_per_sample={fd_ms_sample:.3f}")

    if args.predict_save:
        p = Path(args.predict_save)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "f": batch["f"].cpu(), "g": batch["g"].cpu(),
            "a": batch["a"].cpu(), "u_ref": batch["u_ref"].cpu(),
            "u_pred": u_pred.cpu(), "h": float(batch["h"]), "metrics": vm,
        }, p)
        print(f"saved predictions → {p}")

    for r, info in zip(rows, batch["solver_info"]):
        r["cnn_ms_per_sample"] = cnn_ms_sample
        r["fd_solver_ms_per_sample"] = info["solve_time_ms"]
    summary = summarize_rows(rows, METRIC_KEYS)

    # Always save metrics — use --predict-save, --plot-dir, or default dir
    if args.predict_save:
        out = Path(args.predict_save).parent
    elif args.plot_dir:
        out = Path(args.plot_dir)
    else:
        out = Path("results") / "predict"
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
            f"_fs{args.f_scale:g}_gb{args.g_base:g}_gs{args.g_scale:g}"
            f"_ch{args.base_ch}_L{n_layers}{loss_suffix}{model_suffix}_{cfg}")

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=32)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--train-samples", type=int, default=128)
    p.add_argument("--val-samples", type=int, default=32)
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

    # Data generation
    p.add_argument("--sanity", action="store_true",
                   help="generate solver dataset, print residual/data stats, "
                        "then exit")

    # Linear variable coefficient a(x,y) = exp(sigma * psi)
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--random-sigma", action=argparse.BooleanOptionalAction, default=False,
                   help="sample sigma uniformly from [sigma_min, sigma_max] "
                        "(default: False; use --random-sigma to enable)")
    p.add_argument("--sigma-min", type=float, default=0.5)
    p.add_argument("--sigma-max", type=float, default=1.5)

    # Fourier series
    p.add_argument("--fourier-m", type=int, default=6,
                   help="number of Fourier modes per dimension for random fields")

    # Random field scales
    p.add_argument("--f-scale", type=float, default=6.0,
                   help="scale of random forcing")
    p.add_argument("--g-base", type=float, default=0.0)
    p.add_argument("--g-scale", type=float, default=0.6)

    # Loss
    p.add_argument("--loss", default="physics",
                   choices=["physics", "supervised"],
                   help="physics: PDE residual only | supervised: |u_pred - u_ref|^2")

    # Training control
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--predict-from", default="",
                   help="load checkpoint for prediction (not for resume training)")
    p.add_argument("--predict", action="store_true")
    p.add_argument("--predict-samples", type=int, default=8)
    p.add_argument("--predict-save", default="")
    p.add_argument("--test-seed", type=int, default=9999)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--predict-case", default="data",
                   choices=["data", "fixed-solver"],
                   help="'data': random f,g,sigma -> solver -> u_ref | "
                        "'fixed-solver': manufactured u_target builds f,g via "
                        "the discrete flux operator; u_ref recomputed by the "
                        "sparse FD solver (discrete consistency test, not a "
                        "continuous-accuracy test)")
    p.add_argument("--plot-dir", default="",
                   help="output dir for prediction figures "
                        "(default: results/predict/<case>)")
    p.add_argument("--predict-sigma", type=float, default=0.5,
                   help="fixed sigma (test coefficient-field contrast) for prediction")
    return p.parse_args()

# -- main -------------------------------------------------------------------

def main():
    args = parse_args()

    if args.sanity:
        print("=" * 60)
        print("SANITY CHECK (data=solver, single linear FD solve)")
        print("=" * 60)
        print(f"Equation: -div(a(x,y) grad u) = f  |  sigma = {args.sigma}")
        print(f"Grid: N={args.N}  |  samples: {args.train_samples}")

        _set_seed(args.seed)
        device = _select_device(args.device)
        data = make_solver_dataset(args.N, args.train_samples, device,
                                    args, seed=args.seed)

        ref32in64 = ref_residual_rmse64(data)
        u_min, u_max, bnd_err = _uref_range_and_bnd_err(data)
        fi = data["f"][:, :, 1:-1, 1:-1]

        print(f"  ref residual (float64 pipeline) = {data['ref_res64']:.4e}")
        print(f"  stored float32 data residual    = {ref32in64:.4e}  "
              f"(~h^-2·eps32 rounding floor)")
        print(f"  boundary error = {bnd_err:.2e}  (should be 0)")
        print(f"  u_ref range    = [{u_min:.4f}, {u_max:.4f}]")
        print(f"  f range        = [{float(fi.min()):.4e}, {float(fi.max()):.4e}]")
        print(f"  trivial baseline (a≡1) relL2    = "
              f"{trivial_linear_baseline(data):.4e}")

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
    ckpt_dir = Path("checkpoints") / "cnn" / key
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
        figures_dir = Path("results") / "cnn" / key / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        pa = argparse.Namespace(**vars(args))
        pa.predict_from = str(best_path)
        pa.predict_save = ""
        pa.plot = True
        pa.plot_dir = str(figures_dir)
        pa.test_seed = args.seed + 100000   # disjoint from train RNG stream
        run_prediction(pa)

    print(f"done: checkpoints/cnn/{key}")

if __name__ == "__main__":
    main()
