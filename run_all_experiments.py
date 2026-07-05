"""
Run all 4 CNN experiments + optional PINN baseline.

Smoke mode (--smoke): small grid, few epochs — verify pipeline quickly.
Full mode (--full): production-scale grid and epochs.

Examples:
  python run_all_experiments.py --smoke
  python run_all_experiments.py --full
  python run_all_experiments.py --smoke --run-pinn --pinn-steps 500
"""

import argparse, os, subprocess, sys, time
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parent


def _common_args(smoke):
    if smoke:
        return ["--N", "16", "--epochs", "2", "--train-samples", "8",
                "--val-samples", "4", "--batch-size", "4", "--fourier-m", "4"]
    return ["--N", "32", "--epochs", "100", "--train-samples", "128",
            "--val-samples", "32", "--batch-size", "8", "--fourier-m", "6"]


def _mms_args(smoke, loss):
    return ["--loss", loss, "--random-k", "--k-min", "0", "--k-max", "5",
            "--retrain", "--eval-every", "10"]


def _solver_args(smoke, loss):
    return ["--loss", loss, "--random-k", "--k-min", "0", "--k-max", "5",
            "--retrain", "--eval-every", "10"]


def _find_best_ckpt(base_dir):
    """Find the most recently modified best.pt under base_dir."""
    best = sorted(Path(base_dir).rglob("best.pt"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    return str(best[0]) if best else None


def run_cmd(cmd, desc):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT))
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  FAILED after {elapsed:.0f}s (exit {result.returncode})")
        return False, elapsed
    print(f"  OK ({elapsed:.0f}s)")
    return True, elapsed


def train_predict_mms(smoke, loss, predict_k, test_seed, ckpt_dir):
    common = _common_args(smoke) + _mms_args(smoke, loss)
    train_cmd = [sys.executable, "Baseline_mms.py"] + common
    if not run_cmd(train_cmd, f"MMS train ({loss})")[0]:
        return {}

    ckpt = _find_best_ckpt(ckpt_dir)
    if not ckpt:
        print("  WARNING: no checkpoint found")
        return {}

    pred_common = ["--predict", "--predict-from", ckpt,
                   "--predict-k", str(predict_k),
                   "--test-seed", str(test_seed),
                   "--predict-samples", "8", "--plot"]
    results = {}
    for case, case_label in [("data", "data"), ("analytic", "analytic")]:
        ok, t = run_cmd(
            [sys.executable, "Baseline_mms.py"] + pred_common + ["--predict-case", case],
            f"MMS predict ({loss}, {case_label})")
        if ok:
            results[f"mms_{loss}_{case}"] = {"ckpt": ckpt, "time": t}

    return results


def train_predict_solver(smoke, loss, predict_k, test_seed, ckpt_dir):
    common = _common_args(smoke) + _solver_args(smoke, loss)
    train_cmd = [sys.executable, "Baseline_solver.py"] + common
    if not run_cmd(train_cmd, f"Solver train ({loss})")[0]:
        return {}

    ckpt = _find_best_ckpt(ckpt_dir)
    if not ckpt:
        print("  WARNING: no checkpoint found")
        return {}

    pred_common = ["--predict", "--predict-from", ckpt,
                   "--predict-k", str(predict_k),
                   "--test-seed", str(test_seed),
                   "--predict-samples", "8", "--plot"]
    results = {}
    for case, case_label in [("data", "data"), ("fixed-solver", "fixed-solver")]:
        ok, t = run_cmd(
            [sys.executable, "Baseline_solver.py"] + pred_common + ["--predict-case", case],
            f"Solver predict ({loss}, {case_label})")
        if ok:
            results[f"solver_{loss}_{case}"] = {"ckpt": ckpt, "time": t}

    return results


def run_pinn(smoke, pinn_steps):
    args = ["--N", "16"] if smoke else ["--N", "32"]
    args += ["--epochs", str(pinn_steps),
             "--eval-every", str(max(1, pinn_steps // 10)),
             "--lbfgs-epochs", "0",
             "--hidden-dim", "32" if smoke else "64",
             "--n-layers", "3" if smoke else "4",
             "--bc-weight", "10" if smoke else "50",
             "--retrain"]
    return run_cmd([sys.executable, "PINN.py"] + args, "PINN baseline")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true",
                   help="Small-scale smoke test (N=16, epochs=2)")
    p.add_argument("--full", action="store_true",
                   help="Full-scale experiment (N=32, epochs=100)")
    p.add_argument("--run-pinn", action="store_true",
                   help="Also run PINN baseline")
    p.add_argument("--skip-train", action="store_true",
                   help="Skip training, only run predictions")
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int, default=0,
                   help="Override epochs (overrides --smoke/--full)")
    p.add_argument("--train-samples", type=int, default=0)
    p.add_argument("--val-samples", type=int, default=0)
    p.add_argument("--predict-samples", type=int, default=0)
    p.add_argument("--pinn-steps", type=int, default=500,
                   help="PINN training steps (default 500)")
    p.add_argument("--predict-k", type=float, default=2.0)
    p.add_argument("--test-seed", type=int, default=101234)
    args = p.parse_args()

    if not args.smoke and not args.full and args.epochs == 0:
        p.error("specify --smoke, --full, or --epochs")

    smoke = args.smoke or (not args.full and args.epochs == 0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.chdir(str(PROJECT))

    print(f"timestamp: {timestamp}")
    print(f"smoke={smoke}  pinn={args.run_pinn}  skip_train={args.skip_train}")

    all_results = {}

    # CNN experiments (A-D)
    if not args.skip_train:
        all_results.update(
            train_predict_mms(smoke, "supervised", args.predict_k, args.test_seed,
                              "checkpoints/cnn_mms"))
        all_results.update(
            train_predict_mms(smoke, "physics", args.predict_k, args.test_seed,
                              "checkpoints/cnn_mms"))
        all_results.update(
            train_predict_solver(smoke, "supervised", args.predict_k, args.test_seed,
                                 "checkpoints/cnn"))
        all_results.update(
            train_predict_solver(smoke, "physics", args.predict_k, args.test_seed,
                                 "checkpoints/cnn"))

    # PINN baseline
    if args.run_pinn:
        ok, t = run_pinn(smoke, args.pinn_steps)
        if ok:
            all_results["pinn"] = {"time": t,
                                    "ckpt": str(PROJECT / "checkpoints/pinn")}

    print(f"\n{'='*60}")
    print(f"  Done.  {len(all_results)} experiment(s) completed.")
    print(f"  Results: run  aggregate_results.py  to generate comparison tables.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
