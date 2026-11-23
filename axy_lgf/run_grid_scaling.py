"""Grid scaling of the existing normal-contrast reaction-diffusion solvers.

From the package parent: python -m axy_lgf.run_grid_scaling
The original 46 tests run before any benchmark. Existing outputs are never
overwritten; use --output-dir with a fresh path for a subsequent experiment.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
from statistics import median
import subprocess
import sys
from time import perf_counter
import xml.etree.ElementTree as ET

import numpy as np

from .multigrid import solve_multigrid_pcg
from .operators import apply_reaction_diffusion, assemble_reaction_diffusion, l2_norm
from .reaction_manufactured import make_reaction_problem, validate_exponential_parameters
from .solvers import solve_axy_pcg

SIZES = (31, 63, 127, 255)
METHODS = ("lgf", "gmg", "amg")
REPETITIONS = 5
TOL = 1e-10
MAXITER = 500
PROJECT = Path(__file__).resolve().parent


def run_preflight(output: Path, *, test_files=None, prefix="preflight_tests", expected=46) -> dict:
    """Require all original tests, including optional Torch/AMG checks."""
    if test_files is None:
        test_files = ("tests/test_reaction_diffusion.py", "tests/test_torch_compatibility.py")
    command = [sys.executable, "-m", "pytest", "-q",
               *[str(PROJECT / name) for name in test_files],
               f"--junitxml={output / (prefix + '.xml')}"]
    result = subprocess.run(command, cwd=PROJECT, capture_output=True, text=True)
    (output / (prefix + ".log")).write_text(result.stdout + result.stderr)
    xml_path = output / (prefix + ".xml")
    cases = list(ET.parse(xml_path).iter("testcase")) if xml_path.exists() else []
    failures = sum(case.find("failure") is not None for case in cases)
    errors = sum(case.find("error") is not None for case in cases)
    skipped = sum(case.find("skipped") is not None for case in cases)
    passed = len(cases) - failures - errors - skipped
    return dict(command=command, returncode=result.returncode, passed=passed,
                failures=failures, errors=errors, skipped=skipped,
                successful=result.returncode == 0 and passed == expected and not skipped)


def solver_name(method: str) -> str:
    return "BoomerAMG-PCG" if method == "boomer" else f"{method.upper()}-PCG"


def input_hashes(problem) -> dict:
    return {name: hashlib.sha256(getattr(problem, name).tobytes()).hexdigest()
            for name in ("a", "q", "a_nodes", "f")}


def validate_grid(problem) -> dict:
    """Check the selected coefficient and sparse/matrix-free agreement."""
    for name in ("a", "q", "a_nodes", "f", "u_exact"):
        if getattr(problem, name).dtype != np.float64:
            raise ValueError(f"{name} must be float64")
    contrast = float(problem.a.max() / problem.a.min())
    if not (problem.q.min() >= 0.8 and problem.q.max() <= 1.2):
        raise ValueError("reaction coefficient is outside its specified range")
    if problem.coefficient == "normal":
        if not (problem.a.min() >= 0.8 and problem.a.max() <= 1.2 and contrast <= 1.5 + 1e-14):
            raise ValueError("coefficients are outside the specified normal-contrast range")
    elif not np.all(np.isfinite(problem.a_nodes)) or np.any(problem.a_nodes <= 0):
        raise ValueError("exponential diffusion coefficient must be finite and positive")
    matrix = assemble_reaction_diffusion(problem.a, problem.grid, problem.q,
                                          a_nodes=problem.a_nodes)
    probe = np.random.default_rng(problem.grid.n).standard_normal(problem.a.shape)
    action = apply_reaction_diffusion(probe, problem.a, problem.grid, problem.q,
                                      a_nodes=problem.a_nodes)
    difference = (matrix @ probe.ravel()).reshape(probe.shape) - action
    action_error = float(np.linalg.norm(difference) / np.linalg.norm(action))
    if action_error > 1e-12:
        raise ValueError("sparse and matrix-free actions disagree")
    metadata = dict(n=problem.grid.n, dof=problem.grid.n**2, h=problem.grid.h,
                    a_min=float(problem.a.min()), a_max=float(problem.a.max()),
                    a_contrast=contrast, q_min=float(problem.q.min()),
                    q_max=float(problem.q.max()), input_sha256=input_hashes(problem),
                    sparse_action_relative_difference=action_error)
    if problem.coefficient == "exponential":
        metadata.update(coefficient=problem.coefficient, a_scale=problem.a_scale, chi=problem.chi,
                        a_nodes_min=float(problem.a_nodes.min()),
                        a_nodes_max=float(problem.a_nodes.max()),
                        a_nodes_contrast=float(problem.a_nodes.max() / problem.a_nodes.min()))
    return metadata


def run_once(problem, method: str, phase: str, repetition: int) -> dict:
    """Rebuild the preconditioner and independently check the final residual."""
    record = dict(n=problem.grid.n, dof=problem.grid.n**2,
                  solver=solver_name(method), phase=phase, repetition=repetition,
                  iterations=None, relative_residual=None, setup_time_ms=None,
                  solve_time_ms=None, total_time_ms=None, call_wall_time_ms=None,
                  converged=False, status="error", message="")
    if problem.coefficient == "exponential":
        record.update(coefficient=problem.coefficient, a_scale=problem.a_scale, chi=problem.chi,
                      a_contrast=float(problem.a.max() / problem.a.min()),
                      a_nodes_contrast=float(problem.a_nodes.max() / problem.a_nodes.min()))
    kwargs = dict(q=problem.q, a_nodes=problem.a_nodes, u0=problem.grid.zeros(),
                  tol=TOL, maxiter=MAXITER)
    started = perf_counter()
    try:
        if method == "lgf":
            result = solve_axy_pcg(problem.a, problem.f, problem.grid, **kwargs)
        elif method == "boomer":
            from .boomer_amg import solve_boomer_pcg
            result = solve_boomer_pcg(problem.a, problem.f, problem.grid, **kwargs)
        else:
            result = solve_multigrid_pcg(problem.a, problem.f, problem.grid,
                                         method=method, **kwargs)
        record["call_wall_time_ms"] = 1000 * (perf_counter() - started)
        residual = problem.f - apply_reaction_diffusion(
            result.u, problem.a, problem.grid, problem.q, a_nodes=problem.a_nodes)
        relative_residual = l2_norm(residual, problem.grid) / l2_norm(problem.f, problem.grid)
        converged = result.converged and np.isfinite(relative_residual) and relative_residual <= TOL
        record.update(iterations=result.iterations,
                      relative_residual=relative_residual if np.isfinite(relative_residual) else None,
                      setup_time_ms=1000 * result.setup_time,
                      solve_time_ms=1000 * result.solve_time,
                      total_time_ms=1000 * (result.setup_time + result.solve_time),
                      converged=bool(converged),
                      status="converged" if converged else "not_converged")
        if not converged:
            record["message"] = "Solver status or recomputed residual failed the requested tolerance"
    except Exception as exc:
        record.update(call_wall_time_ms=1000 * (perf_counter() - started),
                      message=f"{type(exc).__name__}: {exc}")
    return record


def summarize_case(n: int, method: str, records: list[dict]) -> dict:
    """Use every timed repetition; never summarize a successful subset."""
    timed = [r for r in records if r["phase"] == "timed"]
    warmups = [r for r in records if r["phase"] == "warmup"]
    def median_value(key):
        values = [r[key] for r in timed]
        return median(values) if values and all(v is not None for v in values) else None

    residuals = [r["relative_residual"] for r in timed]
    complete = len(warmups) == 1 and len(timed) == REPETITIONS and all(r["converged"] for r in records)
    summary = dict(n=n, dof=n**2, solver=solver_name(method),
                iterations=median_value("iterations"),
                relative_residual=max(residuals) if residuals and all(r is not None for r in residuals) else None,
                setup_time_ms=median_value("setup_time_ms"),
                solve_time_ms=median_value("solve_time_ms"),
                total_time_ms=median_value("total_time_ms"),
                converged=complete, status="converged" if complete else "failed",
                successful_repetitions=sum(r["converged"] for r in timed),
                failed_repetitions=sum(not r["converged"] for r in timed),
                warmup_converged=len(warmups) == 1 and warmups[0]["converged"])
    if records and records[0].get("coefficient") == "exponential":
        for key in ("coefficient", "a_scale", "chi", "a_contrast", "a_nodes_contrast"):
            summary[key] = records[0][key]
    return summary


def save_plots(rows: list[dict], output: Path, methods=METHODS,
               problem_title="normal-contrast 2D reaction-diffusion") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    for metric, filename, ylabel, title in (
        ("iterations", "iteration_scaling.png", "PCG iterations", "Iteration scaling"),
        ("total_time_ms", "runtime_scaling.png", "Median total time (ms)", "Runtime scaling"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for method, marker in zip(methods, ("o", "s", "^", "D")):
            values = [r for r in rows if r["solver"] == solver_name(method)]
            # A failed case is a gap, rather than a misleading valid curve point.
            yy = [r[metric] if r["converged"] else np.nan for r in values]
            ax.plot([r["n"] for r in values], yy, marker=marker,
                    label=solver_name(method))
        ax.set(xlabel="Interior grid size n", ylabel=ylabel,
               title=f"{title}: {problem_title}")
        ax.set_xticks(SIZES)
        if metric == "iterations":
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-boomer", action="store_true", help="add optional native hypre BoomerAMG-PCG")
    parser.add_argument("--coefficient", choices=("normal", "exponential"), default="normal")
    parser.add_argument("--a-scale", type=float, default=1.0, help="a at x=0 for exponential stratification")
    parser.add_argument("--chi", type=float, help="domain contrast >=1; required for exponential coefficient")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.coefficient == "exponential" and args.chi is None:
        parser.error("--coefficient exponential requires --chi")
    if args.coefficient == "normal" and (args.chi is not None or args.a_scale != 1.0):
        parser.error("--a-scale and --chi require --coefficient exponential")
    if args.coefficient == "exponential":
        try:
            validate_exponential_parameters(args.a_scale, args.chi)
        except ValueError as exc:
            parser.error(str(exc))
    default_output = PROJECT / ("reaction_results/grid_scaling_boomer" if args.include_boomer
                                else "reaction_results/grid_scaling")
    problem_title = "normal-contrast 2D reaction-diffusion"
    if args.coefficient == "exponential":
        default_output = PROJECT / ("reaction_results/grid_scaling_exponential_boomer" if args.include_boomer
                                    else "reaction_results/grid_scaling_exponential")
        problem_title = f"exponential stratification (chi={args.chi:g})"
    output = (args.output_dir or default_output).resolve()
    methods = METHODS + (("boomer",) if args.include_boomer else ())
    if output.exists():
        parser.error("output directory already exists; use --output-dir with a fresh path")
    output.mkdir(parents=True)
    source_names = ("grid.py", "operators.py", "lgf_inverse.py", "lgf_bank.py",
                    "solvers.py", "multigrid.py", "reaction_manufactured.py", Path(__file__).name)
    if args.include_boomer:
        source_names += ("boomer_amg.py", "native/boomer_bridge.c", "build_boomer.py")
    source_hashes = {name: hashlib.sha256((PROJECT / name).read_bytes()).hexdigest()
                     for name in source_names}
    versions = {}
    for name in ("numpy", "scipy", "pyamg", "matplotlib", "pytest", "torch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    metadata = dict(
        experiment="normal-contrast 2D reaction-diffusion grid scaling",
        started_utc=datetime.now(timezone.utc).isoformat(), sizes=list(SIZES),
        forcing="continuous", domain=[0.0, 1.0], dtype="float64", initial_guess="zero",
        tolerance=TOL, maxiter=MAXITER, warmups_per_case=1, repetitions=REPETITIONS,
        execution="sequential, grid then solver then repetition; fresh setup every execution",
        timing="Existing perf_counter instrumentation: setup includes validation, assembly where "
               "needed and preconditioner/factorization; solve includes PCG and true residual checks. "
               "Each total=setup+solve. Setup, solve and per-run total medians are computed independently. "
               "MMS generation, external residual validation, plotting and I/O excluded. "
               "Raw call wall time additionally includes wrapper/return overhead.",
        summary_convention="Iterations: median; relative residual: maximum of five timed runs; "
                           "warmup excluded from statistics. All runs must converge for case success. "
                           "No failed runs are discarded; missing timings produce null medians.",
        python=sys.version, executable=sys.executable, platform=platform.platform(),
        machine=platform.machine(), processor=platform.processor(), logical_cpus=os.cpu_count(),
        thread_environment={key: os.environ.get(key) for key in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
        versions=versions,
        source_sha256=source_hashes, grids=[], results=[], runs=[], status="preflight",
    )
    def save_summary():
        (output / "grid_scaling_summary.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n")

    if args.coefficient == "exponential":
        metadata["experiment"] = "exponential-stratification 2D reaction-diffusion grid scaling"
        metadata["coefficient"] = dict(family="exponential", formula="a_scale*chi**x",
                                      a_scale=args.a_scale, chi=args.chi)

    if args.include_boomer:
        from .boomer_amg import BoomerUnavailableError, boomer_metadata
        try:
            metadata["boomer_amg"] = boomer_metadata()
            metadata["versions"]["hypre"] = metadata["boomer_amg"]["hypre_version"]
        except BoomerUnavailableError as exc:
            metadata.update(status="dependency_unavailable", message=str(exc))
            save_summary()
            raise SystemExit(str(exc))

    metadata["preflight_tests"] = run_preflight(output)
    if not metadata["preflight_tests"]["successful"]:
        metadata["status"] = "preflight_failed"
        save_summary()
        raise SystemExit(f"Original 46 tests did not all pass; inspect {output / 'preflight_tests.log'}")
    print("[preflight] original 46 tests passed, no skips", flush=True)
    if args.include_boomer:
        metadata["boomer_tests"] = run_preflight(
            output, test_files=("tests/test_boomer_amg.py",), prefix="boomer_tests", expected=12)
        if not metadata["boomer_tests"]["successful"]:
            metadata["status"] = "boomer_preflight_failed"
            save_summary()
            raise SystemExit(f"BoomerAMG verification failed; inspect {output / 'boomer_tests.log'}")
        print("[preflight] 12 native BoomerAMG tests passed, no skips", flush=True)
    metadata["status"] = "running"
    save_summary()
    for n in SIZES:
        problem = make_reaction_problem(n, forcing="continuous", coefficient=args.coefficient,
                                        a_scale=args.a_scale, chi=args.chi)
        grid_metadata = validate_grid(problem)
        metadata["grids"].append(grid_metadata)
        for method in methods:
            records = [run_once(problem, method, "warmup", 0)]
            for repetition in range(1, REPETITIONS + 1):
                records.append(run_once(problem, method, "timed", repetition))
            if input_hashes(problem) != grid_metadata["input_sha256"]:
                raise RuntimeError("A solver modified the shared problem inputs")
            row = summarize_case(n, method, records)
            metadata["runs"].extend(records)
            metadata["results"].append(row)
            save_summary()
            total = row["total_time_ms"]
            print(f"n={n:3d} DOF={n*n:5d} {row['solver']} iterations={row['iterations']} "
                  f"residual={row['relative_residual']} total_ms={total} status={row['status']}", flush=True)
            for record in records:
                if not record["converged"]:
                    print(f"  FAILED {record['phase']} {record['repetition']}: {record['message']}", flush=True)
    rows = metadata["results"]
    with (output / "grid_scaling_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_plots(rows, output, methods=methods, problem_title=problem_title)
    metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["status"] = "complete" if all(r["converged"] for r in rows) else "failed"
    metadata["source_unchanged_during_experiment"] = all(
        hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() == value
        for name, value in source_hashes.items())
    save_summary()
    print(f"Results: {output}", flush=True)
    if metadata["status"] != "complete" or not metadata["source_unchanged_during_experiment"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
