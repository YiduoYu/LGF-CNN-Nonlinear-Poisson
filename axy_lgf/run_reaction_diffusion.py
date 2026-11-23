"""Normal-contrast linear reaction-diffusion MMS and solver comparison.

Run from the package parent: python -m axy_lgf.run_reaction_diffusion --conv
Existing run_demo commands retain their original domain and equation.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from uuid import uuid4

import numpy as np

from .multigrid import AMGUnavailableError, solve_multigrid_pcg
from .operators import apply_reaction_diffusion, l2_norm
from .reaction_manufactured import make_reaction_problem, validate_exponential_parameters
from .solvers import solve_axy_pcg

METHODS = ("lgf", "gmg", "amg")
PRECONDITIONERS = {
    "lgf": "DST-I: (mean(a)*(-Delta_h)+mean(q)*I)^-1",
    "gmg": "geometric Galerkin V-cycle, 2+2 damped Jacobi, Cholesky coarse",
    "amg": "PyAMG Ruge-Stuben V-cycle, forward/backward GS, Cholesky coarse",
}


def run_experiment(sizes, methods, *, forcing="continuous", tol=1e-10, maxiter=500,
                   coefficient="normal", a_scale=1.0, chi=None):
    """Every method receives identical fields, zero initial guess and tolerance."""
    rows = []
    previous = {}
    for n in sizes:
        problem = make_reaction_problem(n, forcing, coefficient=coefficient, a_scale=a_scale, chi=chi)
        grid = problem.grid
        for method in methods:
            row = {
                "n": n, "h": grid.h, "forcing": forcing,
                "a_min": float(problem.a.min()), "a_max": float(problem.a.max()),
                "a_contrast": float(problem.a.max() / problem.a.min()),
                "q_min": float(problem.q.min()), "q_max": float(problem.q.max()),
                "solver": f"{method.upper()}-PCG", "preconditioner": PRECONDITIONERS[method],
                "tol": tol, "maxiter": maxiter, "initial_guess": "zero",
                "a0": float(problem.a.mean()) if method == "lgf" else None,
                "q0": float(problem.q.mean()) if method == "lgf" else None,
                "iterations": None, "converged": False, "status": "unavailable",
                "relative_algebraic_residual": None, "relative_l2_error": None,
                "l2_error": None, "observed_order": None,
                "setup_time_s": None, "solve_time_s": None, "total_time_s": None,
                "message": "",
            }
            if coefficient == "exponential":
                row.update(coefficient=coefficient, a_scale=a_scale, chi=chi,
                           a_nodes_min=float(problem.a_nodes.min()),
                           a_nodes_max=float(problem.a_nodes.max()),
                           a_nodes_contrast=float(problem.a_nodes.max() / problem.a_nodes.min()))
            kwargs = dict(q=problem.q, a_nodes=problem.a_nodes,
                          tol=tol, maxiter=maxiter, u0=grid.zeros())
            try:
                if method == "lgf":
                    result = solve_axy_pcg(problem.a, problem.f, grid, **kwargs)
                else:
                    result = solve_multigrid_pcg(problem.a, problem.f, grid,
                                                 method=method, **kwargs)
            except AMGUnavailableError as exc:
                row["message"] = str(exc)
                rows.append(row)
                print(f"[unavailable] {row['solver']}: {exc}")
                continue
            residual = problem.f - apply_reaction_diffusion(
                result.u, problem.a, grid, problem.q, a_nodes=problem.a_nodes,
            )
            relative_residual = l2_norm(residual, grid) / l2_norm(problem.f, grid)
            error = l2_norm(result.u - problem.u_exact, grid)
            relative_error = error / l2_norm(problem.u_exact, grid)
            converged = result.converged and relative_residual <= tol
            order = None
            if forcing == "continuous" and converged:
                if method in previous:
                    old_h, old_error = previous[method]
                    order = float(np.log(old_error / error) / np.log(old_h / grid.h))
                previous[method] = (grid.h, error)
            row.update(
                iterations=result.iterations, converged=converged,
                status="converged" if converged else "not_converged",
                relative_algebraic_residual=relative_residual,
                relative_l2_error=relative_error, l2_error=error, observed_order=order,
                setup_time_s=result.setup_time, solve_time_s=result.solve_time,
                total_time_s=result.setup_time + result.solve_time,
            )
            rows.append(row)
            rate = "--" if order is None else f"{order:.4f}"
            print(f"n={n:3d} h={grid.h:.6f} {row['solver']:7s} "
                  f"iter={result.iterations:3d} status={row['status']} "
                  f"relres={relative_residual:.3e} relL2={relative_error:.3e} p={rate} "
                  f"setup={result.setup_time:.4f}s solve={result.solve_time:.4f}s "
                  f"total={row['total_time_s']:.4f}s")
    return rows


def save_results(rows, output_root: Path, *, plot: bool = True) -> Path:
    """Use a fresh child directory for every run; never reuse old results."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = output_root / f"{stamp}_{rows[0]['forcing']}_{uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    versions = {}
    for name in ("numpy", "scipy", "pyamg", "matplotlib"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    source_files = ("grid.py", "operators.py", "lgf_inverse.py", "solvers.py",
                    "multigrid.py", "reaction_manufactured.py", Path(__file__).name)
    hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
              for name in source_files}
    payload = {
        "domain": [0.0, 1.0], "dtype": "float64", "boundary": "homogeneous Dirichlet",
        "face_convention": "arithmetic average of adjacent nodal a, including boundary nodes",
        "norm": "L2_h(v)=sqrt(h^2*sum(v^2)); relative residual equals Euclidean ratio",
        "forcing": rows[0]["forcing"], "versions": versions,
        "python": sys.version, "platform": platform.platform(), "source_sha256": hashes,
        "timing": "single run, no warmup/repeats; setup includes validation, assembly and preconditioner; "
                  "solve includes PCG and residual checks; total=setup+solve; excludes MMS and plotting",
        "status": "complete" if all(r["converged"] for r in rows) else "incomplete",
        "results": rows,
    }
    if rows[0].get("coefficient") == "exponential":
        payload["coefficient"] = dict(family="exponential", formula="a_scale*chi**x",
                                      a_scale=rows[0]["a_scale"], chi=rows[0]["chi"])
    (output / "results.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for method in METHODS:
            values = [r for r in rows if r["solver"] == f"{method.upper()}-PCG" and r["converged"]]
            if values:
                ax.loglog([r["h"] for r in values], [r["relative_l2_error"] for r in values],
                          marker="o", label=f"{method.upper()}-PCG", alpha=0.8)
        if rows[0]["forcing"] == "continuous":
            valid = [r for r in rows if r["converged"]]
            if valid:
                hh = np.array(sorted({r["h"] for r in valid}))
                scale = valid[0]["relative_l2_error"] / valid[0]["h"]**2
                ax.loglog(hh, scale * hh**2, "k--", label="O(h²)")
        ax.set(xlabel="Grid spacing h", ylabel="Relative discrete L² solution error",
               title=f"Reaction-diffusion MMS: {rows[0]['forcing']} forcing")
        ax.grid(True, which="both", alpha=0.3)
        if ax.lines:
            ax.legend()
        fig.tight_layout()
        fig.savefig(output / "convergence.png", dpi=180)
        plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=63)
    parser.add_argument("--conv", action="store_true", help="continuous MMS on 15,31,63,127")
    parser.add_argument("--forcing", choices=("continuous", "discrete"), default="continuous")
    parser.add_argument("--solvers", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--coefficient", choices=("normal", "exponential"), default="normal")
    parser.add_argument("--a-scale", type=float, default=1.0, help="a at x=0 for exponential stratification")
    parser.add_argument("--chi", type=float, help="domain contrast >=1; required for exponential coefficient")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "reaction_results")
    parser.add_argument("--no-plot", action="store_true", help="CSV/JSON only, no Matplotlib needed")
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
    if args.conv and args.forcing != "continuous":
        parser.error("--conv measures discretization error and requires --forcing continuous")
    sizes = (15, 31, 63, 127) if args.conv else (args.n,)
    if "gmg" in args.solvers and any(n < 3 or (n+1) & n for n in sizes):
        parser.error("GMG requires n=2^k-1, k>=2")
    print(f"[{args.coefficient}, {args.forcing} MMS] zero Dirichlet, (0,1)^2; "
          "relres is the discrete algebraic residual; relL2 is solution error")
    rows = run_experiment(sizes, args.solvers, forcing=args.forcing,
                          tol=args.tol, maxiter=args.maxiter, coefficient=args.coefficient,
                          a_scale=args.a_scale, chi=args.chi)
    output_root = args.output_dir
    if args.coefficient == "exponential" and output_root == Path(__file__).resolve().parent / "reaction_results":
        output_root = output_root / "exponential"
    output = save_results(rows, output_root, plot=not args.no_plot)
    print(f"Results: {output}")
    if not all(row["converged"] for row in rows):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
