# Optional hypre BoomerAMG baseline

Implemented and measured on 2026-10-26. This adds **hypre 3.1.0 BoomerAMG** as a fourth preconditioner for the existing PCG. The old `AMG-PCG` rows continue to mean PyAMG 5.3.0 Ruge–Stuben. The PDE, continuous MMS forcing, flux operator, boundary-face convention, outer PCG and residual criterion are unchanged.

## Implementation and fixed configuration

The official hypre source is compiled once as a serial CPU library with real float64 values, int32 indices, internal BLAS/LAPACK, and no MPI/OpenMP/GPU. A small C bridge transfers the existing SciPy CSR matrix and RHS/correction vectors to hypre. All AMG hierarchy construction, interpolation, relaxation and coarse solves are performed by hypre; there is no substitute Python AMG implementation under the BoomerAMG label.

Fixed configuration (no parameter search was performed):

| Parameter | Setting |
|---|---|
| coarsening | HMIS (10) |
| interpolation | extended+i (6) |
| P_max_elements | 4 |
| strong_threshold | 0.25 |
| aggressive_levels | 0 |
| restriction | P.T (0) |
| smoother | symmetric hybrid Gauss-Seidel (6) |
| relax_order | lexicographic (0) |
| pre_sweeps | 1 |
| post_sweeps | 1 |
| coarse_solver | Gaussian elimination (9) |
| max_coarse_size | 9 |
| max_levels | 25 |
| cycle | V (1) |
| inner_tolerance | 0.0 |
| inner_miniter | 1 |
| inner_maxiter | 1 |
| initial_correction | zero on every application |

With transpose restriction, symmetric hybrid Gauss–Seidel in natural ordering, and a direct coarse solve, this is a fixed SPD-compatible V-cycle for the real SPD target. Every application starts from zero and executes exactly one cycle (inner tolerance 0 and min/max iterations 1). Small-grid tests explicitly check linearity, symmetry, positive eigenvalues and independence from prior RHS applications.

The public hypre documentation requires symmetric smoothing with CG; the pinned v3.1.0 API header identifies HMIS as coarsening 10 and extended+i as interpolation 6:

- [hypre BoomerAMG user documentation](https://hypre.readthedocs.io/en/latest/solvers-boomeramg.html)
- [Pinned hypre 3.1.0 API and configuration definitions](https://github.com/hypre-space/hypre/blob/v3.1.0/src/parcsr_ls/HYPRE_parcsr_ls.h)
- [Official hypre 3.1.0 release](https://github.com/hypre-space/hypre/releases/tag/v3.1.0)

## Build and run

From the package parent, with the previously prepared Python environment:

```bash
source /tmp/axy-grid-scaling-venv/bin/activate
export MPLCONFIGDIR=/tmp/axy-grid-scaling-mpl
export XDG_CACHE_HOME=/tmp/axy-grid-scaling-cache
# First build only; C/C++ compiler and make are required:
python -m axy_lgf.build_boomer
# Run four solvers, all four grids, one warmup and five measurements:
python -m axy_lgf.run_grid_scaling --include-boomer
```

The native bridge is already built in `.deps/libaxy_boomer.dylib` on this machine. Building is outside all benchmark times. The build command refuses to overwrite an existing bridge; do not rebuild an existing installation unless deliberately replacing it. `.deps/` is ignored by Git.

The verified local build used the downloaded archive explicitly:

```bash
curl -fL -o /tmp/axy-hypre-3.1.0.tar.gz \
  https://github.com/hypre-space/hypre/archive/refs/tags/v3.1.0.tar.gz
python -m axy_lgf.build_boomer --archive /tmp/axy-hypre-3.1.0.tar.gz
```

The archive SHA-256 is pinned to `a6879ae9375d95c26afd97141d61e7a8092807333bf40cd180b385aed7351b2d`. The build manifest records compiler flags, compiler version and bridge/source hashes. No system package installation or MPI runtime is required. Missing native support produces a build-instruction error; it never silently falls back to PyAMG.

The completed comparison is saved in `reaction_results/grid_scaling_boomer/`. To repeat while preserving it:

```bash
python -m axy_lgf.run_grid_scaling --include-boomer \
  --output-dir axy_lgf/reaction_results/grid_scaling_boomer_repeat_01
```

Without `--include-boomer`, the existing three-solver workflow and default output path retain their previous behavior. All output paths must be fresh; existing results are never overwritten.

For direct use:

```python
from axy_lgf.boomer_amg import solve_boomer_pcg
result = solve_boomer_pcg(a, f, grid, q=q, a_nodes=a_nodes, tol=1e-10)
```

## Verification and measurement

Original 46 numerical/Torch tests: **46 passed, zero skips**. Existing three timing-wrapper tests: **3 passed**. Added native BoomerAMG tests: **12 passed, zero skips**. This is 61 passed tests across the validation calls. The native tests cover transferred-matrix actions, SPD/linearity/state independence, all four discrete MMS grids, continuous solution agreement with LGF, q=0, zero RHS, exact initial guesses, explicit nonconvergence, and missing-dependency instructions.

The four-method benchmark reran every method in the same process and session. All **16 warmups and 80 measured solves converged** at recomputed relative algebraic residual <=1e-10. Each case constructs a new preconditioner on every execution. Inputs remain the same continuous normal-contrast MMS (a in [0.8,1.2], contrast 1.5), unit square, homogeneous Dirichlet values, float64 and zero initial guess. The unchanged maximum iteration count is 500.

Timing boundaries match the earlier experiment: setup includes validation, common sparse assembly, SciPy-to-hypre CSR conversion/copy, native matrix/vector allocation, hierarchy setup and a zero-RHS application to include lazy native work. Solve includes the unchanged PCG, all residual/correction transfers, native V-cycles and outer true-residual checks. Per-solve destruction is outside setup+solve, as with the existing solver teardown. Raw call wall time is additionally recorded. Build/download time, problem generation, external validation, plotting and file output are excluded.

Each total is setup+solve for one execution; the table uses independent medians of setup, solve and per-execution totals. The marginal medians need not add. Iteration counts are median counts and the residual is the maximum of the five measured runs. No failed runs were excluded.

## Four-method comparison

All rows converged. Times are milliseconds.

| n | DOF | Solver | Iterations | Relative residual | Setup | Solve | Total |
|---:|---:|---|---:|---:|---:|---:|---:|
| 31 | 961 | LGF-PCG | 11 | 4.9444e-11 | 0.083 | 1.124 | 1.205 |
| 31 | 961 | GMG-PCG | 10 | 4.7090e-11 | 1.168 | 2.332 | 3.572 |
| 31 | 961 | AMG-PCG | 9 | 1.6004e-11 | 2.197 | 2.316 | 4.503 |
| 31 | 961 | BoomerAMG-PCG | 7 | 7.1015e-12 | 0.884 | 1.106 | 1.987 |
| 63 | 3969 | LGF-PCG | 11 | 6.0008e-11 | 0.131 | 2.016 | 2.137 |
| 63 | 3969 | GMG-PCG | 10 | 6.1377e-11 | 2.238 | 4.101 | 6.518 |
| 63 | 3969 | AMG-PCG | 9 | 2.5666e-11 | 4.478 | 4.356 | 8.972 |
| 63 | 3969 | BoomerAMG-PCG | 8 | 3.3284e-12 | 3.387 | 3.723 | 7.126 |
| 127 | 16129 | LGF-PCG | 11 | 6.2974e-11 | 0.287 | 5.885 | 6.165 |
| 127 | 16129 | GMG-PCG | 10 | 8.2337e-11 | 5.394 | 11.543 | 17.402 |
| 127 | 16129 | AMG-PCG | 9 | 4.3853e-11 | 13.369 | 12.429 | 25.749 |
| 127 | 16129 | BoomerAMG-PCG | 8 | 4.5232e-11 | 11.925 | 13.626 | 25.467 |
| 255 | 65025 | LGF-PCG | 11 | 6.3809e-11 | 0.878 | 21.835 | 22.705 |
| 255 | 65025 | GMG-PCG | 10 | 9.9794e-11 | 16.037 | 37.283 | 53.449 |
| 255 | 65025 | AMG-PCG | 9 | 7.4510e-11 | 47.025 | 45.070 | 92.095 |
| 255 | 65025 | BoomerAMG-PCG | 9 | 2.3769e-11 | 45.586 | 61.057 | 106.775 |

![Four-method iteration scaling](reaction_results/grid_scaling_boomer/iteration_scaling.png)

![Four-method runtime scaling](reaction_results/grid_scaling_boomer/runtime_scaling.png)

BoomerAMG iteration counts grew modestly from 7 to 9 across the grid sequence, while every run met the target tolerance. At n=31 and 63, this configuration used fewer iterations and a lower total median than PyAMG. At n=127 their totals were close (25.47 vs 25.75 ms); at n=255 BoomerAMG was slower (106.78 vs 92.10 ms) with the same nine outer iterations. LGF had the lowest total median in this specific four-method comparison. These observations are limited to the fixed configurations, coefficient field, RHS and machine.

This experiment includes the native-interface overhead and freshly rebuilt hierarchies. It does not measure parallel BoomerAMG, hierarchy reuse across multiple RHS vectors, or a tuned BoomerAMG configuration. No timing profile was collected to isolate the cause of the n=255 runtime difference, and no universal solver ranking is claimed. No high-contrast, 3D, neural-network or nonlinear experiments were added.

## Files and preservation

- `boomer_amg.py`: optional native preconditioner ownership and common-PCG solver interface.
- `native/boomer_bridge.c`: hypre IJ/ParCSR transfer and fixed BoomerAMG calls.
- `build_boomer.py`: checksum-verified, project-local build and build provenance.
- `tests/test_boomer_amg.py`: 12 native correctness/regression checks.
- `run_grid_scaling.py`: optional `--include-boomer`, fourth curve and native preflight/provenance.
- `.gitignore`: ignore `.deps/` build artifacts.
- `BOOMER_AMG.md`: this report.

Of 281 existing files checked before editing, only `run_grid_scaling.py` and `.gitignore` changed. All existing solver/discretization implementations, original tests, checkpoint files, previous figures, previous result CSV/JSON and historical verification reports remain byte-for-byte unchanged.

- [Comparison CSV](reaction_results/grid_scaling_boomer/grid_scaling_results.csv)
- [Summary, native version/settings and all 96 raw runs](reaction_results/grid_scaling_boomer/grid_scaling_summary.json)
- [Original 46-test log](reaction_results/grid_scaling_boomer/preflight_tests.log)
- [Native 12-test log](reaction_results/grid_scaling_boomer/boomer_tests.log)
- [Original-file preservation check](reaction_results/grid_scaling_boomer/preservation_check.json)

Failed solver runs: **0**. Failed tests: **0**. Missing dependencies in this verification environment: **0**. The optional native dependency must be built on another machine; the generated binary is platform-specific.
