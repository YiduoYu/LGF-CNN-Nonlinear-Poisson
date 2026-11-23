# 2D reaction–diffusion grid scaling

This experiment measures the existing normal-contrast LGF-PCG, GMG-PCG and
AMG-PCG implementations at n=31,63,127,255 (DOF=n²). It keeps the continuous
MMS forcing, flux discretization, homogeneous Dirichlet conditions, float64,
zero initial guess, tolerance 1e-10 and maximum 500 iterations unchanged.

## Reproduce

From the directory containing `axy_lgf/`, in an environment with the classical
dependencies, PyAMG and PyTorch (needed for all 46 original tests):

```bash
python -m axy_lgf.run_grid_scaling
```

The command first runs the original 46 tests, requiring zero skips/failures,
then executes the benchmark. For each solver and grid it rebuilds the entire
setup once for warmup and five times for measurement. All 72 solver executions
run sequentially in one process. The warmup is excluded from the statistics.

The default output is `axy_lgf/reaction_results/grid_scaling/`. The runner
refuses an existing output directory. To repeat without overwriting a run:

```bash
python -m axy_lgf.run_grid_scaling \
  --output-dir axy_lgf/reaction_results/grid_scaling_repeat_01
```

## Timing and reporting

The existing solver instrumentation uses `time.perf_counter()`. Setup includes
validation, reference inverse or multigrid hierarchy construction, sparse
assembly where needed, and coarse factorization. Solve time includes PCG and
its final true-residual checks. Every execution uses a freshly constructed
preconditioner. MMS creation, the benchmark's independent residual checks,
plotting and file output are outside the timed sections.

For each measured execution, total=setup+solve. The median of the five totals
is computed directly; it need not equal the sum of the separately computed
setup and solve medians. Raw solver-call wall times are also retained in JSON
and include small wrapper/return overhead absent from setup+solve.

The table reports the median iteration count and the maximum relative
algebraic residual across the five measured runs. All individual iteration
counts/residuals and warmup diagnostics are in JSON. A warmup or measured
failure makes the case unsuccessful. Failed runs remain in the data; missing
timings produce null medians. Failures cause a nonzero process exit status.

Files include the requested CSV, summary JSON, iteration plot and runtime
plot, plus the original-test log/XML. Source hashes, shared-input hashes,
coefficient ranges, matrix-free/sparse agreement, library versions, thread
environment and machine information are recorded for reproducibility.

The timing sample covers one machine, one smooth coefficient field and one
RHS family. Fixed solver order, system load, processor frequency, allocation
and cache effects can influence short runs. These results do not establish a
universal ranking or predict high-contrast convergence.

## Measured results — 2026-10-24

Hardware: Apple M1 Pro, 16 GiB RAM, 8 logical CPUs (read using macOS sysctl). All solver measurements were sequential in the same process; no other agent ran computations during timing. Native-library threading was left at its existing defaults; OMP/BLAS/Accelerate thread-count environment overrides were unset.

Python 3.12.14; numpy 2.3.5, scipy 1.18.1, pyamg 5.3.0, matplotlib 3.11.2, pytest 9.1.1, torch 2.14.1.

Before measurement, the original **46 tests passed with zero failures/errors/skips** (9.92 s). The three new aggregation/failure-reporting tests also passed (4.37 s). All **12 warmups and 60 measured solves converged**. Every sampled diffusion field had minimum 0.8, maximum 1.2 and contrast 1.5; q stayed within [0.8,1.2]. Sparse/matrix-free relative action differences were at most 1.475e-16. Shared coefficient/forcing hashes were unchanged after each method.

| n | DOF | Solver | Iterations | Relative residual | Setup (ms) | Solve (ms) | Total (ms) |
|---:|---:|---|---:|---:|---:|---:|---:|
| 31 | 961 | LGF-PCG | 11 | 4.9444e-11 | 0.088 | 1.209 | 1.325 |
| 31 | 961 | GMG-PCG | 10 | 4.7090e-11 | 1.203 | 2.315 | 3.643 |
| 31 | 961 | AMG-PCG | 9 | 1.6004e-11 | 2.074 | 2.224 | 4.488 |
| 63 | 3969 | LGF-PCG | 11 | 6.0008e-11 | 0.122 | 1.987 | 2.107 |
| 63 | 3969 | GMG-PCG | 10 | 6.1377e-11 | 1.995 | 3.838 | 5.799 |
| 63 | 3969 | AMG-PCG | 9 | 2.5666e-11 | 4.578 | 4.095 | 8.724 |
| 127 | 16129 | LGF-PCG | 11 | 6.2974e-11 | 0.276 | 5.452 | 5.743 |
| 127 | 16129 | GMG-PCG | 10 | 8.2337e-11 | 5.214 | 10.619 | 15.528 |
| 127 | 16129 | AMG-PCG | 9 | 4.3853e-11 | 13.305 | 12.326 | 25.792 |
| 255 | 65025 | LGF-PCG | 11 | 6.3809e-11 | 0.869 | 20.483 | 21.348 |
| 255 | 65025 | GMG-PCG | 10 | 9.9794e-11 | 16.052 | 36.404 | 52.456 |
| 255 | 65025 | AMG-PCG | 9 | 7.4510e-11 | 47.245 | 44.618 | 91.255 |

All timing columns are medians of five measured executions. Residual is the worst measured residual; all five iteration counts were identical within each case. Each case had five successful measured executions and a successful warmup. The separate medians are not additive.

![Iteration scaling](reaction_results/grid_scaling/iteration_scaling.png)

![Runtime scaling](reaction_results/grid_scaling/runtime_scaling.png)

## Interpretation

1. The iteration counts remained exactly 11 (LGF), 10 (GMG), and 9 (AMG) on all four grids and all five repetitions. This is evidence of mesh-insensitive convergence for this specific smooth normal-contrast problem.
2. Setup, solve and total medians increased with refinement for every method. From n=127 to n=255, DOF increased by about 4.03 times, while the total median increased by 3.72 times (LGF-PCG), 3.38 times (GMG-PCG), 3.54 times (AMG-PCG). Four modest grid sizes are insufficient to establish an asymptotic cost model.
3. LGF had the smallest setup and total medians in these measurements. AMG used fewer iterations but incurred more setup time and a larger total time. At n=255, AMG setup and solve medians were about 47.25 and 44.62 ms, respectively; iteration count alone does not describe the total cost.
4. All true algebraic residuals passed 1e-10. The largest was 9.9794390e-11 (GMG at n=255), close to the stopping threshold. No result was filtered out as a failure.
5. Small cases are sensitive to overhead and timing variation: for example, the five n=31 LGF totals ranged from about 1.216 to 1.636 ms. There was one warmup and five repeats per case, no CPU affinity, no fixed processor frequency, and no uncertainty estimate across independent sessions. This experiment does not rank the methods universally or isolate memory/cache costs.

The baseline is ready to serve as the verified numerical and performance control for a subsequent high-contrast study. Its unchanged operator, common residual criterion, successful test suite, repeatable experiment wrapper and preserved records provide that foundation. High-contrast robustness and runtime remain untested; the present convergence behavior must not be extrapolated to those coefficients. No high-contrast experiments were run.

## Saved evidence and preservation

- [Complete CSV table](reaction_results/grid_scaling/grid_scaling_results.csv)
- [Summary JSON and all 72 raw runs](reaction_results/grid_scaling/grid_scaling_summary.json)
- [Original 46-test log](reaction_results/grid_scaling/preflight_tests.log)
- [JUnit test record](reaction_results/grid_scaling/preflight_tests.xml)
- [Preservation check: 269 existing files, zero changed](reaction_results/grid_scaling/preservation_check.json)

Only three new source/document files were added: `run_grid_scaling.py` (benchmark wrapper and plots), `tests/test_grid_scaling.py` (three aggregation/failure tests), and this report. Experiment files are under the separate `reaction_results/grid_scaling/` directory. All existing files, including solver/preconditioner/discretization code, original tests, `VERIFICATION.md`, checkpoints and prior results, remain byte-for-byte unchanged.

Failed solver runs: **0**. Failed tests: **0**. Missing dependencies in the measurement environment: **0**. Unresolved experiment issues: **none identified**. The performance limitations listed above still apply.

## Exact local reproduction command

From the current `axy_lgf/` directory, the environment created for this run can be reused:

```bash
source /tmp/axy-grid-scaling-venv/bin/activate
export MPLCONFIGDIR=/tmp/axy-grid-scaling-mpl
export XDG_CACHE_HOME=/tmp/axy-grid-scaling-cache
cd ..
python -m axy_lgf.run_grid_scaling \
  --output-dir axy_lgf/reaction_results/grid_scaling_repeat_01
```

The original execution used `python -m axy_lgf.run_grid_scaling` without an output override. That directory now contains this completed experiment, so repetitions require a fresh path. The script automatically reruns the original 46 tests before timing.

The temporary environment may eventually be removed. Recreate it with Python 3.12 and the measured primary dependency versions:

```bash
python3.12 -m venv /tmp/axy-grid-scaling-venv
source /tmp/axy-grid-scaling-venv/bin/activate
python -m pip install numpy==2.3.5 scipy==1.18.1 pyamg==5.3.0 \
  matplotlib==3.11.2 pytest==9.1.1 torch==2.14.1
```

To check the additional wrapper tests from the package parent:

```bash
python -m pytest -q axy_lgf/tests/test_grid_scaling.py
```
