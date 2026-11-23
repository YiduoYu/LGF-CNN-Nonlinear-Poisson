# Verified 2D normal-contrast reaction–diffusion baseline

Verified on 2026-10-19. The supplied working directory already contained the extracted project; the extension was applied there. All reported runs use float64, the requested unit-square MMS, zero Dirichlet data, zero initial guesses, tolerance `1e-10`, and a 500-iteration limit.

## Actual outcome

**46 tests passed, zero failed, zero skipped** in the final run (2.33 seconds). All 12 continuous-MMS solves and all 12 discrete-MMS solves converged. Matrix-free/sparse agreement, target and preconditioner SPD, boundary-face coefficients, and second-order accuracy all passed. No high-contrast study, 3D work, neural-network modification, or training was performed.

- [Final test log](reaction_results/verification_20261019T072431Z_e1800c4a/pytest.log)
- [JUnit test results](reaction_results/verification_20261019T072431Z_e1800c4a/pytest.xml)
- [Legacy solver and v8 checkpoint comparison](reaction_results/verification_20261019T072431Z_e1800c4a/compatibility.json)
- [Continuous MMS CSV](reaction_results/20261019T072336Z_continuous_c4ebfa72/results.csv)
- [Continuous MMS JSON, parameters, versions and source hashes](reaction_results/20261019T072336Z_continuous_c4ebfa72/results.json)
- [Discrete MMS CSV, all four grids](reaction_results/20261019T072337Z_discrete_04d936d4/results.csv)
- [Discrete MMS JSON](reaction_results/20261019T072337Z_discrete_04d936d4/results.json)

## Environment and exact commands

Python: `3.12.14 (main, Aug 25 2026, 13:50:33) [Clang 22.1.3 ]`. Platform: `macOS-27.0.1-arm64-arm-64bit`.
Numerical dependencies: numpy 2.3.5, scipy 1.18.1, pyamg 5.3.0, matplotlib 3.11.2.

The default system Python had no scientific dependencies; an existing Conda SciPy installation failed to load a compiled library. Verification therefore used a temporary isolated environment, with NumPy inherited from the bundled runtime and the other dependencies installed there. No project dependency or solver remains missing in this environment. The following commands work from the current package directory:

```bash
source /tmp/axy-rd-venv/bin/activate
export MPLCONFIGDIR=/tmp/axy-rd-mpl
export XDG_CACHE_HOME=/tmp/axy-rd-cache
python -m pytest -q
cd ..
python -m axy_lgf.run_reaction_diffusion --n 63
python -m axy_lgf.run_reaction_diffusion --n 63 --forcing discrete
python -m axy_lgf.run_reaction_diffusion --conv
```

The temporary environment may be removed by the operating system. To recreate an environment, install `requirements-classical.txt` and optional `pyamg` as documented in README. PyTorch is optional for the classical baseline; it was present for the parity and checkpoint checks. The test suite explicitly simulates missing PyAMG and verifies the installation message and unavailable status, and verifies CLI exit code 2 for nonconvergence.

## Continuous MMS convergence

All three methods produce the same errors to the displayed precision. Error is relative discrete L2 error against the sampled continuous exact solution; `p` uses successive grid spacings. These are continuous-forcing results, independently of the discrete-consistency tests.

| n | h | a min / max | actual contrast | q min / max | relative L2 error | p |
|---:|---:|---:|---:|---:|---:|---:|
| 15 | 0.0625000 | 0.8 / 1.2 | 1.500000 | 0.8152241 / 1.2 | 3.058747621e-03 | — |
| 31 | 0.0312500 | 0.8 / 1.2 | 1.500000 | 0.8038429 / 1.2 | 7.625100527e-04 | 2.004113 |
| 63 | 0.0156250 | 0.8 / 1.2 | 1.500000 | 0.8009631 / 1.2 | 1.904914751e-04 | 2.001030 |
| 127 | 0.0078125 | 0.8 / 1.2 | 1.500000 | 0.8002409 / 1.2 | 4.761436626e-05 | 2.000258 |

![Continuous MMS convergence](reaction_results/20261019T072336Z_continuous_c4ebfa72/convergence.png)

The three solution-error curves overlap because each solver reaches algebraic error far below the spatial error. The measured rates approach two. The plotted reference line has slope two; the table and automated tests provide the numerical evidence.

## Solver comparison: continuous forcing

Every row below converged with a recomputed relative algebraic residual at most `1e-10`. Setup and solve times are separate; total is their sum. Measurements are single runs, without statistical benchmarking. The first AMG setup includes importing PyAMG. The solve time includes matrix-free actions and residual checks. No speed ranking is inferred from iteration counts.

| n | solver / V-cycle or LGF | iterations | final relative algebraic residual | setup ms | solve ms | total ms |
|---:|---|---:|---:|---:|---:|---:|
| 15 | LGF-PCG | 11 | 2.4907e-11 | 0.553 | 2.985 | 3.538 |
| 15 | GMG-PCG | 10 | 2.0630e-11 | 3.552 | 3.472 | 7.024 |
| 15 | AMG-PCG | 9 | 9.8312e-12 | 89.568 | 1.613 | 91.181 |
| 31 | LGF-PCG | 11 | 4.9444e-11 | 0.098 | 1.243 | 1.341 |
| 31 | GMG-PCG | 10 | 4.7090e-11 | 1.256 | 2.153 | 3.409 |
| 31 | AMG-PCG | 9 | 1.6004e-11 | 2.133 | 2.698 | 4.831 |
| 63 | LGF-PCG | 11 | 6.0008e-11 | 0.170 | 2.072 | 2.242 |
| 63 | GMG-PCG | 10 | 6.1377e-11 | 2.164 | 4.332 | 6.496 |
| 63 | AMG-PCG | 9 | 2.5666e-11 | 4.456 | 4.468 | 8.924 |
| 127 | LGF-PCG | 11 | 6.2974e-11 | 0.383 | 5.508 | 5.891 |
| 127 | GMG-PCG | 10 | 8.2337e-11 | 5.819 | 10.931 | 16.750 |
| 127 | AMG-PCG | 9 | 4.3853e-11 | 13.644 | 12.770 | 26.414 |

## Discrete MMS consistency

These use `f_h=A_h*u_exact`, so they test algebraic recovery. No spatial convergence order is assigned. Full timing and coefficient columns are in the CSV/JSON.

| n | solver | iterations | final relative algebraic residual | relative L2 solution error |
|---:|---|---:|---:|---:|
| 15 | LGF-PCG | 11 | 2.4962e-11 | 5.7138e-13 |
| 15 | GMG-PCG | 10 | 2.1018e-11 | 8.1349e-13 |
| 15 | AMG-PCG | 9 | 9.8078e-12 | 2.5732e-13 |
| 31 | LGF-PCG | 11 | 4.9470e-11 | 3.1952e-13 |
| 31 | GMG-PCG | 10 | 4.7146e-11 | 1.0703e-12 |
| 31 | AMG-PCG | 9 | 1.6002e-11 | 1.7556e-13 |
| 63 | LGF-PCG | 11 | 6.0018e-11 | 2.9354e-13 |
| 63 | GMG-PCG | 10 | 6.1425e-11 | 4.6969e-13 |
| 63 | AMG-PCG | 9 | 2.5665e-11 | 1.1642e-13 |
| 127 | LGF-PCG | 11 | 6.2975e-11 | 2.8766e-13 |
| 127 | GMG-PCG | 10 | 8.2361e-11 | 2.3172e-13 |
| 127 | AMG-PCG | 9 | 4.3846e-11 | 8.3813e-14 |

Maximum discrete relative solution error: `1.0703e-12`.

## Legacy compatibility

The original source files were copied to a temporary directory before editing and imported as a separate package for comparison. At `n=15`, `smooth_sine`, discrete forcing, the old PCG/Richardson/LGF-bank solutions were bit-for-bit equal to the new default solutions, with identical iteration counts 18/29/21. Richardson and bank residual histories were identical. PCG final residual reporting changed by at most `9.99e-16` because it is now recomputed from `f-A_h*u`.

Both existing v8 checkpoints (physics and supervised, n=32) loaded with the unchanged v8 loader. With input seed 487, both default forward outputs were bit-for-bit equal between original and current source; maximum output difference was zero. Checkpoint paths and hashes are in `compatibility.json`. A v5 checkpoint was initially selected and rejected by the original version check; a supported v8 checkpoint was then used. Old checkpoint files were preserved.

The following original commands were also executed successfully:

```bash
python -m axy_lgf.run_demo --solver pcg
python -m axy_lgf.run_demo --solver richardson
python -m axy_lgf.run_demo --solver lgf_bank
python -m axy_lgf.run_demo --conv --problem smooth_sine
```

The legacy default n=63 analytic solve returned convergence for all three solvers (19, 30, 44 iterations respectively), with L2 error `5.181e-4`. The constant-coefficient DST check gave L2 error `9.746e-15`. Legacy analytic refinement orders were approximately 1.96, 1.97, 1.99; those older boundary rules are retained by default.

## Files changed

| File | Purpose |
|---|---|
| `operators.py` | Share face coefficients; add optional boundary-node samples, reaction action, validation and identical sparse assembly. |
| `solvers.py` | Add keyword-only reaction/boundary inputs; screened reference; share PCG iteration and true-residual success check; timing fields. |
| `__init__.py` | Lazy public CNN imports so classical runs need only NumPy/SciPy. |
| `reaction_manufactured.py` (new) | Requested exact solution, moderate coefficients, analytic forcing, separate discrete forcing. |
| `multigrid.py` (new) | Genuine geometric V-cycle and optional symmetric AMG configuration with common PCG. |
| `run_reaction_diffusion.py` (new) | Independent CLI, unified comparison, machine-readable results and convergence plot. |
| `tests/conftest.py`, `tests/test_reaction_diffusion.py`, `tests/test_torch_compatibility.py` (new) | 46 automated verification and regression cases; optional dependency skips. |
| `requirements-classical.txt` (new) | Classical dependencies and optional PyAMG/PyTorch instructions. |
| `README.md`, `VERIFICATION.md` | Mathematical conventions, commands, measured results and compatibility evidence. |
| `reaction_results/` (new) | Isolated experiment outputs, provenance, test reports and figures. |

`grid.py`, `manufactured.py`, `lgf_inverse.py`, `lgf_bank.py`, `run_demo.py`, all PyTorch/network/training/evaluation source files were checked byte-for-byte unchanged against the pre-edit snapshot. Existing figures, checkpoints and experimental results were preserved.

## Numerical definitions and limitations

`A_h*u=-div_h(a*grad_h(u))+q*u` uses the original conservative flux form plus positive nodal reaction. The new baseline averages adjacent nodal diffusion values on every edge, including sampled Dirichlet boundary nodes. Old calls without `a_nodes` retain their copied-interior boundary coefficients. The matrix is symmetric positive definite by its positive edge energy plus nonnegative nodal reaction; sparse and matrix-free versions share the same faces.

LGF exactly inverts `P_h=mean(a)*(-Delta_h)+mean(q)*I` through the original orthonormal DST-I denominator. It approximates the variable-coefficient target and reduces to the pure-Poisson reference when `q=0`. GMG uses geometric bilinear `P`, `R=P.T`, Galerkin coarse systems, two pre/post damped Jacobi sweeps and Cholesky; AMG uses transpose restriction, paired forward/backward GS and Cholesky. Both preconditioners are fixed SPD-compatible maps used with CG.

The residual is `||f-A_h*u||_2/||f||_2`, also equal to the ratio of discrete L2 norms on the same uniform grid. It measures algebraic system accuracy. Continuous MMS solution error measures discretization accuracy after algebraic error has been reduced. No continuous PDE residual is claimed.

Failed tests: **none**. Missing dependencies in the verification environment: **none**. Unresolved correctness issues: **none identified**. GMG is restricted to `n=2^k-1`, `k>=2`; PyAMG remains optional. Timing results are not a performance benchmark. The verification is limited to the requested smooth normal-contrast 2D linear problem and backward compatibility, and does not establish high-contrast, 3D or nonlinear behavior.
