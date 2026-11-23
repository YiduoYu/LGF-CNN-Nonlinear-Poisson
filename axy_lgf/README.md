# axy_lgf

Solves the linear variable-coefficient elliptic equation with homogeneous
Dirichlet boundary conditions:

\[
-\nabla\cdot\left(a(x,y)\nabla u\right)=f
\quad\text{in }(x_{\min},x_{\max})^2,
\qquad
u=0\quad\text{on }\partial\Omega.
\]

The project includes PCG, Richardson, an LGF filter bank, and an LGF-CNN.
The default domain is \([-1,1]^2\); use `--domain-min` and `--domain-max`
to change it. The Dirichlet LGF on a rectangle is implemented via DST-I
(no BAE needed).

## Environment

- Python 3.10+
- NumPy, SciPy, PyTorch
- Matplotlib (only needed for plotting)

All commands below should be run from the parent directory of `axy_lgf/`.

## Data and prediction convention

Two independent training data modes are supported, set by `--data-kind`:

| `--data-kind` | how training samples are generated |
|---|---|
| `mms` | sample \(u_*\), then compute \(f = A_h u_*\) |
| `solver` | sample \(f\), solve for \(u_*\) with a high-accuracy PCG |

The model is trained separately for each data mode (separate experiment
directories). But `--predict` always uses the same four fixed MMS
solutions u1–u4. All four share one random coefficient field, so the
comparison is about how the model handles different solution complexity,
not different coefficient fields.

In other words: an MMS-trained model and a solver-trained model are both
evaluated on the same u1–u4.

## Quick start

### Classical solvers

```bash
python -m axy_lgf.run_demo --solver pcg
python -m axy_lgf.run_demo --solver richardson
python -m axy_lgf.run_demo --solver lgf_bank
python -m axy_lgf.run_demo --conv --problem smooth_sine
```

### Training the two models

```bash
python -m axy_lgf.train_lgf_cnn \
  --data-kind mms --loss-mode supervised \
  --domain-min -1 --domain-max 1 \
  --epochs 120 --n 31 --K 6 --contrast 0.8

python -m axy_lgf.train_lgf_cnn \
  --data-kind solver --loss-mode physics \
  --domain-min -1 --domain-max 1 \
  --epochs 120 --n 31 --K 6 --contrast 0.8
```

Checkpoints land in:

```text
axy_lgf/checkpoints/lgf_cnn_axy/<experiment_params_and_hash>/
├── latest.json
└── runs/
    ├── <train_id_1>/
    │   ├── completed.json
    │   ├── best.pt
    │   ├── last.pt
    │   ├── history.csv
    │   └── figures/training_curve.png
    └── <train_id_2>/
        └── ...
```

The experiment directory name includes `mms` or `solver`, so the two runs
don't collide. Each epoch uses exactly `--samples-per-epoch` samples; any
leftover at the end is handled as a short batch. Reusing the same configuration
loads the newest completed training run. `--retrain` creates a new `train_id`;
it never overwrites the old checkpoint. Flat checkpoint directories produced
by earlier versions remain readable and are not moved or deleted.

Every evaluation or prediction creates a new immutable run. The training
experiment key groups related evaluations; each `eval_id` contains a UTC
timestamp, test seed, checkpoint digest, and random suffix:

```text
axy_lgf/result/<experiment_params_and_hash>/
├── latest.json
└── evaluations/
    ├── <eval_id_1>/
    │   ├── manifest.json
    │   ├── data/
    │   └── figures/
    └── <eval_id_2>/
        ├── manifest.json
        ├── data/
        │   ├── u1_u4_metrics.csv
        │   ├── u1_u4_summary.json
        │   ├── u1_u4_solver_comparison.csv
        │   ├── u1_u4_residual_history.csv
        │   ├── u1_u4_predictions.pt
        │   ├── heldout_metrics.csv
        │   ├── heldout_summary.json
        │   ├── heldout_solver_comparison.csv
        │   ├── heldout_residual_history.csv
        │   ├── heldout_predictions.pt
        │   └── generalization_sweeps.csv
        └── figures/
            ├── u1_centerline.png
            ├── u1_panel.png
            ├── ...
            ├── u4_centerline.png
            └── u4_panel.png
```

`manifest.json` records the evaluation parameters, immutable `train_id`,
relative training-artifact URIs, checkpoint path and SHA-256, code/environment
provenance, timestamps, status, and a size/hash inventory of every evaluation
artifact. Training history and its curve belong to the training run and are not
copied into every evaluation.
`latest.json` is only a small pointer to the most recently completed run.
Completed evaluation directories are never reused, supplemented, or copied
to the experiment root. The first successful evaluation creates
`<eval_id_1>`; the next creates `<eval_id_2>` and updates `latest.json` to
point to it while `<eval_id_1>` remains unchanged. A running evaluation
exists only as a hidden `.attempt-*` directory. A failed attempt is discarded
and does not change `latest.json`. Attempt locks let a later process remove
staging directories abandoned by a killed process without touching active
evaluations. Updates to `latest.json` are locked and monotonic, and a missing
or stale pointer is rebuilt from completed manifests. Dot-prefixed staging
directories are made visible after their atomic rename so macOS Finder shows
every completed run.

By default all four cases u1–u4 are plotted with a shared coefficient
field. Pick one with `--plot-case u1|u2|u3|u4`, or skip plotting with
`--no-plot`. Each choice creates a separate run, so a u1-only run cannot
inherit old u2–u4 figures. Axis limits come from the checkpoint's stored
domain bounds, not hardcoded constants.

### Comparing models on the same u1–u4

Point both commands at their respective `best.pt`:

```bash
python -m axy_lgf.train_lgf_cnn \
  --predict --predict-from \
  axy_lgf/checkpoints/lgf_cnn_axy/<mms_exp>/runs/<train_id>/best.pt \
  --test-seed 123 --output-dir results/predict_mms_model

python -m axy_lgf.train_lgf_cnn \
  --predict --predict-from \
  axy_lgf/checkpoints/lgf_cnn_axy/<solver_exp>/runs/<train_id>/best.pt \
  --test-seed 123 --output-dir results/predict_solver_model
```

Prediction doesn't need `--data-kind`. Without `--output-dir`, its unique
run goes below the checkpoint's training experiment under `--result-root`.
With `--output-dir results/predict_mms_model`, the run is created below
`results/predict_mms_model/evaluations/<eval_id>/`; the supplied directory
is an experiment root, not a reusable run directory.

### What's in the output files

`u1_u4_metrics.csv` and `heldout_metrics.csv` contain per-case relL2,
RMSE, MAE, max error, prediction/reference discrete residual, gradient
error, constant baseline, constant-coefficient Poisson baseline, RHS
RMS, and CNN inference time (ms). The corresponding `*_summary.json`
files give mean/std/min/max for each metric.

There's no boundary error column because the zero Dirichlet halo enforces
u=0 exactly — that metric would be identically zero.

For manufactured references, `*_solver_comparison.csv` compares LGF-CNN,
Richardson (same iteration count), and high-accuracy PCG. `execution_mode`
is `fixed_budget` for LGF-CNN/Richardson and `tolerance_driven` for PCG.
`meets_target` applies one reporting threshold to all methods, while
`solver_converged` is empty for fixed-budget methods and records PCG's actual
stopping status. The PCG stopping tolerance and reporting target are separate
metadata fields. When `--data-kind solver` is used for the random held-out
batch, PCG is intentionally omitted because the same high-accuracy PCG
generated `u_exact`; rerunning it would compare the labels with themselves
rather than provide an independent baseline. The summary JSON records the
reference source and whether PCG is included.

`*_residual_history.csv` stores the available methods' per-step relative
residuals. `*_predictions.pt` saves the full coefficient field, RHS,
reference solution, network prediction, residual maps, and valid baseline
solutions. Timing is per-sample; CNN timing uses a plain forward pass
without diagnostic tensors so the comparison is fair.

### Generalization sweeps

The fixed u1–u4 and held-out batch both sit inside the training
distribution (\(p,q \in \{1,2,3\}\), N(0,1) amplitudes, single
contrast) — neither actually tests extrapolation. The `evaluate` command
runs three additional sweeps with same-budget Richardson as a reference.
Not every non-reference setting is strictly OOD, so each row has an explicit
`regime` instead of an ambiguous Boolean:

| sweep | values | regimes |
|---|---|---|
| `frequency` | \(p=q=1,2,3,4,5,6,8\) | `training_support` or `extrapolation` |
| `amplitude` | \(\|f\| \times 10^{-2}\dots 10^{2}\) | `reference` or `scale_stress` |
| `contrast` | \(0.2,0.5,0.8,0.95,1.0\) | `reference` or `coefficient_shift` |

Both commands need the same `--test-seed` so the coefficient field and
u1–u4 are identical.

### Standalone evaluation and plotting

```bash
python -m axy_lgf.train_lgf_cnn --eval-only path/to/best.pt --test-seed 123
```

Training uses a fixed validation set with `seed + 1`; final evaluation
defaults to `--test-seed 123` to keep them separate. `--eval-only` loads
the checkpoint and generates the training curve, u1–u4 panels, and
centerlines via `plot_lgf_cnn.py`. Use `--no-plot` to save only data.

## Math and implementation

The discrete operator uses a conservative flux-form 5-point stencil:

\[
A_h u = -\operatorname{div}_h(a\nabla_h u).
\]

Interior face coefficients are arithmetic averages of their two adjacent
nodes. A boundary face uses the adjacent interior coefficient, while the
solution outside the interior grid is the homogeneous Dirichlet value zero.
The NumPy and PyTorch implementations must represent this same operator.

The inverse of the constant-coefficient reference operator

\[
\left(a_0(-\Delta_h)+\lambda I\right)^{-1}
\]

is applied exactly by 2D orthogonal DST-I. Both the NumPy and PyTorch LGF
banks do one forward transform then batch-process all \(\lambda_j\) to
avoid redundant work. The PyTorch version uses the same orthonormal
DST-I sine basis as the NumPy implementation.

The classical LGF bank uses the same sample reference coefficient \(a_0\)
in its filters and local correction:

\[
\delta u = \omega\,\frac{a_0}{a}\sum_j\alpha_j G_{\lambda_j}r.
\]

LGF-CNN uses one fixed reference coefficient \(a_0=1\) by default. The
mixer uses the sample mean \(\bar a\) only to form scale-free input channels:

\[
\frac{a}{\bar a},\quad \frac{\bar a}{a},\quad
\frac{|r|}{\lVert r\rVert},\quad
\frac{|z_j|}{\max_k\lVert z_k\rVert}.
\]

The bank itself remains fixed at \(a_0=1\), and the learned correction is

\[
\delta u = \eta_{\max}\eta(x)\frac{1}{a(x)}
            \sum_j\alpha_j(x)G_{\lambda_j}r.
\]

Only the mixer features use magnitudes; the filtered correction keeps its
sign. Consequently \(u(a,-f)=-u(a,f)\) by construction, while positive
forcing-scale equivariance is preserved for the frozen-coefficient linear
map.

Random coefficient fields use `contrast` \(c\in[0,1]\) and the clipping rule
\[
a\in[1-0.95c,\;1+c].
\]
All functions with spatial frequencies are scaled by the domain length
\(L\).

## Mathematical compatibility contract: do not change silently

This section defines the legacy regression contract; the new automated tests
cover the classical extension and NumPy/PyTorch operator parity. Future changes
must preserve all items below unless the mathematical
model is deliberately revised, documented, versioned, and—when model
semantics change—retrained:

1. The PDE is the homogeneous-Dirichlet flux operator above, using arithmetic
   interior-face coefficients and the adjacent node value on boundary faces.
   NumPy and PyTorch paths must remain numerically consistent.
2. The default domain is \([-1,1]^2\). The approved training domain, data
   region, coefficient generator, and clipping rule
   \([1-0.95c,1+c]\) must not be changed as incidental refactoring.
3. The CNN bank remains fixed at \(a_0=1\), with the existing screened
   \(\lambda_j\) schedule. A Poisson initialization prior applies only to the
   actual \(\lambda_j=0\) filter; if no such filter exists, initial mixing
   logits are uniform.
4. Mixer inputs remain exactly
   \(a/\bar a,\bar a/a,|r|/\|r\|\), and
   \(|z_j|/\max_k\|z_k\|\). The signed correction remains
   \(\eta_{\max}\eta(1/a)\sum_j\alpha_j z_j\).
5. For a fixed input coefficient field, the model must preserve positive
   RHS-scale equivariance and oddness in the forcing.
6. Supervised training minimizes squared discrete solution \(L^2\) error.
   Physics training minimizes the squared preconditioned residual norm
   \(\|G_0(f-A_hu)\|^2\), using \(a_0=\bar a\). These approved objectives must
   not be replaced by raw residual loss without an explicit experiment and
   decision.
7. `LGFCNN.forward(a, f, u0, K)` keeps its current equations, tensor shapes,
   parameter names, and default behavior so version 8 checkpoints remain
   meaningful.
8. PCG stopping tolerance and the common reporting target remain separate.
   Fixed-budget LGF-CNN/Richardson rows never claim solver convergence.
9. Completed training and evaluation runs remain immutable. Their
   `latest.json` files remain pointers only; retraining creates a new
   `train_id` and never replaces an earlier checkpoint.
10. Evaluation-only CSV/schema changes increment
    `EVALUATION_FORMAT_VERSION`. Model equations or state-layout changes
    require deliberate checkpoint/experiment versioning and retraining.

Before accepting a future change, manually confirm the relevant identities,
load a version 8 checkpoint, compare the unchanged default-forward output on
a fixed seed, and run the manual checks below plus a short end-to-end pipeline.
This documentation makes the intended invariants explicit, but it cannot
detect a regression automatically; executable tests should be added before
future mathematical extensions.

## Checkpoint safety

Checkpoints use format version 8 and are loaded with
`torch.load(..., weights_only=True)`. Earlier checkpoints that use a
different architecture are rejected rather than silently interpreted
under the current equations; retrain with the current code.

The solver-comparison rows retain evaluation schema version 2; the containing
evaluation-run manifest is format version 3 because it adds immutable training
lineage and runtime provenance. This is an output/storage-only change:
checkpoint and mathematical experiment formats remain at version 8, and
existing version 8 models do not need retraining.

Training writes `best.pt`, `last.pt`, `history.csv`, the training curve, and
`completed.json` inside `runs/.attempt-<train_id>/`. The entire completed
directory is then renamed once to `runs/<train_id>/`; files are never promoted
one by one into an existing run. Concurrent training with the same experiment
key is serialised by a file lock. An interrupted run cannot replace or mix
files with an earlier checkpoint.

Evaluation follows the same commit discipline. Files are first generated
under `evaluations/.attempt-<eval_id>/`. Only after all requested data and
figures succeed is `manifest.json` written and the directory atomically
renamed to `evaluations/<eval_id>/`. A failed attempt is removed and never
becomes a visible completed run. Each evaluation hashes the exact checkpoint
byte snapshot loaded into memory and verifies the path again before commit.
If updating `latest.json` fails after a successful directory commit, the run
remains valid and the pointer is repaired from manifests next time. Existing
checkpoint and result directories from the old layout are left untouched.

## Legacy checkpoint and pipeline self-check

The reaction–diffusion extension adds executable tests under `tests/` (see
below). These do not cover the full training/storage pipeline. For future
training-pipeline changes, load a version 8 checkpoint, compare default-forward
output on a fixed seed, and run a 1-epoch pipeline. The current classical
extension checks existing checkpoint inference without running training:

```bash
python -m axy_lgf.train_lgf_cnn \
  --n 16 --J 3 --K 2 --hidden 8 --epochs 1 --eval-every 1 \
  --samples-per-epoch 8 --batch-size 4 --val-samples 4
```

`plot_lgf_cnn.py` is imported when the program starts, so module-level import
errors fail fast. Matplotlib itself is loaded lazily when a new training run
writes its single training curve, or when evaluation figures are requested.

## Main files

| file | does what |
|---|---|
| `grid.py` | square interior grid |
| `operators.py` | NumPy discrete operators and norms |
| `lgf_inverse.py` | Dirichlet LGF |
| `lgf_bank.py` | NumPy screened LGF bank |
| `solvers.py` | PCG, Richardson, LGF bank solver |
| `run_demo.py` | CLI for classical solvers + convergence study |
| `manufactured.py` | manufactured solution problems |
| `torch_ops.py` | PyTorch operators, orthonormal DST-I, LGF bank |
| `lgf_cnn.py` | unrolled LGF-CNN |
| `data.py` | MMS and solver random training data |
| `train_lgf_cnn.py` | training, eval, and fixed u1–u4 prediction |
| `plot_lgf_cnn.py` | training curve, panel, and centerline figures |
| `evaluation.py` | shared metrics, solver comparison, generalization sweeps |

## 2D normal-contrast linear reaction–diffusion baseline

The separate `run_reaction_diffusion` workflow solves

\[
-\nabla\cdot(a\nabla u)+qu=f\quad\text{in }(0,1)^2,
\qquad u|_{\partial\Omega}=0,
\]

with float64 NumPy/SciPy and optional PyAMG. It does not train or alter the
CNN. `run_demo`, the legacy default domain `[-1,1]^2`, existing outputs,
checkpoint versions, and default training equations remain unchanged.
Classical package imports no longer require PyTorch; the two public CNN
classes are imported lazily under their existing names.

### Run commands

From the parent directory containing `axy_lgf/`, in an active Python 3.10+
environment:

```bash
python -m pip install -r axy_lgf/requirements-classical.txt
python -m pip install pyamg                 # optional, required for AMG
python -m pytest -q axy_lgf/tests
python -m axy_lgf.run_reaction_diffusion --n 63
python -m axy_lgf.run_reaction_diffusion --n 63 --forcing discrete
python -m axy_lgf.run_reaction_diffusion --conv
```

`--n` counts interior nodes per axis and `h=1/(n+1)`. `--conv` uses
15, 31, 63, 127 with continuous forcing. Default methods are `lgf gmg amg`;
use `--solvers lgf gmg` to run without PyAMG. A requested but unavailable AMG
produces an explicit `unavailable` row and installation message, with exit
code 2. An unconverged solve also yields exit code 2 and a
`not_converged` row. It is never reported as successful. Invalid input raises
an error. `--tol`, `--maxiter`, `--output-dir`, and `--no-plot` are supported.
Matplotlib is only required when generating figures. PyTorch parity tests
skip explicitly if PyTorch is absent; AMG tests likewise skip if PyAMG is
absent. No replacement method is labelled AMG.

Every invocation writes a **new** child directory under
`axy_lgf/reaction_results/` (or `--output-dir`), with `results.csv`,
`results.json`, and `convergence.png` unless `--no-plot` is used. Existing
`result/` and `checkpoints/` directories are never modified by this workflow.
The CSV contains grid size/spacing, nodal coefficient ranges and contrast,
method/preconditioner, tolerance and zero initial guess, iteration count,
status, final relative algebraic residual, absolute/relative discrete L2
solution error, observed order, setup/solve/total seconds, and diagnostic
messages. JSON additionally records source hashes and runtime versions.

### Coefficient locations and compatibility

The sampled solution, reaction `q`, and forcing live on interior nodes
`(x_i,y_j)=(i*h,j*h)`, `i,j=1,...,n`, with zero solution on boundary nodes.
The new MMS additionally samples the diffusion coefficient at all nodes,
including the boundary, in `a_nodes` of shape `(n+2,n+2)`. Interior values
must agree exactly with the `(n,n)` array `a`.

All edge coefficients are arithmetic nodal averages, for example

\[
a_{i+1/2,j}=\tfrac12(a_{i,j}+a_{i+1,j}),\qquad
(A_hu)_{i,j}=h^{-2}\sum_{k\sim(i,j)}a_{(i,j),k}(u_{i,j}-u_k)
              +q_{i,j}u_{i,j}.
\]

Edges adjacent to the Dirichlet boundary also use both endpoint coefficient
samples. These are midpoints of **node-to-node edges**; the coefficient on
an edge to the boundary is located half a grid spacing inside the domain.
This uniform averaging convention gives second-order consistency for smooth
coefficients. The solver always applies the conservative flux operator; an
expanded differential expression is used only to derive continuous forcing.

The old boundary convention copies the nearest interior `a` to a boundary
edge. It approximates a smooth midpoint coefficient only to first order in
general, so the new continuous-MMS path supplies boundary samples explicitly.
Calling the old APIs without `a_nodes` preserves that convention exactly,
including NumPy/PyTorch training parity. The two conventions represent
separate, documented discrete systems. Every method within a new comparison
uses the same `a_nodes`, `q`, RHS, initial guess, tolerance and operator.

`apply_reaction_diffusion(u,a,grid,q=0,a_nodes=None)` reuses
`apply_div_a_grad`. `assemble_reaction_diffusion` builds the same matrix in
C-order flattening from the same face coefficients. The added reaction has
positive sign. For any nonzero interior vector `v`, with boundary entries
set to zero,

\[
v^T A_hv=h^{-2}\sum_{\text{edges }e}a_e(v_{e,+}-v_{e,-})^2
           +\sum_{i,j}q_{i,j}v_{i,j}^2>0.
\]

Each interior edge is counted once; boundary edges connect to prescribed
zero. Positive edge coefficients, nonnegative reaction, and the connected
Dirichlet grid give a symmetric positive definite matrix, including `q=0`.

### Manufactured solution and forcing

\[
u_*=\sin(\pi x)\sin(\pi y),\quad
 a=1+0.2\sin(2\pi x)\sin(2\pi y),\quad
 q=1+0.2\cos(2\pi x)\cos(2\pi y).
\]

Both coefficients lie in `[0.8,1.2]`; the theoretical diffusion contrast is
at most 1.5. Reported extrema and contrast use the actual interior samples.
Since `Delta(u_*)=-2*pi^2*u_*`,

\[
f=(2\pi^2a+q)u_*
 -0.4\pi^2\left[
 \cos(2\pi x)\sin(2\pi y)\cos(\pi x)\sin(\pi y)
 +\sin(2\pi x)\cos(2\pi y)\sin(\pi x)\cos(\pi y)\right].
\]

* **Continuous MMS** (default) samples this analytic expression, independently
  of `A_h`. Its solution error measures discretization plus algebraic error.
  The refinement study reports
  `p=log(E_previous/E_current)/log(h_previous/h_current)`.
* **Discrete MMS** (`--forcing discrete`) instead sets `f=A_h*u_*` to test
  algebraic solver consistency. Its solution error should reach the solver
  tolerance; it provides no evidence of spatial convergence. No observed
  spatial order is assigned to these rows. `--conv --forcing discrete` is
  rejected to keep the convergence study unambiguous.

The norm is `||v||_L2h=sqrt(h^2*sum(v^2))`. Relative L2 solution error divides
`||u-u_*||_L2h` by `||u_*||_L2h`. The relative algebraic residual is
`||f-A_h*u||_L2h/||f||_L2h`, identical to the Euclidean norm ratio on one
uniform grid. It is distinct from solution error and from a continuous PDE
residual; the latter is not measured here. For zero RHS, the solver uses
`max(||f||_L2h, float64.tiny)` as denominator.

### Preconditioners and stopping rule

All three baseline methods use the same PCG loop and stop at relative
algebraic residual `<= --tol`, default `1e-10`, starting from zero with
at most 500 iterations. A recursively updated residual is used during CG;
a recomputed `f-A_h*u` must pass before success is returned. If recurrence
drift causes a false stop, CG replaces the residual and restarts. The final
residual is always recomputed with the matrix-free target operator.

* **LGF-PCG:** the unchanged DST-I implementation exactly inverts the reference
  `P_h=a0*(-Delta_h)+q0*I`, with `a0=mean(a)` and `q0=mean(q)`. Its sine-basis
  denominator is `a0*(lambda_p+lambda_q)+q0`, where
  `lambda_p=4/h^2*sin^2(p*pi/(2*(n+1)))`. This is a constant-coefficient
  approximation to the variable-coefficient target; it is not the exact
  target inverse. At `q=0`, it reduces to the original Poisson LGF.
* **GMG-PCG:** a genuine geometric hierarchy `n -> (n-1)/2 -> ... -> 3`
  uses bilinear geometric interpolation `P`, restriction `R=P.T`, and
  Galerkin operators `A_c=P.T*A*P`. Coarse matrices can have nine-point
  stencils. This transpose restriction uses variational scaling; the
  usual factor `1/4` of full weighting is absorbed by the coarse system.
  Two damped-Jacobi sweeps precede and follow every coarse correction.
  On each level, `omega=1/max_i(sum_j(abs(A_ij))/A_ii)` bounds the scaled
  maximum eigenvalue by one. Symmetric smoothing, transpose transfers,
  and an exact Cholesky solve on the 3-by-3 interior grid make a fixed,
  linear SPD-compatible V-cycle. GMG accepts `n=2^k-1`, `k>=2`.
* **AMG-PCG:** optional PyAMG Ruge–Stuben with Galerkin coarse systems,
  transpose restriction, forward pre-Gauss–Seidel and backward
  post-Gauss–Seidel, one V-cycle, and Cholesky on at most nine coarse
  unknowns. These adjoint sweeps preserve symmetry and positivity for
  the SPD target. Small-grid tests explicitly construct both V-cycle
  preconditioner matrices and check linearity, symmetry, and eigenvalues.

Setup time includes validation, fine-grid sparse assembly where needed,
preconditioner construction, and coarse factorization. AMG's first call
includes importing PyAMG. Solve time includes PCG and residual checks;
`total=setup+solve`. MMS generation, final error reporting, and plotting are
excluded. These are single-run measurements without warmup or repetitions;
iteration counts alone do not establish a timing advantage.

### Verification scope and files

`tests/test_reaction_diffusion.py` verifies the frozen legacy stencil,
reaction sign and linearity, all boundary-face averages, sparse/matrix-free
agreement, symmetry/SPD and the edge energy identity, reference LGF inverses,
analytic forcing values, geometric transfers, both V-cycle preconditioners,
discrete recovery and continuous second-order convergence on all four grids,
initial guesses/zero RHS, failure status, invalid coefficients, an independent
sparse direct solution, optional-AMG messaging, and immutable result output.
It also tests second-order accuracy using a polynomial MMS with nonconstant
boundary coefficients. `tests/test_torch_compatibility.py` checks the legacy
NumPy/PyTorch diffusion operator and filter-bank parity, without training.

New implementation files are `reaction_manufactured.py` (analytic/discrete
MMS), `multigrid.py` (geometric V-cycle and optional AMG), and
`run_reaction_diffusion.py` (CLI and isolated results). `operators.py` adds
shared face construction, reaction action and sparse assembly; `solvers.py`
extends existing solvers with keyword-only `q` and `a_nodes`, retains all old
positional arguments, and adds common PCG verification and timing. Richardson
uses screening `mean(q)`; the bank shifts its old screening schedule by
`mean(q)`. `grid.py`, `manufactured.py`, `lgf_inverse.py`, `lgf_bank.py`,
`run_demo.py`, and all neural-network/training source files remain unchanged.
See `VERIFICATION.md` for actual measured outcomes and environment details.
