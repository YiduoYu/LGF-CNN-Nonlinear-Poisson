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

Until executable tests are added, this section is the manual regression
contract. Future changes must preserve all items below unless the mathematical
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

## Manual self-check while tests are deferred

This source folder intentionally does not add a `tests/` directory yet.
Therefore the mathematical compatibility contract above is a review
checklist, not an executable guarantee. Do not claim automated coverage until
test source files are actually added. After making changes, load a version 8
checkpoint, compare default-forward output on a fixed seed, and run a 1-epoch
pipeline:

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
