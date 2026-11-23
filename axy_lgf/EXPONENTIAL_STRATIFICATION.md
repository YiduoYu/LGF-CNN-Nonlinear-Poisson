# Exponential-stratification coefficient

The new selectable coefficient is

\[
a(x,y)=a_0\chi^x,\qquad a_0>0,\quad\chi\geq1,\quad(x,y)\in(0,1)^2.
\]

The code names the physical scale `a_scale` (`--a-scale`, default 1). It is the
coefficient at x=0. The LGF reference diffusivity is still `mean(a)`, as before;
it is not the same parameter as the image's a0. The contrast `chi` has no
implicit default and must be supplied explicitly for the new family.

The old normal-contrast case remains the default for backward compatibility.
Use `--coefficient exponential` to select the changed coefficient in either
classical workflow. Existing solvers, the conservative flux stencil, boundary
conditions, q and the exact solution are unchanged.

## Continuous and discrete forcing

The exact solution and reaction remain

\[
u=\sin(\pi x)\sin(\pi y),\qquad
q=1+0.2\cos(2\pi x)\cos(2\pi y).
\]

For exponential stratification,

\[
a_x=\log(\chi)a,\quad a_y=0,\qquad
f=(2\pi^2a+q)u-\pi\log(\chi)a\cos(\pi x)\sin(\pi y).
\]

Continuous forcing uses this analytic formula, independently of the discrete
operator. The optional discrete MMS still uses `f_h=A_h*u_exact`.

## Coefficient locations and contrast

The coefficient is sampled on all grid nodes, including x=0 and x=1.
All face values retain the existing arithmetic nodal average, including edges
adjacent to the homogeneous Dirichlet boundary. No solver stencil was changed.
The field is constant in y and increases smoothly with x for chi>1.

On the closed domain, a ranges from a_scale to a_scale*chi, so the ratio is
exactly chi. On an n-by-n interior grid with h=1/(n+1), sampled a ranges from
`a_scale*chi**h` to `a_scale*chi**(1-h)`. Its ratio is `chi**(1-2*h)`, smaller
than chi when chi>1. The usual `a_min`, `a_max`, `a_contrast` columns retain
their interior-node meaning. New exponential records also save `coefficient`,
`a_scale`, `chi`, and boundary-inclusive `a_nodes_min`, `a_nodes_max`,
`a_nodes_contrast` in the MMS/summary metadata. The grid-scaling CSV saves both
interior and boundary-inclusive contrasts.

## Commands

From the package parent, with the existing Python verification environment:

```bash
source /tmp/axy-grid-scaling-venv/bin/activate
export MPLCONFIGDIR=/tmp/axy-grid-scaling-mpl
export XDG_CACHE_HOME=/tmp/axy-grid-scaling-cache
```

For example, chi=2 (the moderate-contrast value used in verification):

```bash
python -m axy_lgf.run_reaction_diffusion --coefficient exponential \
  --a-scale 1 --chi 2 --n 63
```

A later four-solver scaling experiment can select the same coefficient:

```bash
python -m axy_lgf.run_grid_scaling --coefficient exponential \
  --a-scale 1 --chi 2 --include-boomer \
  --output-dir axy_lgf/reaction_results/grid_scaling_exponential_chi2
```

These are reproducible command examples, not records of a newly executed
scaling experiment. Choose the intended a_scale and chi before starting that
experiment. All solver/preconditioner settings and the existing warmup/repeat
protocol remain unchanged. New single-MMS outputs default to
`reaction_results/exponential/`; new scaling outputs use
`reaction_results/grid_scaling_exponential/` or
`reaction_results/grid_scaling_exponential_boomer/`. Previous outputs are
preserved, and scaling refuses an existing output directory.

For direct numerical use:

```python
from axy_lgf.reaction_manufactured import make_reaction_problem
p = make_reaction_problem(63, coefficient="exponential", a_scale=1.0, chi=2.0)
```

## Verification

All 87 tests passed (61 existing + 26 new), with zero failures and zero skips.
The added tests check analytical values and forcing, chi=1 constant diffusion,
positivity/parameter validation, y independence, exact boundary-inclusive and
interior sampled contrasts, boundary-face averaging, separate continuous and
discrete forcing, matrix consistency/SPD, four-solver discrete recovery,
second-order continuous MMS at chi=2, CLI parameter recording, and selection in
the scaling driver (mocked solver calls for that routing test).

No high-contrast scan or new parameter-specific performance experiment was
performed. The new-family solver unit tests use chi=2; chi=1 is also checked
analytically. Full benchmark parameters remain for the user to choose.

Only reaction_manufactured.py, run_reaction_diffusion.py and run_grid_scaling.py
were modified. The new test file is tests/test_exponential_stratification.py.
All existing solver/native-preconditioner code, checkpoint files and historical
results remain unchanged.

Verification record:

[exponential_verification_20261026T081615Z_6a17b78c](reaction_results/exponential_verification_20261026T081615Z_6a17b78c/verification.json)
