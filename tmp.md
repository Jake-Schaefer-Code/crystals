# SME Propagation Tests

Gravitational-wave (GW) propagation and Lorentz-invariance violation (LIV) inference using GW event posterior samples. LIV parameters are represented using spherical harmonics and inferred using KDE, and SVD. The inference repository is JAX-based and also contains sampling kernels and learned-flow research code for adaptive local sampling and global diffeomorphism techniques.

```text
GW event posterior files
  -> parsing, angle and propagation-parameter conversion
  -> spherical-harmonic representation
  -> KDE / SVD inference
  -> diagnostics and plots
```

This is a JAX-based research library undergoing preparations for collaborator handoff. `src/` is the only active library tree. 
Synthetic parser-to-SVD checks and a cached real-event dispersion workflow are executable. 
An experimental JAX flow-adaptive joint-inference path is also executable; its scientific coverage and production convergence still need validation.

## Installation

Package metadata requires Python 3.10 or newer. To install, proceed from the repository root, using your chosen environment's interpreter:

```bash
python -m pip install -e .
```

`setup.py` declares JAX and the active numerical stack as required dependencies;
`pyproject.toml` configures setuptools and Ruff. `build/requirements*.txt` and
`build/postjax.txt` contain additional CPU/GPU dependency snapshots. They have not yet been reconciled into a verified clean-install recipe.

The current maintainer uses `/home/jacob.schaefer/SogGitlab/cem_env/bin/python` on the remote host or `/Users/jakeschaefer/Desktop/Research_Stuff/base_env/bin/python` locally.
These are development environments, not paths collaborators must reproduce. 
In the future for collaborators, the code should be run in apptainer containers. 

## First Run: Synthetic Posterior Files To SVD

After installation, as a first example, run this from the repository root:

```bash
python -m examples.parser_svd --nsamples 32 --seed 42
```

The example creates temporary HDF5 posterior files, parses their sky detection angles and propagation parameters, and solves a single-coefficient system. 
It serves to demonstrate that the code reproduces coefficients correctly; it asserts the expected symmetric posterior magnitude `|k00| = 2`. Afterwards, it removes its temporary files. 
It needs no private data, network access, or plotting backend. This is a numerical sanity check, not a real-event scientific result. 

## Reusable Dispersion Data

The main study of this repository is the mass-dimension-6 study, concerning 31 selected events and 25 coefficients. We wish to infer the 25 constituent coefficients making up
$$
A_4(\hat{\mathbf{n}}) = -2 \sum_{jm}Y_{jm}(\hat{\mathbf{n}})k_{(I)jm}^{(6)},
$$
where $A_4$ is a Lorentz-violating parameter containing the information of GW dispersion.
For the mass-dimension-6 study, first convert the original positive/negative event files once. Then, run either least-squares or chi-square SVD from the saved data:

```bash
python -m examples.dispersion_svd --data-dir /path/to/LIV --convert-only
python -m examples.dispersion_svd --nsamples 10000 --seed 42 \
  --output-dir results/dispersion_svd
```

This defaults to the 31 selected events and 25 coefficients. The second command
needs only `data/converted_dispersion/`; it performs no cosmological integrations. 
It saves coefficient draws, percentile tables, plots, and a reproducibility report.
Evidence and the log-lambda-to-flat-amplitude Jacobian are normalized jointly across both signs without discarding samples. Chi-square uses the resulting weighted event uncertainties. Outputs are SVD random-draw estimates, not Bayesian credible intervals.

[The dispersion workflow guide](docs/dispersion_workflow.md) explains the paper
equations, the historical `A4 = phenomenological A_4 / 2` convention, cache reuse,
replotting, diagnostics, and scientific limitations.

For experimental joint Bayesian inference from those caches:

```bash
python -m examples.adaptive_flow_inference \
  --cache-dir data/converted_dispersion \
  --output-dir results/adaptive_flow_2026-09-28
```

This uses SVD only for initialization, adapts a continuous normalizing flow during
warmup, freezes adaptation, and returns a Metropolis-corrected production chain.
The [physics audit](docs/physics_audit.md) states what has been verified and what
must still be established before reporting physical credible intervals.

## Direct Parser API

For local event files, use `GWEventParser(data_dir=...)`. Each event directory should
contain `aplus_alpha4.h5` and `aminus_alpha4.h5`; `draw_events` accepts event-directory
names or existing registry tuples. A minimal call sequence is:

```python
from src.gw_liv.GWEventParser import GWEventParser
from src.probabilistic.bayesian.GWSVD import SVD

events = GWEventParser(data_dir="/path/to/events").draw_events(["GW190503bf"])
coefficients = SVD("least", events, ndim=1).solve(nsamples=100)
```

Choose the harmonic dimension and weighting for your study. Chi-square mode scales
both the harmonic rows and amplitudes by positive standard deviations; a scalar or
per-event/per-sample values can be passed as `input_stdev`. SVD truncates singular
values below `eps * max(matrix.shape)` times the largest singular value and returns
the minimum-norm solution. It does not establish that every coefficient is identifiable.
Explicit `weights` replace the default absolute-amplitude Jacobian weights; chi
uses their weighted event standard deviations when `input_stdev` is omitted.
Each solver defaults to its own seed-42 generator; pass a NumPy `Generator` as `rng`
to control posterior draws explicitly.

## Where To Start Reading

| Task | Current implementation |
| --- | --- |
| Locate event data | [Event registry](src/data/gw_events.py) |
| Parse posterior samples | [GWEventParser](src/gw_liv/GWEventParser.py) |
| Inspect cosmology and conversions | [Astrophysics helpers](src/core/astro.py) |
| Build the harmonic representation | [GW helpers](src/gw_liv/utils.py) |
| Run classical inference | [SVD](src/probabilistic/bayesian/GWSVD.py), [KDE](src/probabilistic/bayesian/GWKDE.py) |
| Run flow-adaptive inference | [Joint target](src/gw_liv/coeff_inference.py), [flow workflow](src/gw_liv/flow_inference.py), [driver](examples/adaptive_flow_inference.py) |
| Inspect learned models and training | [JAX ML](src/probabilistic/ml/) |
| Inspect sampling and orchestration | [MCMC](src/probabilistic/mcmc/), [FSM](src/probabilistic/fsm/) |
| Plot results | [Visualization helpers](src/visuals/visuals.py) |

`experiments/` and `notebooks/` contain study drivers and analysis, with varying levels
of migration. [examples/parser_svd.py](examples/parser_svd.py) is the small portable
example. Inspect imports and inputs before treating other scripts as supported entry points.

[src/run.py](src/run.py) exposes Tyro options including `--data-dir`, `--nsamples`,
and `--no-plot`. Its SVD path is covered with synthetic inputs, and
`python -m src.run --help` works without loading numerical or plotting modules. The CLI still selects
the registry's default event list. KDE and plotting need separate validation; package
metadata installs no console script.

## Data And Scientific Conventions

Parser outputs contain `theta`, `phi`, and `A4`. Angles are in radians: `theta` is
`pi / 2 - dec`, and `phi` is right ascension. The optional `unit_convert=True` parser
argument applies `ev2_to_m2` to `A4`; preserve that choice through downstream inference.
Despite its historical name, `A4` holds half the paper's phenomenological A_4.
Check signs, harmonic coefficient order, units, and event/sample axes when comparing
implementations.

Event files are external inputs. The registries still include personal and cluster
paths, and `src/data/gw_events.py` overlaps with `src/data/gw_events_new.py`. An explicit
parser `data_dir` avoids those roots; omitting it preserves the registry default.
Record the input-data identity, configuration, random seed, dependency versions, code
revision, and output location with each scientific result.

## Verification

Run from the repository root with the same interpreter used for installation:

```bash
python -m compileall -q src experiments examples tests
python -c "import importlib; importlib.import_module('src')"
```

These check syntax and the minimal package root only. Install pytest into the selected
environment and run the numerical suite:

```bash
python -m pip install pytest
JAX_PLATFORMS=cpu python -m pytest -q tests
```

[tests/README.md](tests/README.md) describes coverage and remaining work. The tests use
synthetic data, analytic harmonics, known coefficients, and weighted least-squares
references. Optional JAX execution is currently part of this suite.

The harmonic argument-order and chi-square weighting fixes change results relative
to earlier revisions. Re-run affected analyses; historical agreement alone is not a
validation of the corrected scientific workflow.
The harmonic call follows [SciPy's degree/order and polar/azimuth conventions](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.sph_harm_y.html).

## Remaining Handoff Work

- Extend the synthetic checks to independently validated real-event references.
- Repair remaining workflow/API mismatches and stale imports, including
  `src/probabilistic/bayesian/bandwidths.py` and remaining experiment scripts.
- Reconcile event registries, filesystem paths, and CEM-derived configuration names.
- Calibrate the joint likelihood and adaptive-flow sampler with mock coverage,
  convergence diagnostics, and independent reduced-dimensional references.
- Reconcile installation constraints and test an installed package outside the checkout.

## Contributing

[AGENTS.md](AGENTS.md) is the canonical maintainer guidance. It adapts CEM's
`paper-release` practices to SME: focused changes, shared-tree safety, typed Tyro CLIs,
explicit host/JAX boundaries, reproducible results, and numerical regression checks.
[SKILLS.md](SKILLS.md) describes recurring workflows; [DEVLOG.md](DEVLOG.md) records
historical changes. [CLAUDE.md](CLAUDE.md) delegates to the same instructions.

For GitLab SSH authentication failures, see the
[SSH push troubleshooting guide](docs/gitlab_ssh_troubleshooting.md), including
the September 2026 authentication failure and verified recovery steps.




## Submitting to condor:
- https://computing.docs.ligo.org/guide/htcondor/credentials/
- https://computing.docs.ligo.org/guide/htcondor/access/
- https://computing.docs.ligo.org/guide/computing-centres/ldg/
- https://ldas-gridmon.ligo.caltech.edu/ldg_accounting/user


