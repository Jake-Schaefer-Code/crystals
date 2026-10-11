# crystals

Research sandbox for symmetry, geometry and many-body physics. It holds six largely independent areas (listed under Layout) as importable modules, plus notebooks and example scripts. It is not an installable package: there is no `pyproject.toml`, and everything runs from the repository root.

`AGENTS.md` has conventions and working rules. `DEVLOG.md` records what changed and why, newest first. If a path or command below stops working, fix this file.

## Setup and running things

On a fresh Mac, install Apple's Command Line Tools (`xcode-select --install`), then:

```bash
cd /Users/js5947/Desktop/Research/crystals  # use your clone's location
make setup
make test
make example
make notebook
```

`make setup` bootstraps a pinned uv tool into `.tools/`, downloads Python 3.13,
creates `.venv/`, and installs the exact versions in `requirements.txt`.
It needs internet on the first run and does not change the system Python.
JAX uses the CPU on macOS. There is no compiled application to build.
Run `make help` for the available commands.

For your own scripts, activate the environment from the repository root:

```bash
source .venv/bin/activate
python examples/calogero_vmc.py
python -m pytest tests/test_young.py -q
```

In VS Code, select `.venv/bin/python` as the Python interpreter and notebook
kernel. `make notebook` launches JupyterLab using this same environment.
The test and example targets use headless plotting; interactive notebooks do not.

`requirements.in` lists direct dependencies; `requirements.txt` locks their
resolved versions. After intentionally changing dependencies, run `make lock`
and `make setup`, then check `make test` before committing the new lock file.
`make setup` synchronizes the environment to the lock file, removing extra packages.

Specialized quantum and geospatial notebooks need additional packages:
run `make notebooks` after setup (or again after a later `make setup`). These
optional packages are listed in `requirements-notebooks.txt` and are not locked.
Some exploratory notebooks also reference FEniCS (`dolfin`) or local modules
(`PDE`, `geometry`) that are not shipped here; those need their original external
code/environment. Notebook outputs are tracked, so avoid rerunning whole notebooks
just to check installation.

There is no poetry, Hydra or `jaxqmc` setup here. In a sandbox where
`~/.matplotlib` or the bytecode cache is read-only, point `MPLCONFIGDIR` and
`PYTHONPYCACHEPREFIX` at a temp directory. `python -I` ignores every `PYTHON*`
variable, so pair it with `-B` to prevent bytecode beside the sources.

## Layout

Dependencies run one way: `plotting` imports `physics`, `physics` imports `src`. `physics` returns numbers and result objects and never imports `plotting`; `plotting` functions take a result and return a `Figure`.

| Area | Modules | Needs |
|---|---|---|
| Crystal and lattice geometry | `src/{coordinates, planar_geometry, geo_ops_utils, crystal_funcs, lattice_geometry, categorization, bases}.py`, `TriMap.py`, `plotting_utils.py`, `unit_cell_utils.py` | numpy, scipy, matplotlib |
| Finite symmetry and representation theory | `src/symmetry/` (`symmetry`, `permutations`, `operations`, `named_groups`, `group_laws`, `group_algebra`, `young`) | numpy |
| Fermion antisymmetry and nodes | `src/{invariants, grassmann, exact_gb, exact_poly}.py`, `physics/{nodal, nodal_models, ansatz1d, models1d, vmc}.py`, `plotting/nodal.py` | jax (`exact_*` are pure Python) |
| Stochastic thermodynamics | `physics/{markov, glauber, ising_sectors, speed_limits, sk_theory, dfa_memory, simplex}.py`, `plotting/{heatmaps, ising_pop, simplex, style}.py` | jax, optax, scipy; diffrax in the ODE helpers |
| Random matrices | `physics/spectral_curves.py`, `plotting/spectral_curves.py` | numpy, scipy, sympy |
| Wave and qubit demos | `physics/{bloch, dbl, density_evolution, hartree_fock, utils}.py` | numpy, scipy, matplotlib |
| Shared | `src/core/` (type aliases ported from CEM), `src/config/` (ported from CEM; only `notebooks/physics/shift_operator.ipynb` uses it), `src/quantum_symmetry.py` (Lindblad and commutant helpers for the quantum notebooks), `src/permutations.py` (alias for `src/symmetry/permutations.py`) | |

## Tests

`tests/` has 345 tests: finite symmetry (`young`, `group_algebra`, `exact_*`), antisymmetry and nodes, stochastic thermodynamics, spectral curves, and plotting smoke tests. The geometry layer and the wave and qubit demos have none.

## Examples

Scripts, not importable modules. `crystal_nodes.py` and `proton_slab.py` write to `examples/out/`; git ignores `*.png` and `*.gif`.

| Script | What it does |
|---|---|
| `calogero_vmc.py` | Determinant-free VMC on Calogero-Sutherland, whose exact ground state lies in the ansatz (`--n`, `--lam`, `--steps`) |
| `crystal_nodes.py` | How a triangular lattice of wells moves the nodal surface of spin-polarized electrons (`--quick`, about a minute) |
| `proton_slab.py` | A slab of protons as a programmable spin Hamiltonian, at toy scale (about a minute) |
| `ising_pop_physics.py` | Periodic-optimal-prior mismatch analyses for the Curie-Weiss model (`--quick --output-dir DIR`) |
| `comps_plots.py`, `figures.py`, `lattice_plots.py`, `per_sym.py` | Presentation figures and animations for free particles, bands, lattices and Bloch waves |

## Notebooks

`notebooks/{geometry, physics, quantum, rep, rmt, stoch_thermo}`. Their outputs are tracked, so avoid re-running whole notebooks casually. The `stoch_thermo` and `rmt` notebooks import the modules above instead of carrying copies of them.

## Other projects

`agca_jax` is a separate project and not a dependency. `src/exact_gb.py` is vendored from it, with provenance in its docstring. Keep checkouts of other projects out of this tree: a nested copy once made a bare `pytest` abort at collection, because both projects ship a top-level `src`.

## Known issues (checked 2026-10-08)

- `TriMap.create_mapping()` raises `FrozenInstanceError`: it assigns to the frozen `TriMapConfig`. `map_polygon_to_triangle` reads `self.kind`, which is never set. `define_distribution(..., distribution=None)` raises `UnboundLocalError` (`dist_wts` is assigned but `dst_wts` is returned). The branch with a given `distribution` works.
- `physics/dbl.py` runs a simulation when imported, `physics/bloch.py` opens a figure when imported, and `physics/hartree_fock.py` prints when imported. Every other module under `src/`, `physics/` and `plotting/` imports cleanly.
- `examples/lattice_plots.py` has no `if __name__ == "__main__"` guard.
- The geometry layer has no tests.
