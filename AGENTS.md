# AGENTS.md

This file provides guidance to coding agents working in this directory.

This repository implements representation-theoretic crystallography utilities.
Prefer mathematically faithful abstractions over plotting shortcuts.

This is a research sandbox for representation theory, geometry, crystal/orbifold mappings, symmetry  
operations, lattice visualizations, and Poisson-ratio experiments. It is not an  
installable Python package right now. Treat it as a folder of importable helper  
modules plus exploratory notebooks and scripts.

## Development Commands

For real import and plotting checks in this sandbox, use the local scientific
virtual environment:

```bash
PY=/Users/jakeschaefer/Desktop/Research_Stuff/base_env/bin/python
export PYTHONPYCACHEPREFIX=/tmp/crystals_pycache
export MPLCONFIGDIR=/tmp/crystals_mplconfig
export MPLBACKEND=Agg
```

Keep generated plot smoke-test artifacts in `/tmp`, not in the project tree.


### Syntax Checks

Prefer the local scientific virtual environment for real checks:

```bash
PY=/Users/jakeschaefer/Desktop/Research_Stuff/base_env/bin/python
export PYTHONPYCACHEPREFIX=/tmp/crystals_pycache
export MPLCONFIGDIR=/tmp/crystals_mplconfig
export MPLBACKEND=Agg
```

`MPLCONFIGDIR` matters because `~/.matplotlib` may not be writable in this
sandbox. `PYTHONPYCACHEPREFIX` keeps bytecode caches out of user cache
directories that may also be outside writable roots.

```bash
$PY -m py_compile \
  coordinates.py geo_ops_utils.py crystal_funcs.py TriMap.py categorization.py \
  plotting_utils.py unit_cell_utils.py figures.py functions6.py legacy/old.py physics/poisson.py unit_cell.py \
  per_sym.py comps_plots.py lattice_plots.py \
  geometry/voronoi.py
```

### Import Smoke Tests

Use the local venv above for import tests. The system `python3` may not have
`numpy`, `scipy`, or `matplotlib`.

```bash
$PY - <<'PY'
import coordinates
import geo_ops_utils
import crystal_funcs
from TriMap import TriMap
import categorization
import per_sym
import comps_plots
print("core imports ok")
PY
```

Do not import every `*.py` file as a validation step. Several scripts execute
plotting, animation, or FEniCS solves at import time.

### Plot Smoke Tests

For plotting changes, keep generated smoke-test artifacts in `/tmp`:
Figure constructors in `comps_plots.py` return figures; save them externally
with `plotting_utils.save_path()` instead of passing `savepath` into the
constructor.

```bash
$PY - <<'PY'
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import comps_plots
from plotting_utils import save_path

out = Path("/tmp/crystals_plot_smoke")
if out.exists():
    shutil.rmtree(out)
out.mkdir(parents=True)

paths = []
fig = comps_plots.plot_free_particle_paraboloid()
paths.append(save_path(fig, out / "free_particle_paraboloid.png"))
plt.close(fig)
fig = comps_plots.make_free_vs_central_potential_figure(ngrid=50)
paths.append(save_path(fig, out / "free_vs_central.png"))
plt.close(fig)
paths.append(comps_plots.make_discrete_laplacian_figure(out))
paths.append(comps_plots.make_monatomic_dispersion_figure(out, n_sites=8))
paths.append(comps_plots.make_diatomic_dispersion_figure(out))
paths.append(comps_plots.make_shift_operator_animation(out, n_frames=3, filename="shift_operator_smoke.gif"))
paths.append(comps_plots.make_gap_opening_animation(out, n_frames=3, filename="gap_opening_smoke.gif"))

missing = [str(p) for p in paths if not Path(p).exists() or Path(p).stat().st_size == 0]
if missing:
    raise SystemExit(f"missing/empty outputs: {missing}")
print("plot smoke ok")
PY
```

### Tests

There is currently no formal test suite, `requirements.txt`, `pyproject.toml`,
or package metadata. Use focused smoke tests around the function you changed.
For geometry changes, prefer small deterministic arrays with known areas,
barycentric coordinates, or affine transforms.

## Project Architecture

### Core Library Modules

- `coordinates.py` - Barycentric/cartesian conversion, stereographic
projection, and spherical/cartesian coordinate conversion.
- `geo_ops_utils.py` - Quaternion helpers, rotation matrices, 2D rotation
matrices, point rotation, and polyhedron rotation.
- `planar_geometry.py` - Planar polygon helpers: edge sampling, polygon area
  and centroid, intersection and containment predicates, and triangle-index
  helpers used by `TriMap` and plotting code.
- `crystal_funcs.py` - Higher-dimensional geometry, interpolation, affine and
  polynomial transforms, geodesic helpers, weighted distributions, and
  piecewise matrix collection. It also re-exports older planar helpers for
  notebook compatibility.
- `TriMap.py` - `TriMap` class for mapping polygons to triangular fundamental
domains and selecting low-energy piecewise affine maps.
- `categorization.py` - Limited parser for selected space-group-like symmetry
symbols into `(matrix, translation)` operations.
- `plotting_utils.py` - Matplotlib helpers for visualizing simplices,
  distributions, symmetrized functions, and mapping results.
- `unit_cell_utils.py` - Shared geometry and Matplotlib helpers for unit-cell
  sketches: equilateral/rhombic vertices, point rotation/reflection, polygon
  drawing, triangular grid generation, spiral coordinates, and legend cleanup.
- `per_sym.py` - Periodic symmetry helpers for lattice construction,
  reciprocal lattices, Bloch waves, shift operators, Brillouin-zone sketches,
  and simple monatomic/diatomic dispersions.
- `figures.py` - Mostly import-safe CLI/script for lattice animation and
  unit-cell figures. Requires `tyro` for command-line usage.
- `comps_plots.py` - Import-safe presentation figure and animation generators
  for free-particle, central-potential, shift-operator, and band-structure
  comparisons.

### Exploratory Or Script-Like Files

- `unit_cell.py` - Import-safe triangular unit-cell animation script. Running
  `main()` creates a `FuncAnimation`, writes `spiral_animation2.gif`, and calls
  `plt.show()`.
- `functions6.py` - Import-safe legacy triangular unit-cell sketch. Running
  `main()` creates and shows a single Matplotlib figure.
- `physics/poisson.py` - Not import-safe. Imports `dolfin`, solves a FEniCS linear
elasticity problem, forces the `MacOSX` backend, prints results, and calls
`plt.show()`.
- `legacy/old.py` - Legacy plotting/mapping helper file. It imports broadly and
duplicates routines that now exist in current modules. Keep it for reference
only; do not add new code there.
- `*.ipynb` - Research notebooks. Avoid rewriting large outputs unless the user
explicitly asks for notebook cleanup.
- `geometry/` - Separate exploratory geometry area. `geometry/flows.md` is
explanatory text. `geometry/voronoi.py` is currently empty apart from
whitespace.

## Dependencies

Common dependencies:

- `numpy`
- `scipy`
- `matplotlib`

Optional or context-specific dependencies:

- `scikit-learn` for smoother Gaussian-mixture sampling in
  `plotting_utils.plot_symmetry_op` and for importing `legacy/old.py`
- `tyro` for `figures.py`
- `dolfin`/FEniCS for `physics/poisson.py` and parts of `poisson_ratio.ipynb`
- `ffmpeg` for `unit_cell.py` GIF generation
- `sympy` in `root_plots.ipynb`

## Important Conventions

### Array Shape Conventions

Most code assumes NumPy arrays where points are rows and coordinate components
are in the final axis:

```python
points_2d.shape == (n_points, 2)
points_3d.shape == (n_points, 3)
triangle.shape == (3, 2)
tetrahedron.shape == (4, 3)
```

Batch dimensions usually come before the simplex/point axes. Check existing
call sites before changing broadcasting behavior.

### Polygon Ordering

2D polygon routines generally assume vertices are ordered counterclockwise.
This affects:

- `area`
- triangulation and inclusion checks
- interpolation along polygon edges
- self-intersection tests
- mapping quality calculations

If a result looks sign-flipped or self-intersecting, check vertex order before
rewriting the math.

### Barycentric Coordinates

`coordinates.py` is the preferred home for coordinate conversion logic.
`legacy/old.py` contains older duplicate barycentric functions; do not copy
those into new code unless the task is explicitly about reproducing legacy
behavior.

Use:

- `barycentric_coordinates2D`
- `barycentric_to_cartesian_2D`
- `cart_to_bary3D`
- `bary_to_cart`
- `calculate_barycentric_coordinates`

`calculate_barycentric_coordinates` solves by least squares and prints residuals
if present.

### Rotations And Symmetry Operations

Quaternion vectors are represented as `[w, x, y, z]`.

Primary rotation helpers live in `geo_ops_utils.py`:

- `rotation_quaternions`
- `make_q_rot_mats`
- `q_rot_mat`
- `rotate`
- `generate_rotation_matrices`
- `rotate_polyhedron`
- `rot_mat`

Symmetry operations are generally represented as `(matrix, translation)` pairs.
The parser in `categorization.py` supports only a small subset of notation.
Extend it incrementally and document examples when adding new symbols.

### Mapping Workflow

Typical `TriMap` usage:

```python
from TriMap import TriMap

mapper = TriMap(polygon)
mapper.define_distribution()
maps, matrices, triangulations, rotated_polygons = mapper.create_mapping()
idx = mapper.minimize_jacobian()
```

Important behavior:

- `TriMap` recenters both the polygon and triangle around their centroids.
- `create_mapping()` rotates through 12 polygon orientations.
- Candidate maps are built by assigning unique polygon vertex triples to
triangle vertices, then interpolating the remaining polygon vertices.
- Delaunay triangulations are created on destination points.
- `minimize_jacobian()` chooses the lowest Dirichlet-style energy candidate.

### Plotting Workflow

Use plotting helpers from notebooks or explicit scripts. Many plotting helpers
call `plt.show()`.

Do not add top-level plotting side effects to import-safe modules. If a script
needs to show a plot or write a file, put it behind:

```python
if __name__ == "__main__":
    ...
```

## Error-Prone Areas

### Import Safety Is Critical

Safe to import:

- `coordinates.py`
- `geo_ops_utils.py`
- `crystal_funcs.py`
- `TriMap.py`
- `categorization.py`
- `plotting_utils.py`
- `unit_cell_utils.py`
- `figures.py`
- `per_sym.py`
- `comps_plots.py`
- `unit_cell.py`
- `functions6.py`

Avoid casual imports:

- `physics/poisson.py`
- `legacy/old.py`

### Compatibility Re-Exports

`crystal_funcs.py` still re-exports several planar helpers from
`planar_geometry.py` for notebook compatibility. If you are reorganizing more
geometry code, prefer moving the real implementation into the focused module
and keeping `crystal_funcs.py` as a thin import layer.

### In-Place Normalization

Some spherical helpers normalize inputs in place, including parts of
`geodesic`, `dist_to_geodesic`, and related routines. Pass copies if input
preservation matters.

### Diagnostics Instead Of Exceptions

Some routines print diagnostics rather than raising structured exceptions, for
example barycentric residuals, singular matrices, or zero geodesic denominators.
If you need stronger error handling, keep it local to the requested change.

### Broad Imports

Some legacy files use wildcard imports. In new code, prefer explicit imports:

```python
import crystal_funcs as cfuncs
import coordinates as cconv
```

## Development Principles

1. Keep changes narrow and compatible with notebook workflows.
2. Prefer improving the import-safe modules over extending legacy scripts.
3. Do not rewrite notebooks, generated GIFs, or cached files unless asked.
4. Add small reproducible examples or smoke checks when changing math-heavy
  helpers.
5. Preserve current array conventions unless the user explicitly asks for a
  larger API cleanup.
6. Be cautious with broad refactors. This directory mixes polished utilities,
  active experiments, and archived code.
