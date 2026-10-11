# AGENTS.md

Conventions and working rules for coding agents and maintainers. Layout, commands and known issues are in `README.md`; history is in `DEVLOG.md`. Prefer mathematically faithful abstractions over plotting shortcuts.

This is a research sandbox: a folder of importable modules plus notebooks and example scripts. Keep changes narrow and compatible with the notebooks that import them.

## Before you edit

- Read the module docstring. The newer modules (`physics/markov`, `glauber`, `nodal`, `dfa_memory`, `src/symmetry/group_laws`, ...) state their conventions there: generator orientation, which probability `p` means, protocols. They are not repeated here.
- Check `DEVLOG.md` for why something is the way it is, and add an entry in its style for notable changes: what moved, and what it was checked against.
- Do not nest another project's checkout inside this tree. Both this repo and `agca_jax` ship a top-level `src`, so a nested copy made a bare `pytest` abort at collection.

## Conventions

### Array shapes

Points are rows and coordinate components are on the last axis:

```python
points_2d.shape == (n_points, 2)
points_3d.shape == (n_points, 3)
triangle.shape == (3, 2)
tetrahedron.shape == (4, 3)
```

Batch dimensions usually come before the simplex and point axes. Check existing call sites before changing broadcasting behavior.

### Polygon ordering

2D polygon routines assume vertices are ordered counterclockwise; `planar_geometry.area` returns a signed area. This affects area, triangulation and inclusion checks, edge interpolation, self-intersection tests and mapping quality. If a result looks sign-flipped or self-intersecting, check vertex order before rewriting the math.

### Barycentric coordinates

`src/coordinates.py` is the home for coordinate conversion: `barycentric_coordinates2D`, `barycentric_to_cartesian_2D`, `cart_to_bary_tetra`, `bary_to_cart`, `barycentric_weights` and `calculate_barycentric_coordinates` (least squares; prints the residual when it is nonzero). `legacy/old.py` (local, gitignored) has older duplicates; do not copy them unless the task is reproducing legacy behavior.

### Rotations and symmetry operations

Quaternions are `[w, x, y, z]`. Rotation helpers live in `src/geo_ops_utils.py` (`rotation_quaternions`, `make_q_rot_mats`, `q_rot_mat`, `rotate`, `quaternion_multiply`, `generate_rotation_matrices`, `generate_rot_mats2d`, `rotate_polyhedron`, `rot_mat`, ...).

A symmetry operation is an `AffineOperation` (`x -> matrix @ x + translation`) or a `SeitzOp` (`R`, `tau`), both in `src/symmetry/symmetry.py`; `src/symmetry/operations.py` builds the common ones. `src/categorization.py` parses only a small subset of space-group notation: extend it incrementally and document examples when adding symbols.

### Wavefunctions

Antisymmetric wavefunctions follow the `physics.nodal` protocol `psi(R) -> (sign, log|psi|)`. Nodal and VMC work needs float64. Library modules never set `jax_enable_x64`; scripts and tests do, and `physics.ansatz1d` reads the flag, so do not flip it from library code.

### Plotting

Figure code lives in `plotting/` (and in `plotting_utils.py` for the geometry helpers); numerics live in `physics/`. A figure function takes a result and returns a `Figure`; saving is the caller's job (`plotting_utils.save_path`). Do not add plotting side effects at import time: anything that shows or writes a figure goes behind `if __name__ == "__main__":`. A few older helpers still call `plt.show()` inside functions (`physics/bloch.py`, `physics/density_evolution.py`).

## Error-prone areas

### Import safety

Everything under `src/`, `plotting/` and `physics/` imports cleanly except `physics/dbl.py` (runs a simulation at import), `physics/bloch.py` (opens a figure) and `physics/hartree_fock.py` (prints). `TriMap.py`, `plotting_utils.py` and `unit_cell_utils.py` are safe too. Scripts in `examples/` are not modules; do not import them (`lattice_plots.py` has no `__main__` guard). Do not import every `*.py` file as a validation step.

The numpy-only layers must stay free of a module-level `import jax`: `src/symmetry/`, `src/exact_gb.py`, `src/exact_poly.py`, `src/core/` (its type aliases use `TYPE_CHECKING`), `src/config/` and the geometry modules. That is why `import src.exact_poly` takes a few hundredths of a second and why notebooks that only need numpy start fast.

### Compatibility re-exports

`src/crystal_funcs.py` still re-exports planar helpers from `src/planar_geometry.py` for notebook compatibility, and `src/permutations.py` aliases `src/symmetry/permutations.py` for the `rep` and `quantum` notebooks and `src/quantum_symmetry.py`. When reorganizing, move the implementation into the focused module and leave a thin alias. Check `notebooks/` before breaking an import path.

### In-place normalization and printed diagnostics

`coordinates.geodesic` and `coordinates.dist_to_geodesic` normalize their arguments in place; pass copies. `calculate_barycentric_coordinates` and `geodesic` print a diagnostic (residual, zero denominator) instead of raising. Keep any stronger error handling local to the change you were asked for.

### Wildcard imports

`src/symmetry/__init__.py` and `src/core/__init__.py` star-import their submodules, and `src/permutations.py` star-imports its target. In new code, import names explicitly.

### Bytecode

`__pycache__` and `*.pyc` are gitignored and untracked. Do not `git add -f` them. `python -I` ignores `PYTHONPYCACHEPREFIX`, so use `python -B` when you import repo modules that way.

## Testing

`pytest` collects only `tests/` (`pytest.ini`), one file per module. When you change a math-heavy helper, add a small deterministic check with a known answer. Prefer identities that two independent routes must agree on, as the Molien checks do (`tests/test_young.py` against `src/invariants.py`), and keep a plotting smoke test beside any new figure module. Keep generated figures and animations out of version control.

## Development principles

1. Keep changes narrow and compatible with notebook workflows.
2. Improve the import-safe modules rather than extending scripts.
3. Do not rewrite notebooks (their outputs are tracked), generated GIFs or cached files unless asked.
4. Add small reproducible checks when changing math-heavy helpers.
5. Preserve current array conventions unless asked for a larger API cleanup.
6. Be cautious with broad refactors: this directory mixes polished utilities, active experiments and archived code.

## Behavioral guidelines

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.