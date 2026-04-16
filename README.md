# generative_crystals / rep-theory library

Research sandbox for crystal geometry, symmetry operations, representation
visualizations, and exploratory materials-science notebooks.

Will be migrated to JAX in the future

## Periodic Symmetry / Presentation Helpers

- `per_sym.py` contains import-safe lattice, reciprocal-lattice, Bloch-wave,
  shift-operator, and simple chain-dispersion helpers.
- `comps_plots.py` builds presentation figures and animations from those helpers.
  Run it directly to generate the default asset bundle:

  ```bash
  python3 comps_plots.py
  ```

- `shift_operator.ipynb` is a cleaned notebook front end for the same ideas. Its
  outputs are intentionally cleared so reruns start from the local helper modules.

## Unit-Cell Helpers

- `unit_cell_utils.py` centralizes the shared geometry and Matplotlib primitives
  used by `unit_cell.py`, `figures.py`, and `functions6.py`.
- `unit_cell.py` and `functions6.py` are import-safe demos; their `main()`
  functions perform plotting or animation work.

## Legacy

- `legacy/old.py` keeps the older plotting/mapping scratch helpers for
  reference. Active barycentric utilities live in `coordinates.py`; the useful
  symmetry-density diagnostic plotting helpers were moved to `plotting_utils.py`.
