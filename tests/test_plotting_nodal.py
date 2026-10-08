# tests/test_plotting_nodal.py
import sys
from pathlib import Path

import numpy as onp
import jax

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
jax.config.update("jax_enable_x64", True)

from physics import nodal, nodal_models as models
from plotting import nodal as plot_nodal

KEY = jax.random.PRNGKey(0)
HO2 = models.slater_determinant(models.harmonic_orbitals(2, 3), [0] * 3)


def _census(psi, spins, dim, step):
  k_equil, k_census = jax.random.split(KEY)
  samples, _ = nodal.metropolis(psi, jax.random.normal(k_equil, (16, len(spins), dim)), k_equil, 200, step)
  return nodal.cell_census(psi, spins, samples[-1][0], k_census, step=step)


def test_plotting_smoke():
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  census = _census(HO2, [0] * 3, 2, 0.5)
  fig, axs = plt.subplots(1, 3)
  R = census.witnesses[census.certified[0]][0]
  plot_nodal.plot_particle_slice(axs[0], HO2, R, 0, [0] * 3, n=40)
  plot_nodal.plot_exchange_path(HO2, census.witnesses[census.certified[0]], [0] * 3, axs=axs[1:])
  plt.close(fig)
  R_be = onp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.2], [1.0, 0.0, 0.3], [-0.5, 0.0, -0.9]])
  fig = plot_nodal.plot_particle_isosurface(models.beryllium(-0.2), R_be, 0, [0, 0, 1, 1], extent=2.0, n=12)
  assert len(fig.data) == 4
