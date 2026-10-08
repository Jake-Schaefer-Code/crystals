# tests/test_plotting_ising_pop.py
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as onp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physics import ising_sectors as sec
from plotting import ising_pop


def test_every_figure_draws_from_computed_data():
  figures = [
    ising_pop.plot_same_G_hidden_housekeeping(
      sec.hidden_housekeeping(n_spins=5, n_cycles=40, contrasts=[0.0, 0.2, 0.5, 1.0])
    ),
    ising_pop.plot_symmetry_bit([sec.symmetry_bit_relaxation(n, n_cycles=100) for n in (5, 7)]),
    ising_pop.plot_critical_scaling(
      sec.critical_scan(n_spins=9, J_values=onp.linspace(0.3, 0.7, 4), max_time=5.0, horizons=[0.5, 5.0])
    ),
    ising_pop.plot_periodic_drive(
      sec.periodic_drive_scan(n_spins=7, periods=onp.logspace(-1, 1, 4), n_observation_cycles=10)
    ),
  ]
  for fig in figures:
    fig.canvas.draw()
    assert len(fig.axes) >= 2
    plt.close(fig)
