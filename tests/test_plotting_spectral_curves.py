# tests/test_plotting_spectral_curves.py
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as onp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physics import spectral_curves as sc
from plotting import spectral_curves as plot_sc


def test_figures_draw_from_computed_data():
  plot_sc.use_style()
  curve = sc.spectral_curve(sc.QuarticModel(1.0, -3.0))
  samples = sc.sample_log_gas(curve.model.V, 8, n_chains=2, n_sweeps=20, n_burn=5, rng=0)[0]
  for fig in (plot_sc.plot_joukowski(), plot_sc.plot_interlacing(rng=0),
              plot_sc.plot_real_section(curve, samples), plot_sc.plot_chord_diagrams(2)):
    fig.canvas.draw()
    plt.close(fig)


def test_extracted_helpers_match_their_definitions():
  rolls = sc.rotated_char_polys([1.0, 1.0, -1.0, -1.0], 5, rng=0)
  assert rolls.shape == (5, 5) and onp.allclose(rolls[:, 0], 1.0)
  roots = sc.rank_one_update_roots([-1.0, 0.0, 1.0], 4, rng=0)
  assert len(roots) == 4 and all(len(r) == 3 for r in roots)
  w = onp.array([0.3, -0.5])
  assert onp.allclose(sc.bernoulli_r(1.4, w) * w * 2, onp.sqrt(1 + 4 * 1.4**2 * w**2) - 1)
