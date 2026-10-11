# tests/test_plotting_clock_information.py
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as onp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from matplotlib.figure import Figure

import physics.clock_information as ci
from plotting import clock_information as plot_ci

XS = jnp.linspace(-6.0, 6.0, 241)


def _draw(fig):
  assert isinstance(fig, Figure)
  fig.canvas.draw()
  plt.close(fig)


def test_space_time_draws_the_nodal_contour():
  fig = plot_ci.plot_space_time(ci.ou_space_time(T=3.0, n_t=61, xs=XS))
  assert len(fig.axes[1].collections) >= 1                              # contour of log(p_t/qbar) = 0
  _draw(fig)


def test_decay_and_window_figures_draw():
  _draw(plot_ci.plot_clock_decay(ci.ou_clock_decay(onp.linspace(0.0, 2.0, 5), T=3.0, n_t=61, xs=XS)))
  scan = ci.ou_window_scan([1.0, 4.0], xs=XS, n_t=121)
  Ts = onp.array([1.0, 4.0])
  fig = plot_ci.plot_window_scans([("far", scan)], prediction=("2g", Ts, 2 * onp.asarray(ci.mode_filter_uniform(Ts))))
  assert len(fig.axes[0].lines) == 3                                    # scan, prediction, budget line
  _draw(fig)


def test_blur_and_ring_figures_draw():
  blur = ci.ou_blur_scan([0.0, 0.5], T=3.0, n_t=60, xs=XS, display_sigmas=(0.0, 0.5))
  _draw(plot_ci.plot_blur(blur))
  _draw(plot_ci.plot_blur(blur, cross=False))
  ring = ci.ring_comparison(6, 1.0, 0.1, t_max=10.0, n_t=51, Ts=(2.0, 6.0))
  fig = plot_ci.plot_ring(ring)
  assert len(fig.axes) == 3
  _draw(fig)


def test_blur_figure_stacks_weighted_over_unweighted_and_overlays_the_predicted_cross():
  blur = ci.ou_blur_scan([0.0, 2.0], T=3.0, n_t=60, xs=XS, display_sigmas=(0.0, 2.0))
  fig = plot_ci.plot_blur(blur)
  panels = sorted((ax for ax in fig.axes if ax.images), key=lambda ax: (-round(ax.get_position().y0, 3), ax.get_position().x0))
  assert len(panels) == 4                                               # two rows, a panel per display sigma
  keep = (blur.xs >= -3.0) & (blur.xs <= 4.0)
  weighted = blur.display_P[0][:, keep] / (blur.xs[1] - blur.xs[0]) * blur.log_ratios[0][:, keep]
  assert onp.allclose(onp.asarray(panels[0].images[0].get_array()), weighted.T)                       # top left: the integrand
  assert onp.allclose(onp.asarray(panels[2].images[0].get_array()),
                      onp.clip(blur.log_ratios[0][:, keep], -3.0, 3.0).T)                             # below it: the log-ratio
  for ax, sigma in zip(panels, (0.0, 2.0, 0.0, 2.0)):
    branches = [line for line in ax.lines if len(line.get_xdata()) > 2]
    assert len(ax.lines) == 4 and len(branches) == 2                    # t = T/2, x = x*, two hyperbola branches
    for line in branches:
      t, x = line.get_data()
      ok = onp.isfinite(x)
      assert onp.allclose((t[ok] - 1.5) * (x[ok] - blur.cross.x_star[0]), blur.cross.product(sigma)[0])
  assert len(fig.legends) == 1
  _draw(fig)
  bare = plot_ci.plot_blur(blur, cross=False)
  assert not any(ax.lines for ax in bare.axes if ax.images) and not bare.legends
  _draw(bare)


def test_ring_hamiltonian_figure_draws_levels_and_four_eigenfunctions():
  r = ci.ring_hamiltonian(6, 1.0, 0.05, T=6.0, n_t=41, depth=2.0, n_levels=15)
  fig = plot_ci.plot_ring_hamiltonian(r)
  panels = [ax for ax in fig.axes if ax.images]
  assert len(fig.axes) == 5 and len(panels) == 4                        # the level diagram and a 2 x 2 grid of states
  assert all(len(ax.collections) >= 1 and len(ax.lines) == 1 for ax in panels)   # nodal contour, the dashed t = T/2
  assert len(fig.axes[0].collections) == 8                              # numerical and formula levels of four rings
  assert len(fig.legends) == 1
  _draw(fig)
