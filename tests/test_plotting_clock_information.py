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
  ring = ci.ring_comparison(6, 1.0, 0.1, t_max=10.0, n_t=51, Ts=(2.0, 6.0))
  fig = plot_ci.plot_ring(ring)
  assert len(fig.axes) == 3
  _draw(fig)
