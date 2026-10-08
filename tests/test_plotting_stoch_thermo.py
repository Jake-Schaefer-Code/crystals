# tests/test_plotting_stoch_thermo.py
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as onp
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax.numpy as jnp

from physics import simplex
from plotting import heatmaps, simplex as plot_simplex, style


# --------------------------------------------------------------------------- #
# Simplex geometry (physics) and drawing (plotting)
# --------------------------------------------------------------------------- #

def test_pure_states_map_to_the_corners_and_the_centroid_to_the_centre():
  assert onp.allclose(simplex.simplex_to_cartesian(onp.eye(3)), simplex.SIMPLEX_VERTICES)
  assert onp.allclose(simplex.simplex_to_cartesian([1 / 3] * 3), simplex.SIMPLEX_VERTICES.mean(axis=0))
  sides = [onp.linalg.norm(simplex.SIMPLEX_VERTICES[i] - simplex.SIMPLEX_VERTICES[j]) for i, j in ((0, 1), (1, 2), (0, 2))]
  assert onp.allclose(sides, 1.0)                                       # equilateral


@pytest.mark.parametrize("levels", [1, 5, 45])
def test_the_mesh_covers_the_simplex_once(levels):
  pts = onp.asarray(simplex.sample_simplex(levels))
  assert pts.shape == ((levels + 1) * (levels + 2) // 2, 3)
  assert onp.allclose(pts.sum(axis=1), 1.0, atol=1e-6) and pts.min() >= 0.0
  assert len({tuple(onp.round(p * levels).astype(int)) for p in pts}) == len(pts)


def test_regularization_keeps_the_mesh_inside_and_normalized():
  pts = simplex.regularize_simplex(simplex.sample_simplex(10), eps=1e-3)
  assert float(pts.min()) > 0.0 and onp.allclose(onp.asarray(pts).sum(axis=1), 1.0, atol=1e-6)
  assert onp.allclose(simplex.regularize_simplex(jnp.array([[0.2, 0.3, 0.5]])), [[0.2, 0.3, 0.5]], atol=1e-6)
  assert onp.allclose(simplex.normalize_simplex_point([2.0, 1.0, 1.0]), [0.5, 0.25, 0.25])


def test_probability_formatting_in_both_styles():
  assert plot_simplex.format_prob([0.25, 0.5, 0.25]) == "(0.25, 0.50, 0.25)"
  assert plot_simplex.format_prob([0.25, 0.5, 0.25], latex=True) == r"\left(0.25,\,0.50,\,0.25\right)"


@pytest.mark.parametrize("inside", [False, True])
def test_frame_and_corner_labels_draw(inside):
  fig, ax = plt.subplots()
  plot_simplex.draw_simplex_frame(ax)
  plot_simplex.add_corner_labels(ax, inside=inside)
  assert len(ax.lines) == 1 and len(ax.texts) == 3
  fig.canvas.draw()
  plt.close(fig)


# --------------------------------------------------------------------------- #
# Heatmaps and style
# --------------------------------------------------------------------------- #

def test_heatmap_figures_draw_and_are_returned_not_shown():
  rng = onp.random.default_rng(0)
  h1, h2 = rng.random((6, 7)), rng.random((6, 7))
  extent = [-1.0, 1.0, 0.5, 3.0]
  fig = heatmaps.render_heatmap(h1, 10, extent)
  fig.canvas.draw()
  plt.close(fig)
  fig = heatmaps.compare_heatmaps(h1, h2, 10, 4, 0.1, 0.2, 1.0, 2.0, extent, cmap="magma", delta_cmap="RdBu_r")
  assert len(fig.axes) >= 3
  assert fig.axes[2].images[0].norm.vcenter == 0.0                      # difference on a diverging scale
  fig.canvas.draw()
  plt.close(fig)


def test_identical_heatmaps_get_a_finite_difference_scale():
  h = onp.ones((3, 3))
  fig = heatmaps.compare_heatmaps(h, h, 1, 2, 0.1, 0.0, 1.0, 1.0, [0, 1, 0, 1])
  norm = fig.axes[2].images[0].norm
  assert norm.vmin == -1.0 and norm.vmax == 1.0
  plt.close(fig)


def test_log_norm_masks_nonpositive_entries_and_rejects_all_nonpositive_data():
  data = onp.array([[1.0, 10.0], [0.0, -2.0]])
  masked, norm = heatmaps.make_log_data_and_norm(data)
  assert masked.mask.tolist() == [[False, False], [True, True]]
  assert (norm.vmin, norm.vmax) == (1.0, 10.0)
  with pytest.raises(ValueError):
    heatmaps.make_log_data_and_norm(onp.zeros((2, 2)))


def test_paper_style_is_a_plain_dict_that_applies_on_request(monkeypatch):
  assert style.PAPER_RCPARAMS["text.usetex"] is True and style.PAPER_RCPARAMS["pdf.fonttype"] == 42
  saved = dict(matplotlib.rcParams)
  try:
    monkeypatch.setitem(style.PAPER_RCPARAMS, "text.usetex", False)
    style.use_paper_style()
    assert matplotlib.rcParams["font.size"] == 16
  finally:
    matplotlib.rcParams.update(saved)
