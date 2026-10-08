# plotting/heatmaps.py
r"""Heatmaps of a quantity over a two-parameter sweep (field and inverse temperature, coupling and disorder).

``add_heatmap`` draws one panel onto an axis; ``render_heatmap`` and ``compare_heatmaps`` build whole
figures and return them, leaving ``plt.show`` and saving to the caller.
"""
from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def add_heatmap(ax: Axes, heatmap, extent, *, cmap="turbo", norm=None):
  r""" ``imshow`` with the origin at the lower left, ``extent = [x0, x1, y0, y1]``, free aspect. """
  return ax.imshow(heatmap, origin="lower", extent=extent, aspect="auto", cmap=cmap, norm=norm)


def render_heatmap(heatmap, Nsteps, extent, *, cmap="turbo") -> Figure:
  r""" One MMC heatmap in the ``(h, beta)`` plane at ``Nsteps`` repetitions. """
  fig, ax = plt.subplots(1, 1, figsize=(8, 6), squeeze=True)
  im = add_heatmap(ax, heatmap, extent, cmap=cmap)
  ax.set(xlabel=r"$h$", ylabel=r"$\beta$", title=fr"MMC at $N={Nsteps}$")
  fig.colorbar(im, ax=ax, label=r"$\mathcal{MC}_N$")
  fig.tight_layout()
  return fig


def compare_heatmaps(
  h1, h2, Nsteps, N, dt, h, J, beta1, extent, *, cmap="turbo", delta_cmap="seismic"
) -> Figure:
  r""" Multi-reservoir MMC, single-reservoir MMC, and their difference on a symmetric diverging scale. """
  fig, _axs = plt.subplots(1, 3, figsize=(20, 6), squeeze=False)
  axs: list[Axes] = _axs.flatten().tolist()
  delta = np.asarray(h1 - h2)
  delta_absmax = float(np.max(np.abs(delta)))
  half_range = delta_absmax if delta_absmax > 0.0 else 1.0
  delta_norm = mcolors.TwoSlopeNorm(vmin=-half_range, vcenter=0.0, vmax=half_range)

  im = add_heatmap(axs[0], h1, extent, cmap=cmap)
  axs[0].set(xlabel=r"$h$", ylabel=r"$\beta$", title=fr"MMC multi-reservoir; $\beta_1={beta1}$")
  fig.colorbar(im, ax=axs[0], label=r"$\mathcal{MC}_N$")

  im = add_heatmap(axs[1], h2, extent, cmap=cmap)
  axs[1].set(xlabel=r"$h$", ylabel=r"$\beta$", title=r"MMC single-reservoir")
  fig.colorbar(im, ax=axs[1], label=r"$\mathcal{MC}_N$")

  im = axs[2].imshow(
    delta, origin="lower", extent=extent, aspect="auto", cmap=delta_cmap, norm=delta_norm
  )
  axs[2].set(xlabel=r"$h$", ylabel=r"$\beta$", title=r"$\Delta$MMC")
  fig.colorbar(im, ax=axs[2], label=r"$\Delta\mathcal{MC}_N$")
  fig.suptitle(fr"Minimal Periodic MMC for $N={Nsteps}$, $|\Lambda|={N}$, $dt={dt:.4f}$, $h={h}$, $J={J}$")
  fig.tight_layout()
  return fig


def make_log_data_and_norm(data):
  r""" Mask nonpositive and non-finite entries and return ``(masked data, LogNorm)`` over the positive range. """
  data = np.asarray(data)

  positive = data[np.isfinite(data) & (data > 0)]
  if positive.size == 0:
    raise ValueError("Log-scaled data contains no positive values")

  masked = np.ma.masked_invalid(data)
  masked = np.ma.masked_less_equal(masked, 0.0)

  norm = mcolors.LogNorm(vmin=positive.min(), vmax=positive.max())
  return masked, norm
