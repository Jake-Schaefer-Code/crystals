# plotting/nodal.py
r"""Figures for ``physics.nodal``: signed-amplitude slices, node isosurfaces, exchange paths.

Everything here draws results of ``physics.nodal``; no computation of its own beyond display
transforms. ``physics.nodal`` does not import this module.
"""
from __future__ import annotations

import numpy as onp
import matplotlib.pyplot as plt

from physics import nodal



def signed_amplitude(sign, logabs, gamma=0.5):
  r"""``sign * (|psi| / max|psi|)^gamma``, floored away from zero so the sign survives underflow."""
  sign, logabs = onp.asarray(sign), onp.asarray(logabs)
  finite = onp.isfinite(logabs)
  top = logabs[finite].max() if finite.any() else 0.0
  return sign * onp.exp(gamma * onp.maximum(logabs - top, -300.0))


def plot_signed(ax, u, v, sign, logabs, *, gamma=0.5, cmap="RdBu_r", node_color="k"):
  r"""Draw ``sign(psi) |psi|^gamma`` on a slice with the node (psi = 0) as a solid line."""
  f = signed_amplitude(sign, logabs, gamma)
  im = ax.imshow(f, extent=(u[0], u[-1], v[0], v[-1]), origin="lower", cmap=cmap, vmin=-1, vmax=1,
                 interpolation="bilinear")
  if (f > 0).any() and (f < 0).any():
    ax.contour(u, v, f, levels=[0.0], colors=node_color, linewidths=1.4)
  return im


def plot_particle_slice(ax, psi, R, i, spins, *, axes=(0, 1), extent=None, n=201, gamma=0.5):
  r"""The node seen by particle ``i`` moving in the plane of ``axes``, others frozen at ``R``.

  Filled markers are same-spin particles, open markers opposite-spin ones, the star is r_i in R.
  Markers are projections: the node passes through a same-spin particle only if it lies in the
  plane. ``extent`` defaults to a square around the particles. Returns ``(u, v, sign, logabs)``.
  """
  R, spins = onp.asarray(R, dtype=float), onp.asarray(spins)
  a, b = axes
  if extent is None:
    center = 0.5 * (R[:, [a, b]].min(0) + R[:, [a, b]].max(0))
    half = 1.5 * onp.abs(R[:, [a, b]] - center).max() + 1e-6
    extent = (center[0] - half, center[0] + half, center[1] - half, center[1] + half)
  grid = nodal.slice_grid(psi, *nodal.coordinate_plane(R, (i, a), (i, b)), extent, n)
  im = plot_signed(ax, *grid, gamma=gamma)
  others = onp.arange(len(R)) != i
  same = others & (spins == spins[i])
  ax.scatter(R[same, a], R[same, b], s=40, c="k", zorder=3, label="same spin")
  ax.scatter(R[others & ~same, a], R[others & ~same, b], s=40, facecolors="none", edgecolors="k",
             linewidths=1.3, zorder=3, label="opposite spin")
  ax.scatter(R[i, a], R[i, b], s=180, marker="*", c="gold", edgecolors="k", zorder=4,
             label=f"particle {i}")
  u0, u1, v0, v1 = nodal.plane_extent(extent)
  ax.set(xlim=(u0, u1), ylim=(v0, v1), xlabel=f"$r_{{{i}}}$ · {'xyz'[a]}",
         ylabel=f"$r_{{{i}}}$ · {'xyz'[b]}")
  ax.figure.colorbar(im, ax=ax, shrink=0.8, label=rf"sign$(\psi)\,|\psi|^{{{gamma:g}}}$")
  return grid


def plot_particle_isosurface(psi, R, i, spins, *, extent=None, n=48, color="#4a6fa5"):
  r"""Interactive plotly figure of the nodal surface seen by particle ``i`` in 3D, others frozen.

  ``extent`` is as in ``particle_volume`` and defaults to a cube around the origin enclosing ``R``.
  """
  import plotly.graph_objects as go
  R, spins = onp.asarray(R, dtype=float), onp.asarray(spins)
  if extent is None:
    extent = 1.5 * onp.abs(R).max() + 1e-6
  x, s, l = nodal.particle_volume(psi, R, i, extent, n)
  X, Y, Z = onp.meshgrid(x, x, x, indexing="ij")
  fig = go.Figure(go.Isosurface(  # psi itself is linear across the node: smoothest interpolation
    x=X.ravel(), y=Y.ravel(), z=Z.ravel(), value=signed_amplitude(s, l, gamma=1.0).ravel(),
    isomin=0.0, isomax=0.0, surface_count=1, colorscale=[[0, color], [1, color]], showscale=False,
    opacity=0.55, caps=dict(x_show=False, y_show=False, z_show=False), name="node"))
  others = onp.arange(len(R)) != i
  same = others & (spins == spins[i])
  for mask, name, marker in [
      (same, "same spin", dict(size=6, color="black")),
      (others & ~same, "opposite spin", dict(size=6, color="white", line=dict(color="black", width=2))),
      (~others, f"particle {i}", dict(size=8, color="gold", symbol="diamond", line=dict(color="black", width=1)))]:
    fig.add_trace(go.Scatter3d(x=R[mask, 0], y=R[mask, 1], z=R[mask, 2], mode="markers", name=name,
                               marker=marker))
  fig.update_layout(scene=dict(aspectmode="cube"), margin=dict(l=0, r=0, t=30, b=0))
  return fig


def plot_exchange_path(psi, path, spins, *, axes=(0, 1), gamma=0.5, axs=None):
  r"""Left: particle trajectories along a node-free exchange path, e.g. a ``CellCensus`` witness
  (circles mark the start, squares the end, grey lines are spectators). Right: sign(psi)|psi|^gamma
  along the path, which never reaches zero."""
  path, spins = onp.asarray(path), onp.asarray(spins)
  if axs is None:
    _, axs = plt.subplots(1, 2, figsize=(10, 4))
  ax_path, ax_psi = axs
  a, b = axes
  moving = onp.linalg.norm(path[-1] - path[0], axis=-1) > 1e-9
  for j in range(path.shape[1]):
    color = ("tab:orange" if spins[j] == 0 else "tab:purple") if moving[j] else "0.65"
    ax_path.plot(path[:, j, a], path[:, j, b], color=color, lw=2.0 if moving[j] else 0.8)
    ax_path.plot(*path[0, j, [a, b]], "o", color=color)
    ax_path.plot(*path[-1, j, [a, b]], "s", color=color, mfc="none", ms=9)
  ax_path.set(aspect="equal", xlabel="xyz"[a], ylabel="xyz"[b], title="exchange path (up: orange, down: purple)")
  s, l = nodal.evaluate(psi, path)
  ax_psi.plot(onp.linspace(0.0, 1.0, len(path)), signed_amplitude(s, l, gamma), color="k")
  ax_psi.axhline(0.0, color="0.6", lw=0.8)
  ax_psi.set(xlabel="t", ylabel=rf"sign$(\psi)\,|\psi|^{{{gamma:g}}}$", title=r"$\psi$ along the path")
  return axs
