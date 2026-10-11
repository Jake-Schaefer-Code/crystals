# plotting/clock_information.py
r"""Figures for ``physics.clock_information``: clock information and time-averaged mismatch cost.

One function per analysis, each taking the dataclass the matching ``physics.clock_information``
function returns and giving back a ``Figure``. The numerics and their consistency checks live
there; this module only draws, and ``physics`` does not import it. Save with
``plotting_utils.save_path`` or ``fig.savefig``.

Log-ratio panels use a diverging scale clipped at ``+-clip``: blue where a state is evidence for that
time (``p_t > qbar``), red where it is evidence against, and black contours on the nodal set
``p_t = qbar``. ``plot_blur`` stacks the weighted integrand over the log-ratio and draws the
heavy-blur prediction for that nodal set (``physics.clock_information.nodal_cross``) on both rows.
``plot_ring_hamiltonian`` draws the levels and first eigenfunctions of the dynamical Hamiltonian.
"""
from __future__ import annotations

from collections.abc import Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from physics.clock_information import (BlurScan, ClockDecay, NodalCross, RingComparison, RingHamiltonian, SpaceTimeWindow,
                                       WindowScan, ring_interpolate)

_CROSS_COLOR = "limegreen"
_HYPERBOLA_COLOR = "magenta"


def _log_ratio_panel(ax: Axes, ts, xs, field, *, clip: float):
  r""" Diverging log-ratio heatmap (time across, state up) with the nodal contour ``field = 0``. """
  norm = mcolors.TwoSlopeNorm(vmin=-clip, vcenter=0.0, vmax=clip)
  extent = [ts[0], ts[-1], xs[0], xs[-1]]
  im = ax.imshow(np.clip(field, -clip, clip).T, origin="lower", extent=extent, aspect="auto",
                 cmap="RdBu", norm=norm)
  ax.contour(ts, xs, field.T, levels=[0.0], colors="k", linewidths=0.7)
  return im


def _crop(xs, x_range):
  keep = (xs >= x_range[0]) & (xs <= x_range[1])
  return keep, xs[keep]


def plot_space_time(result: SpaceTimeWindow, *, x_range=(-3.0, 4.0), clip: float = 3.0, weighted: bool=True) -> Figure:
  r""" Space-time density ``p_t(x)`` and the clock-information field ``log(p_t / qbar)``. """
  r = result
  keep, xs = _crop(r.xs, x_range)
  dx = r.xs[1] - r.xs[0]

  fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
  im = axes[0].imshow((r.P[:, keep] / dx).T, origin="lower", extent=[r.ts[0], r.ts[-1], xs[0], xs[-1]],
                      aspect="auto", cmap="Blues")
  axes[0].set(xlabel=r"time $t$", ylabel=r"state $x$", title=r"Space-time density $p_t(x)$")
  fig.colorbar(im, ax=axes[0], label="density")
  if weighted:
    field = r.P[:, keep] * r.log_ratio[:, keep]
  else:
    field = r.log_ratio[:, keep]
  im = _log_ratio_panel(axes[1], r.ts, xs, field, clip=clip)
  axes[1].set(xlabel=r"time $t$", ylabel=r"state $x$",
              title=fr"$\log(p_t/\bar q)$, $I(\tau;X)={r.info:.3f}$ nats")
  fig.colorbar(im, ax=axes[1], label="pointwise clock information")
  return fig


def plot_clock_decay(result: ClockDecay) -> Figure:
  r""" ``I(tau; X_{tau+s})`` against extra evolution time, with its tangent at ``s = 0``. """
  r = result
  fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
  ax.plot(r.shifts, r.info, lw=2.2, label=r"$I(\tau; X_{\tau+s})$")
  s_tan = np.linspace(0.0, r.info[0] / r.rate, 50)
  ax.plot(s_tan, r.info[0] - r.rate * s_tan, "--", lw=1.6,
          label=fr"tangent, slope $=-{r.rate:.3f}$ (minimal cost rate)")
  ax.set(xlabel=r"extra evolution time $s$", ylabel="clock information (nats)",
         title="Clock information erased by the homogeneous dynamics", ylim=(0.0, 1.08 * r.info[0]))
  ax.legend(frameon=False)
  return fig


def plot_window_scans(scans: Sequence[tuple[str, WindowScan]], *,
                      prediction: tuple[str, np.ndarray, np.ndarray] | None = None) -> Figure:
  r""" ``MC*_T / D(p_0 || pi)`` against window length; ``prediction = (label, Ts, values)`` is drawn dashed. """
  fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
  for label, scan in scans:
    ax.plot(scan.Ts, scan.mmc / scan.budget, marker="o", ms=3.5, lw=1.8, label=label)
  if prediction is not None:
    label, Ts, values = prediction
    ax.plot(Ts, values, "k--", lw=1.2, label=label)
  ax.axhline(1.0, color="0.5", lw=0.8, ls=":")
  ax.set(xlabel=r"window length $T$", ylabel=r"$\mathcal{MC}^*_T \,/\, D(p_0\Vert\pi)$",
         title="Minimal total mismatch cost over a window", ylim=(0.0, 1.05))
  ax.legend(frameon=False)
  return fig


def _cross_overlay(ax: Axes, cross: NodalCross, sigma: float, xs):
  r""" Heavy-blur nodal set on a (time across, state up) panel: ``t = T/2`` and ``x = x*`` dashed, the hyperbola solid. """
  t_mid = 0.5 * cross.T
  ax.axvline(t_mid, color=_CROSS_COLOR, ls="--", lw=1.2, zorder=1.5)
  for x_star, product in zip(cross.x_star, cross.product(sigma)):
    if not xs[0] <= x_star <= xs[-1]:
      continue
    ax.axhline(x_star, color=_CROSS_COLOR, ls="--", lw=1.2, zorder=1.5)
    u = np.geomspace(max(abs(product) / max(xs[-1] - x_star, x_star - xs[0]), 1e-9 * cross.T), t_mid, 400)
    for side in (-1.0, 1.0):
      x = x_star + product / (side * u)
      ax.plot(t_mid + side * u, np.where((x >= xs[0]) & (x <= xs[-1]), x, np.nan),
              color=_HYPERBOLA_COLOR, lw=1.2, zorder=1.5)


def plot_blur(result: BlurScan, *, x_range=(-3.0, 4.0), clip: float = 3.0, cross: bool = True) -> Figure:
  r""" Smoothed log-ratio fields at ``display_sigmas`` and the retained fractions against clock blur.

  Top row: the integrand ``p^sigma_t(x) log(p^sigma_t / qbar)`` (a density in ``x``), whose window average
  is ``I_sigma``, on one symmetric scale shared by the row. Bottom row: ``log(p^sigma_t / qbar)``, clipped
  at ``+-clip``. Both rows have the same nodal set (black). With ``cross=True`` they also show its
  heavy-blur prediction ``result.cross``: the lines ``t = T/2`` and ``x = x*`` (dashed) and the hyperbola
  ``(t - T/2)(x - x*) = c exp(-3 pi^2 sigma^2 / (2 T^2))`` (solid), which is only accurate at large ``sigma``.
  """
  r = result
  keep, xs = _crop(r.xs, x_range)
  n = len(r.display_sigmas)
  weighted = r.display_P[:, :, keep] / (r.xs[1] - r.xs[0]) * r.log_ratios[:, :, keep]
  rows = [(weighted, float(np.max(np.abs(weighted))), r"$p^\sigma_t\,\log(p^\sigma_t/\bar q)$"),
          (r.log_ratios[:, :, keep], clip, r"$\log(p^\sigma_t/\bar q)$")]
  fig = plt.figure(figsize=(4.0 * (n + 1), 7.4), constrained_layout=True)
  gs = fig.add_gridspec(2, n + 1)
  axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(n)] for i in range(2)])
  for row_axes, (fields, vmax, label) in zip(axes, rows):
    for ax, sigma, field in zip(row_axes, r.display_sigmas, fields):
      im = _log_ratio_panel(ax, r.ts, xs, field, clip=vmax)
      if cross:
        _cross_overlay(ax, r.cross, sigma, xs)
      ax.set(xlim=(r.ts[0], r.ts[-1]), ylim=(xs[0], xs[-1]))
    row_axes[0].set_ylabel(r"state $x$")
    fig.colorbar(im, ax=row_axes, label=label)
  for ax, sigma in zip(axes[0], r.display_sigmas):
    ax.set_title(fr"clock blur $\sigma={sigma:.2f}$")
  for ax in axes[1]:
    ax.set_xlabel(r"time $t$")
  ax = fig.add_subplot(gs[:, n])
  ax.plot(r.sigmas, r.info / r.info[0], lw=2.0, label=r"clock information $I_\sigma/I_0$")
  ax.plot(r.sigmas, r.mmc / r.mmc[0], "--", lw=2.0, label=r"minimal cost $M_\sigma/M_0$")
  for sigma in r.display_sigmas:
    ax.axvline(sigma, color="0.7", lw=0.8)
  ax.set(xlabel=r"clock blur $\sigma$", ylabel="fraction retained", ylim=(0.0, 1.05), title="Retained under blur")
  ax.legend(frameon=False, fontsize=9)
  if cross:
    stars = ", ".join(f"{x:.3f}" for x in r.cross.x_star if xs[0] <= x <= xs[-1])
    fig.legend(handles=[
      Line2D([], [], color="k", lw=0.8, label=r"nodal set $p^\sigma_t=\bar q$"),
      Line2D([], [], color=_CROSS_COLOR, ls="--", lw=1.2,
             label=r"heavy-blur limit: $t=T/2$" + (fr", $x^*={stars}$" if stars else "")),
      Line2D([], [], color=_HYPERBOLA_COLOR, lw=1.2, label=r"hyperbola $(t-T/2)(x-x^*)=c\,e^{-3\pi^2\sigma^2/(2T^2)}$")],
      loc="outside lower center", ncol=3, frameon=False)
  return fig


def plot_ring(result: RingComparison, *, labels=("driven ring", "reversible ring")) -> Figure:
  r""" Probability of the start state, the slowest-mode trajectory, and ``MC*_T / D(p_0||pi)`` for both rings. """
  r = result
  fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
  ax = axes[0]
  for row, label in zip(r.start_prob, labels):
    ax.plot(r.ts, row, lw=1.9, label=label)
  ax.axhline(r.start_prob[:, -1].mean(), color="0.5", lw=0.8, ls=":")
  ax.set(xlabel=r"time $t$", ylabel=r"$p_t(0)$", title="Probability of the start state")
  ax.legend(frameon=False)

  ax = axes[1]
  for row, label in zip(r.mode, labels):
    ax.plot(row.real, row.imag, lw=1.6, label=label)
  ax.plot([1.0], [0.0], "ko", ms=4)
  ax.plot([0.0], [0.0], "k*", ms=9, label=r"$\pi$")
  lam = r.slowest[0]
  ax.set(xlabel="Re (slowest-mode amplitude)", ylabel="Im", aspect="equal",
         title=fr"Slowest mode, $\lambda={lam.real:.3f}\pm{abs(lam.imag):.3f}i$")
  ax.legend(frameon=False, fontsize=9)

  ax = axes[2]
  for scan, label in zip(r.scans, labels):
    ax.plot(scan.Ts, scan.mmc / scan.budget, marker="o", ms=3.5, lw=1.8, label=label)
  ax.axhline(1.0, color="0.5", lw=0.8, ls=":")
  ax.set(xlabel=r"window length $T$", ylabel=r"$\mathcal{MC}^*_T \,/\, D(p_0\Vert\pi)$",
         title="Minimal total mismatch cost", ylim=(0.0, 1.05))
  ax.legend(frameon=False)
  return fig


def _gap_label(gap: float) -> str:
  return r"$<10^{-6}$" if gap < 1e-6 else f"${gap:.3f}$"


def plot_ring_hamiltonian(result: RingHamiltonian, *, up: int = 12) -> Figure:
  r""" Levels of ``H_dyn`` for four rings and the first eigenfunction of each with nodes in the state.

  Left: the lowest numerical levels (black) beside ``normal_levels`` (coloured by ring), with the ``n``-fold zero
  level at the bottom. Right: ``p(t, x)`` of the first level that changes sign along the ring, with its nodal lines
  (black) and the node ``t = T/2`` of the blur's first time mode (dashed); rows are uniform and in a potential,
  columns reversible and driven. The state is trigonometrically interpolated between sites, ``up`` points per site.
  """
  r = result
  fig = plt.figure(figsize=(15, 7.6), constrained_layout=True)
  gs = fig.add_gridspec(2, 3, width_ratios=[0.8, 1.5, 1.5])
  ax = fig.add_subplot(gs[:, 0])
  for i, s in enumerate(r.systems):
    color = f"C{i}"
    ax.hlines(s.energies, i - 0.4, i - 0.05, color="k", lw=1.6)
    ax.hlines(s.normal, i + 0.05, i + 0.4, color=color, lw=1.6)
  zero = int(np.sum(r.systems[0].energies < 1e-6))
  ax.text(1.5, 0.06, fr"zero level, $\times{zero}$: the trajectories", ha="center", fontsize=9)
  ax.set_xticks(range(4), [s.label.replace(", ", "\n") for s in r.systems], fontsize=9)
  ax.set(ylabel=r"level of $H_{\rm dyn}$", title="Levels", xlim=(-0.6, 3.6),
         ylim=(-0.1, 1.2 * max(s.energies.max() for s in r.systems)))
  ax.legend(handles=[Line2D([], [], color="k", lw=1.6, label="numerics"),
                     Line2D([], [], color="0.6", lw=1.6, label=r"$(k\pi/T)^2+(\mathrm{Re}\,\mu_j)^2$")],
            loc="upper left", frameon=False, fontsize=9)
  T, n = r.ts[-1], r.systems[0].state.shape[1]
  for i, s in enumerate(r.systems):
    ax = fig.add_subplot(gs[i // 2, 1 + i % 2])
    field = ring_interpolate(s.state, up)
    sites = np.arange(field.shape[1]) / up
    ax.imshow(field.T, origin="lower", extent=[r.ts[0], T, -0.5 / up, n - 0.5 / up], aspect="auto", cmap="RdBu",
              norm=mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0))
    ax.contour(r.ts, sites, field.T, levels=[0.0], colors="k", linewidths=1.0)
    ax.axvline(0.5 * T, color="0.35", ls="--", lw=1.0)
    ax.set(title=f"{s.label}\n" + fr"$E={s.level:.3f}$, $s_2/s_1={s.rank_residual:.2f}$, saddle gap {_gap_label(s.gap)}",
           xlim=(r.ts[0], T), ylim=(0.0, n))
    if i % 2 == 0:
      ax.set_ylabel(r"ring site $x$")
    if i // 2 == 1:
      ax.set_xlabel(r"time $t$")
  fig.legend(handles=[Line2D([], [], color="k", lw=1.0, label=r"nodal lines of $p(t,x)$"),
                      Line2D([], [], color="0.35", ls="--", lw=1.0, label=r"$t=T/2$, node of the blur's first time mode")],
             loc="outside lower center", ncol=2, frameon=False)
  return fig
