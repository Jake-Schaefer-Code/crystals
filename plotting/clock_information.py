# plotting/clock_information.py
r"""Figures for ``physics.clock_information``: clock information and time-averaged mismatch cost.

One function per analysis, each taking the dataclass the matching ``physics.clock_information``
function returns and giving back a ``Figure``. The numerics and their consistency checks live
there; this module only draws, and ``physics`` does not import it. Save with
``plotting_utils.save_path`` or ``fig.savefig``.

Log-ratio panels use a diverging scale clipped at ``+-clip``: blue where a state is evidence for that
time (``p_t > qbar``), red where it is evidence against, and black contours on the nodal set
``p_t = qbar``.
"""
from __future__ import annotations

from collections.abc import Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from physics.clock_information import BlurScan, ClockDecay, RingComparison, SpaceTimeWindow, WindowScan


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


def plot_space_time(result: SpaceTimeWindow, *, x_range=(-3.0, 4.0), clip: float = 3.0) -> Figure:
  r""" Space-time density ``p_t(x)`` and the clock-information field ``log(p_t / qbar)``. """
  r = result
  keep, xs = _crop(r.xs, x_range)
  dx = r.xs[1] - r.xs[0]
  fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
  im = axes[0].imshow((r.P[:, keep] / dx).T, origin="lower", extent=[r.ts[0], r.ts[-1], xs[0], xs[-1]],
                      aspect="auto", cmap="Blues")
  axes[0].set(xlabel=r"time $t$", ylabel=r"state $x$", title=r"Space-time density $p_t(x)$")
  fig.colorbar(im, ax=axes[0], label="density")
  im = _log_ratio_panel(axes[1], r.ts, xs, r.log_ratio[:, keep], clip=clip)
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


def plot_blur(result: BlurScan, *, x_range=(-3.0, 4.0), clip: float = 3.0) -> Figure:
  r""" Smoothed log-ratio fields at ``display_sigmas`` and the retained fractions against clock blur. """
  r = result
  keep, xs = _crop(r.xs, x_range)
  n = len(r.display_sigmas)
  fig, axes = plt.subplots(1, n + 1, figsize=(4.0 * (n + 1), 3.8), constrained_layout=True)
  for ax, sigma, field in zip(axes[:n], r.display_sigmas, r.log_ratios):
    im = _log_ratio_panel(ax, r.ts, xs, field[:, keep], clip=clip)
    ax.set(xlabel=r"time $t$", title=fr"clock blur $\sigma={sigma:.2f}$")
  axes[0].set_ylabel(r"state $x$")
  fig.colorbar(im, ax=axes[n - 1], label=r"$\log(p^\sigma_t/\bar q)$")
  ax = axes[n]
  ax.plot(r.sigmas, r.info / r.info[0], lw=2.0, label=r"clock information $I_\sigma/I_0$")
  ax.plot(r.sigmas, r.mmc / r.mmc[0], "--", lw=2.0, label=r"minimal cost $M_\sigma/M_0$")
  for sigma in r.display_sigmas:
    ax.axvline(sigma, color="0.7", lw=0.8)
  ax.set(xlabel=r"clock blur $\sigma$", ylabel="fraction retained", ylim=(0.0, 1.05), title="Retained under blur")
  ax.legend(frameon=False, fontsize=9)
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
