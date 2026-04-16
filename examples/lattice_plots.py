"""Plotting helpers for 1D lattice animations and chain dispersions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec

from per_sym import diatomic_dispersion, monatomic_dispersion
from plotting_utils import (
  PlotStyle,
  inline_latex,
  normalize_to_unit_range,
  resolve_outdir,
  save_path,
)

NDArray = np.ndarray

STYLE = PlotStyle()
_2NBR_DIFF = r"u_{n+1} - 2u_n + u_{n-1}"


def _save_asset(fig, out_dir: str | Path, filename: str, *, tight_layout: bool = True) -> Path:
  return save_path(fig, Path(out_dir) / filename, tight_layout=tight_layout)


def make_shift_operator_animation(
  out_dir: str | Path,
  n_sites: int = 21,
  n_frames: int = 48,
  wavelength_sites: float = 7.0,
  filename: str = "shift_operator.gif",
) -> Path:
  """Animate a discrete Bloch-like mode and its shift by one lattice site."""
  out_dir = resolve_outdir(out_dir)
  x = np.arange(n_sites)
  center = (n_sites - 1) / 2.0
  envelope = np.exp(-0.5 * ((x - center) / (0.28 * n_sites))**2)
  k = 2.0 * np.pi / wavelength_sites

  fig, ax = plt.subplots(figsize=(10.5, 4.2))
  ax.set(
    xlim=(-0.5, n_sites - 0.5),
    ylim=(-1.4, 1.4),
    xlabel="lattice site $n$",
    ylabel="amplitude",
    title="Shift operator on a 1D lattice",
  )
  ax.scatter(x, np.zeros_like(x), s=STYLE.marker_size * 0.7, zorder=2)
  ax.axhline(0.0, linewidth=STYLE.lw_aux)

  original_line, = ax.plot([], [], label="mode $u_n$")
  shifted_line, = ax.plot([], [], linestyle="--", label="shifted mode $(Su)_n = u_{n-1}$")
  ghost_line, = ax.plot([], [], alpha=STYLE.alpha_aux, linewidth=STYLE.lw_aux)
  text = ax.text(
    0.02,
    0.96,
    "",
    transform=ax.transAxes,
    va="top",
    ha="left",
    bbox=dict(boxstyle="round,pad=0.25", alpha=0.12),
  )
  ax.legend(loc="upper right")

  def frame_data(frame: int) -> tuple[NDArray, NDArray, float]:
    phase = 2.0 * np.pi * frame / n_frames
    u = envelope * np.cos(k * x - phase)
    return u, np.roll(u, 1), phase

  def init():
    original_line.set_data([], [])
    shifted_line.set_data([], [])
    ghost_line.set_data([], [])
    text.set_text("")
    return original_line, shifted_line, ghost_line, text

  def update(frame: int):
    u, u_shift, phase = frame_data(frame)
    original_line.set_data(x, u)
    shifted_line.set_data(x, u_shift)
    ghost_line.set_data(x, u)
    text.set_text(
      "\n".join([
        r"$S$ shifts amplitudes by one site",
        r"Bloch mode: $u_n \sim e^{ikna}$",
        rf"frame phase $= {phase:.2f}$ rad",
      ])
    )
    return original_line, shifted_line, ghost_line, text

  anim = FuncAnimation(fig, update, init_func=init, frames=n_frames, interval=90, blit=True)
  out_path = Path(out_dir) / filename
  anim.save(out_path, writer=PillowWriter(fps=12))
  plt.close(fig)
  return out_path


def make_discrete_laplacian_figure(
  out_dir: str | Path,
  filename: str = "discrete_laplacian_local_difference.png",
) -> Path:
  """Illustrate the discrete second difference as local curvature."""
  out_dir = resolve_outdir(out_dir)
  x = np.arange(-4, 5)
  y = np.array([0.15, 0.35, 0.70, 1.00, 0.25, -0.20, -0.40, -0.48, -0.52])
  i0 = int(np.where(x == 0)[0][0])

  fig, ax = plt.subplots(figsize=(9.0, 4.8))
  ax.plot(x, y, marker="o")
  ax.axhline(0.0, linewidth=STYLE.lw_aux)
  ax.scatter([x[i0 - 1], x[i0], x[i0 + 1]], [y[i0 - 1], y[i0], y[i0 + 1]], zorder=5)
  ax.plot(
    [x[i0 - 1], x[i0 + 1]],
    [y[i0 - 1], y[i0 + 1]],
    linestyle="--",
    linewidth=STYLE.lw_aux,
    alpha=0.8,
  )

  avg_neighbors = 0.5 * (y[i0 - 1] + y[i0 + 1])
  ax.vlines(x[i0], avg_neighbors, y[i0], linewidth=2.4)
  ax.annotate(
    inline_latex(_2NBR_DIFF),
    xy=(x[i0], 0.5 * (avg_neighbors + y[i0])),
    xytext=(1.0, 1.05),
    textcoords="data",
    arrowprops=dict(arrowstyle="->", lw=1.2),
  )
  ax.set(
    title="Graph Laplacian measures local mismatch with neighbors",
    xlabel="site index $n$",
    ylabel="displacement $u_n$",
  )
  ax.text(
    0.02,
    0.95,
    inline_latex(fr"\Delta_d u_n = {_2NBR_DIFF}"),
    transform=ax.transAxes,
    va="top",
    ha="left",
    bbox=dict(boxstyle="round,pad=0.25", alpha=0.12),
  )
  return _save_asset(fig, out_dir, filename)


def make_monatomic_dispersion_figure(
  out_dir: str | Path,
  n_sites: int = 24,
  filename: str = "monatomic_dispersion_with_modes.png",
) -> Path:
  out_dir = resolve_outdir(out_dir)
  a = K = m = 1.0
  k_vals = np.linspace(-np.pi / a, np.pi / a, 600)
  w_vals = monatomic_dispersion(k_vals, K=K, m=m, a=a)
  sample_k = np.array([0.20 * np.pi, 0.55 * np.pi, 0.95 * np.pi])
  x = np.arange(n_sites)

  fig = plt.figure(figsize=(12.0, 6.2))
  gs = GridSpec(2, 3, figure=fig, height_ratios=[1.25, 1.0])
  ax_disp = fig.add_subplot(gs[0, :])
  ax_disp.plot(k_vals, w_vals, label="monatomic branch")
  ax_disp.set(
    title="Monatomic 1D chain: dispersion and representative mode shapes",
    xlabel=r"Bloch wavevector $k$",
    ylabel=r"$\omega(k)$",
    xlim=(k_vals[0], k_vals[-1]),
  )
  ax_disp.set_xticks([-np.pi, 0.0, np.pi], [r"$-\pi/a$", "0", r"$\pi/a$"])

  for ax, k0 in zip([fig.add_subplot(gs[1, i]) for i in range(3)], sample_k):
    w0 = monatomic_dispersion(np.array([k0]), K=K, m=m, a=a)[0]
    ax.plot(x, normalize_to_unit_range(np.cos(k0 * x)), marker="o")
    ax.axhline(0.0, linewidth=STYLE.lw_aux)
    ax.set(
      ylim=(-1.2, 1.2),
      xlabel="site $n$",
      ylabel="$u_n$",
      title=rf"$k={k0/np.pi:.2f}\pi/a$,  $\omega={w0:.2f}$",
    )
    ax_disp.scatter([k0], [w0], zorder=5)
    ax_disp.axvline(k0, linestyle="--", linewidth=STYLE.lw_aux, alpha=0.55)

  return _save_asset(fig, out_dir, filename)


def make_diatomic_dispersion_figure(
  out_dir: str | Path,
  filename: str = "diatomic_dispersion.png",
) -> Path:
  out_dir = resolve_outdir(out_dir)
  a = K = m1 = 1.0
  m2 = 2.0
  k_vals = np.linspace(-np.pi / a, np.pi / a, 600)
  w_ac, w_op = diatomic_dispersion(k_vals, K=K, m1=m1, m2=m2, a=a)

  fig = plt.figure(figsize=(12.0, 5.7))
  gs = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.6])
  ax_lat = fig.add_subplot(gs[0, 0])
  n_cells = 6
  xA = np.arange(n_cells) * 2.0
  xB = xA + 1.0

  for i in range(n_cells):
    ax_lat.plot([xA[i], xB[i]], [0, 0], linewidth=STYLE.lw_main)
    if i < n_cells - 1:
      ax_lat.plot([xB[i], xA[i + 1]], [0, 0], linewidth=STYLE.lw_main)

  ax_lat.scatter(xA, np.zeros_like(xA), s=85, label=r"$m_1$")
  ax_lat.scatter(xB, np.zeros_like(xB), s=140, label=r"$m_2$")
  ax_lat.set(
    title="Diatomic lattice: two masses per unit cell",
    xlabel="lattice position",
    yticks=[],
    xlim=(-0.8, xB[-1] + 0.8),
    ylim=(-1.0, 1.0),
  )
  ax_lat.grid(False)
  ax_lat.legend(loc="upper right")
  ax_lat.annotate("unit cell", xy=(0.5, 0.0), xytext=(1.7, 0.55), arrowprops=dict(arrowstyle="->", lw=1.2))

  ax_disp = fig.add_subplot(gs[0, 1])
  ax_disp.plot(k_vals, w_ac, label="acoustic branch")
  ax_disp.plot(k_vals, w_op, label="optical branch")
  ax_disp.set(
    title="Diatomic chain dispersion",
    xlabel=r"Bloch wavevector $k$",
    ylabel=r"$\omega(k)$",
    xlim=(k_vals[0], k_vals[-1]),
  )
  ax_disp.set_xticks([-np.pi, 0.0, np.pi], [r"$-\pi/a$", "0", r"$\pi/a$"])
  ax_disp.legend(loc="upper left")
  ax_disp.text(0.06, 0.10, "acoustic", transform=ax_disp.transAxes)
  ax_disp.text(0.06, 0.86, "optical", transform=ax_disp.transAxes)
  return _save_asset(fig, out_dir, filename)


def make_gap_opening_animation(
  out_dir: str | Path,
  n_frames: int = 56,
  filename: str = "gap_opening.gif",
) -> Path:
  """Animate folded free branches opening a gap at the zone boundary."""
  out_dir = resolve_outdir(out_dir)
  k = np.linspace(-np.pi, np.pi, 900)
  E1 = k**2
  E2 = (k - 2.0 * np.pi)**2

  fig, ax = plt.subplots(figsize=(10.2, 5.2))
  ax.set(
    xlim=(-np.pi, np.pi),
    ylim=(0.0, 16.0),
    xlabel=r"wavevector in first Brillouin zone",
    ylabel="energy",
    title="Zone folding and gap opening at the Brillouin-zone boundary",
  )
  ax.set_xticks([-np.pi, 0.0, np.pi], [r"$-\pi/a$", "0", r"$\pi/a$"])
  ax.axvline(-np.pi, linestyle="--", linewidth=STYLE.lw_aux, alpha=0.5)
  ax.axvline(np.pi, linestyle="--", linewidth=STYLE.lw_aux, alpha=0.5)

  free1, = ax.plot([], [], label="folded free branches")
  free2, = ax.plot([], [], alpha=STYLE.alpha_aux)
  band_lo, = ax.plot([], [], label="lower band")
  band_hi, = ax.plot([], [], label="upper band")
  text = ax.text(
    0.02,
    0.96,
    "",
    transform=ax.transAxes,
    va="top",
    ha="left",
    bbox=dict(boxstyle="round,pad=0.25", alpha=0.12),
  )
  ax.legend(loc="upper center")

  def coupled_bands(V: float) -> tuple[NDArray, NDArray]:
    trace = 0.5 * (E1 + E2)
    delta = 0.5 * (E1 - E2)
    split = np.sqrt(delta**2 + V**2)
    return trace - split, trace + split

  def init():
    free1.set_data([], [])
    free2.set_data([], [])
    band_lo.set_data([], [])
    band_hi.set_data([], [])
    text.set_text("")
    return free1, free2, band_lo, band_hi, text

  def update(frame: int):
    t = frame / max(n_frames - 1, 1)
    V = 1.8 * max(t - 0.20, 0.0) / 0.80
    lo, hi = coupled_bands(V)
    free1.set_data(k, E1)
    free2.set_data(k, E2)
    band_lo.set_data(k, lo)
    band_hi.set_data(k, hi)
    band_lo.set_alpha(max(0.0, (t - 0.10) / 0.90))
    band_hi.set_alpha(max(0.0, (t - 0.10) / 0.90))
    text.set_text(
      "\n".join([
        r"$k~k+G$ under periodicity",
        rf"coupling strength $|V_G| \approx {V:.2f}$",
        r"zone-boundary crossing splits into a gap",
      ])
    )
    return free1, free2, band_lo, band_hi, text

  anim = FuncAnimation(fig, update, init_func=init, frames=n_frames, interval=90, blit=True)
  out_path = Path(out_dir) / filename
  anim.save(out_path, writer=PillowWriter(fps=12))
  plt.close(fig)
  return out_path
