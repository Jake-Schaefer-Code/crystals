"""Plotting helpers for 1D lattice animations and chain dispersions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec

try:
  from examples.per_sym import (
    _2π,
    diatomic_dispersion,
    folded_free_electron_branches_1d,
    free_electron_energy_1d,
    monatomic_dispersion,
    nearly_free_electron_reduced_bands_1d,
  )
except ImportError:
  from per_sym import (  # type: ignore[no-redef]
    _2π,
    diatomic_dispersion,
    folded_free_electron_branches_1d,
    free_electron_energy_1d,
    monatomic_dispersion,
    nearly_free_electron_reduced_bands_1d,
  )
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


def _style_bz_axis(ax: Axes, *, a: float = 1.0, ylabel: str | None = None) -> None:
  kb = np.pi / a
  ax.set_xlim(-kb, kb)
  ax.set_xticks([-kb, 0.0, kb], [r"$-\pi/a$", r"$0$", r"$\pi/a$"])
  ax.axvline(-kb, linestyle="--", linewidth=STYLE.lw_aux, alpha=0.45)
  ax.axvline(kb, linestyle="--", linewidth=STYLE.lw_aux, alpha=0.45)
  ax.axvline(0.0, linestyle="--", linewidth=STYLE.lw_aux, alpha=0.25)
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)
  ax.grid(False)
  if ylabel is not None:
    ax.set_ylabel(ylabel)


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


def plot_mono_vs_diatomic(
  *,
  a: float = 1.0,
  K: float = 1.0,
  M_mono: float = 1.0,
  m1: float = 1.0,
  m2: float = 2.4,
  n_k: int = 1200,
):
  """Side-by-side monatomic and diatomic 1D dispersions."""
  kb = np.pi / a
  k = np.linspace(-kb, kb, n_k)
  ω_mono = monatomic_dispersion(k, K=K, m=M_mono, a=a)
  ω_acoustic, ω_optical = diatomic_dispersion(k, K=K, m1=m1, m2=m2, a=a)

  fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True, constrained_layout=True)

  ax = axes[0]
  ax.plot(k, ω_mono)
  _style_bz_axis(ax, a=a, ylabel=r"$\omega(k)$")
  ax.set(title="Monatomic chain", xlabel=r"$k$")
  ax.annotate(
    "one degree of freedom\nper unit cell",
    xy=(0.52 * kb, 1.05 * float(np.max(ω_mono))),
    xytext=(0.18 * kb, 1.55 * float(np.max(ω_mono))),
    arrowprops=dict(arrowstyle="->", lw=1.5),
  )

  ax = axes[1]
  ax.plot(k, ω_acoustic, label="acoustic")
  ax.plot(k, ω_optical, label="optical")
  _style_bz_axis(ax, a=a)
  ax.set(title="Diatomic chain", xlabel=r"$k$")
  idx = int(np.argmin(np.abs(k - kb)))
  gap_low = ω_acoustic[idx]
  gap_high = ω_optical[idx]
  ax.vlines(0.92 * kb, gap_low, gap_high, linestyles="dashed")
  ax.text(0.53 * kb, 0.5 * (gap_low + gap_high), "gap", va="center")
  ax.annotate(
    "two degrees of freedom\nper unit cell",
    xy=(0.0, ω_optical[len(k) // 2]),
    xytext=(-0.78 * kb, 0.82 * float(np.max(ω_optical))),
    arrowprops=dict(arrowstyle="->", lw=1.5),
  )
  ax.legend(frameon=False, loc="upper center")

  fig.suptitle("More structure per cell changes the spectrum", y=1.02)
  return fig


def plot_folded_parabola_gap(
  *,
  a: float = 1.0,
  hbar: float = 1.0,
  m: float = 1.0,
  V_G: float = 0.4,
  V_2G: float = 0.0,
  n_k: int = 1200,
  n_full: int = 3000,
  n_shells: int = 2,
  n_bands: int = 4,
):
  """
  Show free parabola copies, reduced-zone folding, and multi-band gap opening.
  """
  kb = np.pi / a
  G = _2π / a
  k_full = np.linspace(-3.2 * kb, 3.2 * kb, n_full)
  E0 = free_electron_energy_1d(k_full, hbar=hbar, m=m)
  Em = free_electron_energy_1d(k_full - G, hbar=hbar, m=m)
  Ep = free_electron_energy_1d(k_full + G, hbar=hbar, m=m)

  k_bz = np.linspace(-kb, kb, n_k)
  _, folded = folded_free_electron_branches_1d(
    k_bz, a=a, hbar=hbar, m=m, n_shells=n_shells,
  )
  _, reduced_bands, _ = nearly_free_electron_reduced_bands_1d(
    k_bz, V_G=V_G, V_2G=V_2G, a=a, hbar=hbar, m=m, n_shells=n_shells,
  )
  n_show = min(n_bands, reduced_bands.shape[0])

  fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True, constrained_layout=True)

  ax: Axes = axes[0]
  ax.plot(k_full, E0, color='k')
  ax.plot(k_full, Em, alpha=0.6, color='k')
  ax.plot(k_full, Ep, alpha=0.6, color='k')
  ax.axvline(-kb, ls="--", lw=1.5, alpha=0.25, color="k")
  ax.axvline(kb, ls="--", lw=1.5, alpha=0.25, color="k")
  ax.set(
    xlim=(-3.2 * kb, 3.2 * kb),
    ylim=(0.0, 1.05 * float(np.max(E0[np.abs(k_full) <= 3.2 * kb]))),
    title="Free-electron parabola",
    xlabel=r"$k$",
    ylabel=r"$E(k)$",
    xticks=[-2 * kb, -kb, 0.0, kb, 2 * kb],
    xticklabels=[r"$-2\pi/a$", r"$-\pi/a$", r"$0$", r"$\pi/a$", r"$2\pi/a$"],
  )
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)

  ax = axes[1]
  for band in folded[:n_show]:
    ax.plot(k_bz, band, color="0.35", linestyle="--", linewidth=1.4)
  _style_bz_axis(ax, a=a)
  ax.set(title="Fold free branches into the first Brillouin zone", xlabel=r"$k$")

  ax = axes[2]
  for band in reduced_bands[:n_show]:
    ax.plot(k_bz, band, color="k", linewidth=2.0)
  _style_bz_axis(ax, a=a)
  ax.set(title="Periodic potential opens gaps and rounds the bands", xlabel=r"$k$")
  gap_low = reduced_bands[0, -1]
  gap_high = reduced_bands[1, -1]
  ax.vlines(0.92 * kb, gap_low, gap_high, linestyles="dashed")
  ax.text(0.53 * kb, 0.5 * (gap_low + gap_high), "gap", va="center")

  fig.suptitle("From free-electron dispersion to a zone-edge gap", y=1.02)
  return fig


def plot_nearly_free_electron_schemes(
  *,
  a: float = 1.0,
  hbar: float = 1.0,
  m: float = 1.0,
  V_G: float = 0.35,
  V_2G: float = 0.0,
  n_k: int = 1200,
  n_shells: int = 3,
  n_bands: int = 4,
  zone_count: int = 2,
):
  """
  Compare extended-zone and reduced-zone nearly-free-electron band plots,
  with separate zooms at Γ and at the first Brillouin-zone edge.
  """
  kb = np.pi / a
  G = _2π / a
  k_bz = np.linspace(-kb, kb, n_k)
  basis_indices, reduced_bands, eigvecs = nearly_free_electron_reduced_bands_1d(
    k_bz, V_G=V_G, V_2G=V_2G, a=a, hbar=hbar, m=m, n_shells=n_shells,
  )
  _, folded = folded_free_electron_branches_1d(
    k_bz, a=a, hbar=hbar, m=m, n_shells=n_shells,
  )
  n_show = min(n_bands, reduced_bands.shape[0])

  branch_points: dict[int, list[tuple[float, float]]] = {int(n): [] for n in basis_indices}
  weights = np.abs(eigvecs) ** 2
  dominant_components = np.argmax(weights, axis=1)
  for iq, q in enumerate(k_bz):
    for band in range(reduced_bands.shape[0]):
      dom_idx = int(basis_indices[dominant_components[iq, band]])
      k_ext = float(q + dom_idx * G)
      if abs(k_ext) <= (zone_count + 0.5) * G:
        branch_points[dom_idx].append((k_ext, float(reduced_bands[band, iq])))

  fig = plt.figure(figsize=(16.5, 8.6), constrained_layout=True)
  gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 0.9], width_ratios=[1.15, 1.0])
  ax_ext = fig.add_subplot(gs[0, 0])
  ax_red = fig.add_subplot(gs[0, 1])
  ax_gamma = fig.add_subplot(gs[1, 0])
  ax_edge = fig.add_subplot(gs[1, 1])

  ref_band = 0 if n_show == 1 else 1
  low_energy_max = float(np.max(reduced_bands[ref_band])) + 0.6 * abs(V_G)
  energy_window = (0.0, 1.04 * low_energy_max)
  gamma_idx = len(k_bz) // 2
  gamma_upper_idx = min(2, reduced_bands.shape[0] - 1)
  gamma_center_low = float(reduced_bands[1 if reduced_bands.shape[0] > 1 else 0, gamma_idx])
  gamma_center_high = float(reduced_bands[gamma_upper_idx, gamma_idx])
  gamma_pad = max(0.25, 0.8 * (gamma_center_high - gamma_center_low) + 0.15)
  gap_low = float(reduced_bands[0, -1])
  gap_high = float(reduced_bands[1, -1])
  gap_pad = max(0.25, 0.55 * (gap_high - gap_low))

  k_ext = np.linspace(-(zone_count + 0.5) * G, (zone_count + 0.5) * G, (2 * zone_count + 1) * n_k)
  for n in range(-(zone_count + 1), zone_count + 2):
    ax_ext.plot(
      k_ext,
      free_electron_energy_1d(k_ext - n * G, hbar=hbar, m=m),
      linestyle="--",
      linewidth=1.2,
      alpha=0.45,
      color="0.35",
    )
  for n, pts in branch_points.items():
    if not pts:
      continue
    pts_arr = np.asarray(sorted(pts, key=lambda item: item[0]))
    ax_ext.plot(pts_arr[:, 0], pts_arr[:, 1], linewidth=2.0, color="k")
  for n in range(zone_count + 1):
    boundary = (n + 0.5) * G
    ax_ext.axvline(boundary, linestyle=":", linewidth=1.0, color="0.6")
    ax_ext.axvline(-boundary, linestyle=":", linewidth=1.0, color="0.6")
  ax_ext.set(
    title="Extended-zone scheme",
    xlabel=r"$k$",
    ylabel="energy",
    xlim=(k_ext[0], k_ext[-1]),
    ylim=energy_window,
    xticks=[n * G for n in range(-zone_count, zone_count + 1)],
    xticklabels=[
      "0" if n == 0 else rf"${2*n}\pi/a$"
      for n in range(-zone_count, zone_count + 1)
    ],
  )
  ax_ext.spines["top"].set_visible(False)
  ax_ext.spines["right"].set_visible(False)

  for band in folded[:n_show]:
    ax_red.plot(k_bz, band, linestyle="--", linewidth=1.2, alpha=0.45, color="0.35")
  for band in reduced_bands[:n_show]:
    ax_red.plot(k_bz, band, linewidth=2.0, color="k")
  _style_bz_axis(ax_red, a=a)
  ax_red.set(title="Reduced-zone scheme", xlabel=r"$k$", ylim=energy_window)

  gamma_bands = [1] if reduced_bands.shape[0] == 2 else [1, gamma_upper_idx]
  for band in gamma_bands:
    ax_gamma.plot(k_bz, folded[band], linestyle="--", linewidth=1.2, alpha=0.45, color="0.35")
    ax_gamma.plot(k_bz, reduced_bands[band], linewidth=2.0, color="k")
  ax_gamma.axvline(0.0, linestyle=":", linewidth=1.0, color="0.6")
  ax_gamma.set(
    title="Γ-point zoom",
    xlabel=r"$k$",
    ylabel="energy",
    xlim=(-0.12 * kb, 0.12 * kb),
    ylim=(gamma_center_low - gamma_pad, gamma_center_high + gamma_pad),
    xticks=[0.0],
    xticklabels=[r"$0$"],
  )
  ax_gamma.spines["top"].set_visible(False)
  ax_gamma.spines["right"].set_visible(False)

  for band in folded[:2]:
    ax_edge.plot(k_bz, band, linestyle="--", linewidth=1.2, alpha=0.45, color="0.35")
  for band in reduced_bands[:2]:
    ax_edge.plot(k_bz, band, linewidth=2.2, color="k")
  ax_edge.axvline(kb, linestyle=":", linewidth=1.0, color="0.6")
  ax_edge.set(
    title="Zone-edge zoom",
    xlabel=r"$k$",
    xlim=(0.72 * kb, 1.02 * kb),
    ylim=(gap_low - gap_pad, gap_high + gap_pad),
    xticks=[kb],
    xticklabels=[r"$\pi/a$"],
  )
  ax_edge.spines["top"].set_visible(False)
  ax_edge.spines["right"].set_visible(False)
  ax_edge.vlines(0.93 * kb, gap_low, gap_high, linestyles="dashed")
  ax_edge.text(0.80 * kb, 0.5 * (gap_low + gap_high), rf"$\Delta E \approx {gap_high - gap_low:.2f}$", va="center")

  fig.suptitle(
    "Nearly-free electron bands: free parabolas versus periodic-potential band structure",
    y=1.02,
  )
  return fig


def plot_simple_band_structure():
  """Stylized crystal bands along the path Gamma -> X."""
  x = np.linspace(0.0, 1.0, 1000)
  band1 = 0.35 + 1.4 * np.sin(0.5 * np.pi * x) ** 2
  band2 = 2.7 + 0.8 * np.sin(np.pi * x) ** 2
  band3 = 4.6 + 0.9 * np.sin(0.7 * np.pi * x + 0.25) ** 2

  fig, ax = plt.subplots(figsize=(8.5, 6), constrained_layout=True)
  ax.plot(x, band1)
  ax.plot(x, band2)
  ax.plot(x, band3)

  gap_bottom = float(np.max(band1))
  gap_top = float(np.min(band2))
  ax.axhspan(gap_bottom, gap_top, alpha=0.12)
  ax.text(0.76, 0.5 * (gap_bottom + gap_top), "band gap", va="center")
  ax.set(
    xlim=(0.0, 1.0),
    ylim=(0.0, 6.2),
    title="Stylized crystal band structure",
    ylabel="Energy",
    xticks=[0.0, 1.0],
    xticklabels=[r"$\Gamma$", r"$X$"],
  )
  ax.annotate(
    "valence band",
    xy=(0.22, band1[220]),
    xytext=(0.06, 1.8),
    arrowprops=dict(arrowstyle="->", lw=1.5),
  )
  ax.annotate(
    "conduction band",
    xy=(0.52, band2[520]),
    xytext=(0.11, 3.85),
    arrowprops=dict(arrowstyle="->", lw=1.5),
  )
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)
  return fig
