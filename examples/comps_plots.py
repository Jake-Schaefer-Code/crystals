"""Plotting helpers for illustrating free particles, central potentials, and bands."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Iterable, Sequence
import matplotlib.pyplot as plt
import numpy as onp
from matplotlib.patches import Circle, Polygon
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.axes import Axes


try:
  from examples.lattice_plots import (
    make_diatomic_dispersion_figure,
    make_discrete_laplacian_figure,
    make_gap_opening_animation,
    make_monatomic_dispersion_figure,
    make_shift_operator_animation,
  )
  from examples.per_sym import (
    finite_triangular_patch,
    first_bz_hexagon_vertices,
    gaussian,
    make_k_mesh,
  )
except ImportError:
  from lattice_plots import (  # type: ignore[no-redef]
    make_diatomic_dispersion_figure,
    make_discrete_laplacian_figure,
    make_gap_opening_animation,
    make_monatomic_dispersion_figure,
    make_shift_operator_animation,
  )
  from per_sym import (  # type: ignore[no-redef]
    finite_triangular_patch,
    first_bz_hexagon_vertices,
    gaussian,
    make_k_mesh,
  )
from plotting_utils import (
  PlotStyle, apply_matplotlib_style, _merge_kwargs, _clean_axes, _panel_text,
  inline_latex, resolve_outdir, save,
)

import physics as phys
NDArray = onp.ndarray
from src.lattice_geometry import make_triangular_lattice
from src.symmetry import recip_lattice

# ---------------------------
# Configuration
# ---------------------------

_Ek_prop: str = r"E(\mathbf{k})\propto |\mathbf{k}|^2"
_Ek_eq: str = r"E(\mathbf{k})=\frac{\hbar^2|\mathbf{k}|^2}{2m}"

STYLE = PlotStyle()

# Presets reused across several central-potential figures
SOFT_COULOMB_DEFAULTS = {"depth": 1.65, "softening": 0.40}
DEFAULT_RADIAL_GRID = {"r_min": 0.18, "r_max": 8.0, "n_points": 1500}
SOFT_BOUND_LEVELS = onp.array([-1.00, -0.43, -0.21, -0.12, -0.075])
SOFT_BOUND_LABELS = [r"$1s$", r"$2s/2p$", r"$3\ell$", r"$4\ell$", r"$5\ell$"]


def _unit_vector(vec: Sequence[float], eps: float = 1e-12) -> NDArray:
  arr = onp.asarray(vec, dtype=float)
  norm = float(onp.linalg.norm(arr))
  if norm < eps:
    raise ValueError("Expected a nonzero vector.")
  return arr / norm


def _cross2(a: Sequence[float], b: Sequence[float]) -> float:
  ax, ay = onp.asarray(a, dtype=float)
  bx, by = onp.asarray(b, dtype=float)
  return float(ax * by - ay * bx)


def _ray_segment_intersection(
  origin: Sequence[float],
  direction: Sequence[float],
  p0: Sequence[float],
  p1: Sequence[float],
  *,
  eps: float = 1e-10,
) -> tuple[float, float] | None:
  origin = onp.asarray(origin, dtype=float)
  direction = onp.asarray(direction, dtype=float)
  p0 = onp.asarray(p0, dtype=float)
  p1 = onp.asarray(p1, dtype=float)

  segment = p1 - p0
  denom = _cross2(direction, segment)
  if abs(denom) < eps:
    return None

  delta = p0 - origin
  t_ray = _cross2(delta, segment) / denom
  t_seg = _cross2(delta, direction) / denom
  if t_ray < eps or t_seg < -eps or t_seg > 1.0 + eps:
    return None
  return t_ray, t_seg


def _first_polygon_hit(
  origin: Sequence[float],
  direction: Sequence[float],
  polygon: NDArray,
  *,
  exclude_edge: int | None = None,
  eps: float = 1e-10,
) -> tuple[int, NDArray] | None:
  origin = onp.asarray(origin, dtype=float)
  direction = onp.asarray(direction, dtype=float)
  best_hit: tuple[float, int] | None = None

  nverts = len(polygon)
  for edge_idx in range(nverts):
    if edge_idx == exclude_edge:
      continue
    p0 = polygon[edge_idx]
    p1 = polygon[(edge_idx + 1) % nverts]
    hit = _ray_segment_intersection(origin, direction, p0, p1, eps=eps)
    if hit is None:
      continue
    t_ray, _ = hit
    if best_hit is None or t_ray < best_hit[0]:
      best_hit = (t_ray, edge_idx)

  if best_hit is None:
    return None

  t_ray, edge_idx = best_hit
  return edge_idx, origin + t_ray * direction


def _edge_outward_normal(p0: Sequence[float], p1: Sequence[float]) -> NDArray:
  edge = onp.asarray(p1, dtype=float) - onp.asarray(p0, dtype=float)
  return _unit_vector([edge[1], -edge[0]])


def _refract_direction(
  incident: Sequence[float],
  normal_12: Sequence[float],
  n1: float,
  n2: float,
  *,
  eps: float = 1e-12,
) -> NDArray | None:
  """
  Refract a unit incident ray across an interface.

  `normal_12` must point from medium 1 into medium 2.
  """
  incident = _unit_vector(incident)
  normal_12 = _unit_vector(normal_12)

  tangential = incident - onp.dot(incident, normal_12) * normal_12
  transmitted_tangent = (n1 / n2) * tangential
  tangent_sq = float(onp.dot(transmitted_tangent, transmitted_tangent))
  if tangent_sq > 1.0 + eps:
    return None

  normal_mag = onp.sqrt(max(0.0, 1.0 - tangent_sq))
  return _unit_vector(transmitted_tangent + normal_mag * normal_12)


def _equilateral_prism_vertices(
  side: float,
  *,
  center: tuple[float, float] = (0.0, 0.0),
) -> NDArray:
  height = 0.5 * onp.sqrt(3.0) * side
  cx, cy = center
  return onp.array([
    [cx - 0.5 * side, cy - height / 3.0],
    [cx + 0.5 * side, cy - height / 3.0],
    [cx, cy + 2.0 * height / 3.0],
  ], dtype=float)


def _wavelength_to_rgb(wavelength_nm: float, gamma: float = 0.8) -> tuple[float, float, float]:
  wl = float(onp.clip(wavelength_nm, 380.0, 780.0))
  if wl < 440.0:
    r, g, b = -(wl - 440.0) / 60.0, 0.0, 1.0
  elif wl < 490.0:
    r, g, b = 0.0, (wl - 440.0) / 50.0, 1.0
  elif wl < 510.0:
    r, g, b = 0.0, 1.0, -(wl - 510.0) / 20.0
  elif wl < 580.0:
    r, g, b = (wl - 510.0) / 70.0, 1.0, 0.0
  elif wl < 645.0:
    r, g, b = 1.0, -(wl - 645.0) / 65.0, 0.0
  else:
    r, g, b = 1.0, 0.0, 0.0

  if wl < 420.0:
    scale = 0.3 + 0.7 * (wl - 380.0) / 40.0
  elif wl < 701.0:
    scale = 1.0
  else:
    scale = 0.3 + 0.7 * (780.0 - wl) / 79.0

  rgb = scale * onp.array([r, g, b], dtype=float)
  return tuple(onp.power(onp.clip(rgb, 0.0, 1.0), gamma))


def make_prism_dispersion_figure(
  *,
  prism_side: float = 2.7,
  beam_height: float = 0.08,
  beam_start_x: float = -5.5,
  exit_length: float = 4.8,
  wavelengths_nm: Sequence[float] = (700.0, 650.0, 610.0, 580.0, 540.0, 500.0, 460.0, 430.0),
  air_index: float = 1.0,
  cauchy_A: float = 1.08,
  cauchy_B: float = 0.0080,
  background: str = "transparent",
  prism_color: str = "#cfcfd6",
  prism_fill_alpha: float = 0.0,
  beam_color: str | None = None,
  figsize: tuple[float, float] = (12.0, 4.8),
) -> plt.Figure:
  """
  Plot a stylized prism-dispersion figure with simple 2D geometric optics.

  The refractive index uses a Cauchy-law fit
  n(lambda) = A + B / lambda^2 with lambda measured in micrometers.
  """
  prism = _equilateral_prism_vertices(prism_side)
  source = onp.array([beam_start_x, beam_height], dtype=float)
  incident = onp.array([1.0, 0.0], dtype=float)

  entry_hit = _first_polygon_hit(source, incident, prism)
  if entry_hit is None:
    raise ValueError("Incoming ray does not intersect the prism.")

  entry_edge, entry_point = entry_hit
  entry_p0 = prism[entry_edge]
  entry_p1 = prism[(entry_edge + 1) % len(prism)]
  entry_normal = -_edge_outward_normal(entry_p0, entry_p1)

  inside_segments: list[tuple[NDArray, NDArray, tuple[float, float, float]]] = []
  outside_segments: list[tuple[NDArray, NDArray, tuple[float, float, float]]] = []

  for wavelength_nm in wavelengths_nm:
    wavelength_um = wavelength_nm / 1000.0
    glass_index = cauchy_A + cauchy_B / (wavelength_um**2)
    inside_dir = _refract_direction(incident, entry_normal, air_index, glass_index)
    if inside_dir is None:
      continue

    inside_origin = entry_point + 1e-6 * inside_dir
    exit_hit = _first_polygon_hit(
      inside_origin,
      inside_dir,
      prism,
      exclude_edge=entry_edge,
    )
    if exit_hit is None:
      continue

    exit_edge, exit_point = exit_hit
    exit_p0 = prism[exit_edge]
    exit_p1 = prism[(exit_edge + 1) % len(prism)]
    exit_normal = _edge_outward_normal(exit_p0, exit_p1)
    exit_dir = _refract_direction(inside_dir, exit_normal, glass_index, air_index)
    if exit_dir is None:
      continue

    color = _wavelength_to_rgb(wavelength_nm)
    inside_segments.append((entry_point.copy(), exit_point.copy(), color))
    outside_segments.append((exit_point.copy(), exit_point + exit_length * exit_dir, color))

  fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

  if background == "transparent":
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
  elif background == "white":
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
  elif background == "black":
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
  else:
    raise ValueError("background must be 'transparent', 'white', or 'black'.")

  if beam_color is None:
    beam_color = "white" if background == "black" else "#111111"

  ax.add_patch(
    Polygon(
      prism,
      closed=True,
      facecolor=prism_color,
      edgecolor="none",
      alpha=prism_fill_alpha,
      zorder=0,
    )
  )

  ax.plot(
    [source[0], entry_point[0]],
    [source[1], entry_point[1]],
    color=beam_color,
    lw=2.8,
    solid_capstyle="round",
  )

  for start, stop, color in inside_segments:
    ax.plot(
      [start[0], stop[0]],
      [start[1], stop[1]],
      color=color,
      lw=1.5,
      alpha=0.35,
      solid_capstyle="round",
    )

  if inside_segments:
    mid_start, mid_stop, _ = inside_segments[len(inside_segments) // 2]
    ax.plot(
      [mid_start[0], mid_stop[0]],
      [mid_start[1], mid_stop[1]],
      color=beam_color,
      lw=1.4,
      alpha=0.9,
      solid_capstyle="round",
    )

  for start, stop, color in outside_segments:
    ax.plot(
      [start[0], stop[0]],
      [start[1], stop[1]],
      color=color,
      lw=2.9,
      solid_capstyle="round",
    )

  prism_outline = onp.vstack([prism, prism[0]])
  ax.plot(
    prism_outline[:, 0],
    prism_outline[:, 1],
    color=prism_color,
    lw=2.2,
    solid_joinstyle="round",
  )

  points = [source, entry_point, *prism, *(stop for _, stop, _ in outside_segments)]
  coords = onp.vstack(points)
  pad = 0.45
  ax.set_xlim(coords[:, 0].min() - pad, coords[:, 0].max() + pad)
  ax.set_ylim(coords[:, 1].min() - pad, coords[:, 1].max() + pad)
  ax.set_aspect("equal")
  ax.axis("off")
  return fig


def _draw_crosshair(
  ax: Axes,
  *,
  color: str = "black",
  lw: float = 1.0,
  alpha: float = 0.35,
) -> None:
  ax.axhline(0.0, lw=lw, color=color, alpha=alpha)
  ax.axvline(0.0, lw=lw, color=color, alpha=alpha)


def _draw_energy_circles(
  ax: Axes,
  radii: Sequence[float],
  *,
  color: str | None = None,
  lw: float = 2.0,
  alpha: float = 1.0,
  n_points: int = 500,
  as_patches: bool = False,
) -> None:
  if as_patches:
    patch_kwargs: dict[str, Any] = {"fill": False, "lw": lw, "alpha": alpha}
    if color is not None:
      patch_kwargs["color"] = color
    for radius in radii:
      ax.add_patch(Circle((0.0, 0.0), radius, **patch_kwargs))
    return

  theta = onp.linspace(0.0, 2 * onp.pi, n_points)
  line_kwargs: dict[str, Any] = {"lw": lw, "alpha": alpha}
  if color is not None:
    line_kwargs["color"] = color
  for radius in radii:
    ax.plot(radius * onp.cos(theta), radius * onp.sin(theta), **line_kwargs)


def _draw_radial_arrows(
  ax: Axes,
  *,
  radius: float,
  angles: Sequence[float] | None = None,
  arrowprops: dict[str, Any] | None = None,
) -> None:
  angles = onp.linspace(0, 2 * onp.pi, 8, endpoint=False) if angles is None else angles
  props = _merge_kwargs({"arrowstyle": "->", "lw": 1.1, "alpha": 0.75}, arrowprops)
  for theta in angles:
    ax.annotate(
      "",
      xy=(radius * onp.cos(theta), radius * onp.sin(theta)),
      xytext=(0.0, 0.0),
      arrowprops=props,
    )

def soft_coulomb_panel_defaults() -> dict[str, Any]:
  """Reusable defaults for the soft-Coulomb bound/continuum panel."""
  return {
    "bound_levels": SOFT_BOUND_LEVELS,
    "bound_labels": SOFT_BOUND_LABELS,
    "level_kwargs": {
      "start": 0.18,
      "text_offset": 0.12,
      "linewidth": 2.4,
      "fallback_extent": 1.2,
    },
    "potential_kwargs": {"lw": 2.5, "color": "black", "label": r"$V(r)$"},
    "continuum_span": (0.0, 0.7),
    "show_zero_line": True,
    "zero_line_kwargs": {"lw": 1.8, "ls": "--", "color": "0.4"},
    "continuum_text": "scattering continuum\n$E>0$",
    "bound_region_text": "discrete bound states\n$E<0$",
    "xlim": (0.18, 8.0),
    "ylim": (-1.9, 0.7),
  }


def soft_coulomb_profile(
  *,
  r_min: float = DEFAULT_RADIAL_GRID["r_min"],
  r_max: float = DEFAULT_RADIAL_GRID["r_max"],
  n_points: int = DEFAULT_RADIAL_GRID["n_points"],
  depth: float = SOFT_COULOMB_DEFAULTS["depth"],
  softening: float = SOFT_COULOMB_DEFAULTS["softening"],
  clip: tuple[float, float] | None = None,
) -> tuple[NDArray, NDArray]:
  """
  Generate a softened Coulomb profile with consistent defaults.

  Note: softening is the length scale inside sqrt(r^2 + softening^2), not the
  squared offset that sometimes appears in closed-form sketches.
  """
  r = onp.linspace(r_min, r_max, n_points)
  V = phys.coulomb_like_potential(r, depth=depth, softening=softening)
  if clip is not None:
    V = onp.clip(V, clip[0], clip[1])
  return r, V

def plot_soft_coulomb_panel(
  ax,
  *,
  r: NDArray | None = None,
  V: NDArray | None = None,
  depth: float = SOFT_COULOMB_DEFAULTS["depth"],
  softening: float = SOFT_COULOMB_DEFAULTS["softening"],
  radial_grid: dict[str, float] | None = None,
  clip: tuple[float, float] | None = None,
  panel_kwargs: dict[str, Any] | None = None,
) -> tuple[NDArray, NDArray]:
  """
  Plot a soft-Coulomb central potential with consistent bound/continuum styling.

  Optionally accepts precomputed (r, V) arrays; otherwise they are generated
  from the provided depth/softening/grid parameters.
  """
  if (r is None) != (V is None):
    raise ValueError("Provide both r and V, or neither.")

  if r is None or V is None:
    grid = _merge_kwargs(DEFAULT_RADIAL_GRID, radial_grid)
    r, V = soft_coulomb_profile(
      r_min=grid["r_min"],
      r_max=grid["r_max"],
      n_points=int(grid["n_points"]),
      depth=depth,
      softening=softening,
      clip=clip,
    )

  kwargs = soft_coulomb_panel_defaults()
  if panel_kwargs:
    for key, value in panel_kwargs.items():
      if key.endswith("_kwargs") and isinstance(value, dict) and isinstance(kwargs.get(key), dict):
        kwargs[key] = _merge_kwargs(kwargs[key], value)
      else:
        kwargs[key] = value

  plot_central_potential_panel(ax, r=r, V=V, **kwargs)
  return r, V

# ---------------------------
# Shared geometry helpers
# ---------------------------


def plot_energy_ring(ax: Axes3D, *, radius: float | None = None,
  energy: float | None = None,
  prefactor: float = 0.5,
  offset: float = 0.02,
  n_points: int = 400,
  **plot_kwargs: Any,
) -> tuple[float, float]:
  """
  Draw a constant-energy circle on a 3D paraboloid. Either radius or energy is required.
  Returns (radius, energy) used for the ring.
  """
  if radius is None and energy is None:
    raise ValueError("Provide either radius or energy for a constant-energy ring.")
  if radius is None:
    assert energy is not None
    radius = onp.sqrt(energy / prefactor)
  if energy is None:
    assert radius is not None
    energy = prefactor * radius**2

  theta = onp.linspace(0, 2 * onp.pi, n_points)
  z = onp.full_like(theta, energy + offset)
  ring_style = _merge_kwargs({"lw": STYLE.lw_main}, plot_kwargs)
  ax.plot(radius * onp.cos(theta), radius * onp.sin(theta), z, **ring_style)
  return radius, energy


def plot_paraboloid(
  ax: Axes3D,
  *,
  kmax: float,
  ngrid: int,
  prefactor: float = 0.5,
  surface_kwargs: dict[str, Any] | None = None,
  ring_energy: float | None = None,
  ring_radius: float | None = None,
  ring_offset: float = 0.02,
  ring_kwargs: dict[str, Any] | None = None,
) -> NDArray:
  """
  Render a free-particle paraboloid and optionally overlay one constant-energy ring.
  Returns the energy grid for further use.
  """
  KX, KY = make_k_mesh(kmax, ngrid)
  E = phys.free_particle_energy(KX, KY, prefactor=prefactor)

  kwargs = _merge_kwargs({"rstride": 4, "cstride": 4, "linewidth": 0, "antialiased": True, "alpha": 0.9}, surface_kwargs)
  ax.plot_surface(KX, KY, E, **kwargs)

  if ring_energy is not None or ring_radius is not None:
    plot_energy_ring(
      ax,
      energy=ring_energy,
      radius=ring_radius,
      prefactor=prefactor,
      offset=ring_offset,
      **(ring_kwargs or {}),
    )

  return E


def plot_free_paraboloid_panel(
  ax: Axes3D,
  *,
  kmax: float,
  ngrid: int,
  prefactor: float = 0.5,
  ring_energy: float | None = None,
  ring_radius: float | None = None,
  ring_offset: float = 0.02,
  ring_radii: Sequence[float] | None = None,
  surface_kwargs: dict[str, Any] | None = None,
  ring_kwargs: dict[str, Any] | None = None,
  title: str | None = None,
  labels: tuple[str, str, str] = (r"$k_x$", r"$k_y$", r"$E$"),
  view: tuple[float, float] | None = (28, -55),
  text_lines: Sequence[str] | None = None,
  text_loc: tuple[float, float] = (0.03, 0.95),
) -> NDArray:
  """
  Higher-level wrapper around plot_paraboloid with consistent labeling and view.
  """
  E = plot_paraboloid(
    ax,
    kmax=kmax,
    ngrid=ngrid,
    prefactor=prefactor,
    surface_kwargs=surface_kwargs,
    ring_energy=ring_energy,
    ring_radius=ring_radius,
    ring_offset=ring_offset,
    ring_kwargs=ring_kwargs,
  )

  if ring_radii is not None:
    for r in ring_radii:
      plot_energy_ring(ax, radius=r, prefactor=prefactor, offset=0.0, **(ring_kwargs or {}))

  ax.set(xlabel=labels[0], ylabel=labels[1], zlabel=labels[2])
  if title is not None:
    ax.set_title(title)

  if text_lines:
    ax.text2D(text_loc[0], text_loc[1], "\n".join(text_lines), transform=ax.transAxes, va="top")

  if view is not None:
    ax.view_init(elev=view[0], azim=view[1])

  return E


def _level_extent(r: NDArray, V: NDArray, level: float, fallback: float) -> float:
  idx = int(onp.argmax(V > level))
  return r[idx] if idx > 0 else fallback


def draw_bound_levels(
  ax,
  r: NDArray,
  V: NDArray,
  levels: Sequence[float],
  labels: Sequence[str] | None = None,
  *,
  start: float = 0.18,
  text_offset: float = 0.12,
  linewidth: float | None = None,
  label_size: int = 12,
  fallback_extent: float = 1.2,
  extent: float | None = None,
) -> None:
  """Consistently render discrete bound levels against a central potential."""
  lw = linewidth or STYLE.lw_main
  for i, level in enumerate(levels):
    x_right = extent if extent is not None else _level_extent(r, V, level, fallback=fallback_extent)
    ax.hlines(level, start, x_right, lw=lw)
    if labels is not None and i < len(labels):
      ax.text(x_right + text_offset, level, labels[i], va="center", fontsize=label_size)


def plot_central_potential_panel(
  ax: Axes,
  *,
  r: NDArray,
  V: NDArray,
  bound_levels = None,
  bound_labels = None,
  level_kwargs: dict[str, Any] | None = None,
  potential_kwargs = None,
  show_zero_line: bool = True,
  zero_line_kwargs: dict[str, Any] | None = None,
  continuum_span: tuple[float, float] | None = None,
  continuum_kwargs: dict[str, Any] | None = None,
  continuum_text: str | None = None,
  continuum_text_xy: tuple[float, float] = (0.60, 0.92),
  bound_region_text: str | None = None,
  bound_region_xy: tuple[float, float] = (0.60, 0.16),
  xlim: tuple[float, float] | None = None,
  ylim: tuple[float, float] | None = None,
  xticks: Sequence[float] | Sequence[str] | None = None,
  yticks: Sequence[float] | Sequence[str] | None = None,
  hide_spines: Iterable[str] = ("top", "right"),
) -> None:
  """
  Draw a central potential with optional bound levels and a shaded continuum.

  This keeps annotation choices consistent across the several figures that
  juxtapose bound states with scattering continua.
  """
  pot_kwargs = _merge_kwargs({"lw": STYLE.lw_main, "color": "black"}, potential_kwargs)
  ax.plot(r, V, **pot_kwargs)

  if bound_levels is not None:
    draw_bound_levels(ax, r, V, bound_levels, bound_labels, **(level_kwargs or {}))

  if show_zero_line:
    zero_kwargs = _merge_kwargs({"lw": 1.7, "ls": "--", "color": "0.45"}, zero_line_kwargs)
    ax.axhline(0.0, **zero_kwargs)

  if continuum_span is not None:
    span_kwargs = _merge_kwargs({"alpha": 0.12}, continuum_kwargs)
    ax.axhspan(*continuum_span, **span_kwargs)

  if continuum_text:
    ax.text(continuum_text_xy[0], continuum_text_xy[1], continuum_text, transform=ax.transAxes, ha="left", va="top")

  if bound_region_text:
    ax.text(bound_region_xy[0], bound_region_xy[1], bound_region_text, transform=ax.transAxes, ha="left", va="bottom")

  _clean_axes(ax, xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks, hide_spines=hide_spines)


def make_free_particle_vs_atom_figure():
  """Contrast free-particle k-space contours with discrete atomic bound levels."""
  fig = plt.figure(figsize=(13.5, 6.5))
  gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.15], wspace=0.22)

  ax_free = fig.add_subplot(gs[0, 0])
  ax_atom = fig.add_subplot(gs[0, 1])

  # ============================================================
  # Left panel: free particle in free space
  # ============================================================
  # Show constant-energy contours in k-space. For a free particle,
  # E(k) depends only on |k|, so equal-energy sets are circles in 2D.
  kmax = 3.2
  _clean_axes(ax_free, xlim=(-kmax, kmax), ylim=(-kmax, kmax), aspect="equal", xticks=[], yticks=[])

  _draw_energy_circles(ax_free, [0.8, 1.5, 2.25], lw=2.2, as_patches=True)
  _draw_radial_arrows(ax_free, radius=2.55)
  _draw_crosshair(ax_free)

  _panel_text(ax_free, "Free particle", fontsize=15)
  _panel_text(ax_free, "Continuous translation + rotation symmetry", xy=(0.03, 0.88))
  _panel_text(ax_free, inline_latex(_Ek_prop), xy=(0.03, 0.79))
  _panel_text(ax_free, "Equal-energy sets are circles in 2D\n( spheres in 3D )", xy=(0.03, 0.70))
  _panel_text(ax_free, "Continuous spectrum / continuum of labels", xy=(0.03, 0.11), align="bottom")

  ax_free.set(xlabel=r"$k_x$", ylabel=r"$k_y$", xticks=[], yticks=[], title="Symmetry-organized continuum")

  # ============================================================
  # Right panel: spherically symmetric atom
  # ============================================================
  # Show a radial potential well and discrete bound levels.
  r, V = soft_coulomb_profile(r_min=0.0, r_max=6.0, n_points=1200, depth=3.2, softening=0.22)
  levels = [-2.2, -0.95, -0.45, -0.20]
  labels = [r"$E_1$", r"$E_2$", r"$E_3$", r"$E_4$"]
  plot_central_potential_panel(
    ax_atom,
    r=r,
    V=V,
    bound_levels=levels,
    bound_labels=labels,
    level_kwargs={
      "start": 0.65,
      "extent": 4.0,
      "text_offset": 0.15,
      "linewidth": 2.2,
      "label_size": 13,
      "fallback_extent": 4.0,
    },
    potential_kwargs={"lw": 2.5},
    show_zero_line=True,
    zero_line_kwargs={"lw": 1.0, "color": "black", "alpha": 0.35, "ls": "-"},
    continuum_span=None,
    xlim=(-0.1, 6.2),
    ylim=(-3.7, 0.45),
    xticks=[],
    yticks=[],
  )
  ax_atom.axvline(0, lw=1.0, color="black", alpha=0.35)

  # Nucleus marker
  ax_atom.scatter([0], [V[0]], s=90, zorder=5)
  ax_atom.text(0.12, V[0] + 0.15, "nucleus", fontsize=12)

  _panel_text(ax_atom, "Spherically symmetric atom", fontsize=15)
  _panel_text(ax_atom, "Central potential: translation broken,\nrotation preserved", xy=(0.03, 0.88))
  _panel_text(ax_atom, "Bound states become discrete", xy=(0.03, 0.72))
  _panel_text(ax_atom, "Degeneracies now organized by rotational symmetry", xy=(0.03, 0.63))

  ax_atom.set(xlabel=r"radius $r$", ylabel=r"$V(r)$ / energy", title="Central potential and discrete levels",)

  fig.suptitle(
    "From free-space symmetry to atomic symmetry:\ncontinuous labels vs discrete bound states", y=0.98,
  )
  fig.subplots_adjust(top=0.82)

  return fig





def make_free_vs_atom_figure():
  """Compare a free-particle dispersion with a central attractive potential."""
  fig = plt.figure(figsize=(13.5, 6.2), constrained_layout=True)
  gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0])

  # -----------------------------
  # Left: free particle in k-space
  # -----------------------------
  ax1 = fig.add_subplot(gs[0, 0])
  kmax = 3.2
  KX, KY = make_k_mesh(kmax, 500)
  E = phys.free_particle_energy(KX, KY, prefactor=0.18)

  im = ax1.imshow(E, extent=(-kmax, kmax, -kmax, kmax), origin="lower", cmap="viridis", aspect="equal")
  _draw_energy_circles(ax1, [0.9, 1.6, 2.3, 2.9], lw=1.6, color="white", alpha=0.95)
  _draw_crosshair(ax1, color="white")
  _clean_axes(ax1, xlim=(-kmax, kmax), ylim=(-kmax, kmax), aspect="equal", xticks=[], yticks=[])
  _panel_text(ax1, "Pree Farticle\n" + inline_latex(_Ek_eq), fontsize=15, color="white")
  _panel_text(ax1,
    "continuous translation symmetry\n"
    r"$\Rightarrow$ momentum label $\mathbf{k}$ is continuous" "\n"
    r"$\Rightarrow$ energies fill a continuum",
    xy=(0.03, 0.08),
    align="bottom",
    color="white",
  )

  ax1.set(xlabel=r"$k_x$", ylabel=r"$k_y$", title="Continuous spectrum from unconfined motion")

  cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.03)
  cbar.set_label("Energy")

  # -----------------------------------------
  # Right: central attractive potential + levels
  # -----------------------------------------
  ax2 = fig.add_subplot(gs[0, 1])

  r = onp.linspace(0.15, 8.0, 2000)
  V = -1.0 / r
  V_plot = onp.clip(V, -3.2, 1.2)  # avoid singular spike near r=0 in the drawing

  bound_levels = -1.0 / (2.0 * onp.array([1, 2, 3, 4, 5])**2)
  labels: list[str] = [rf"$E_{n}$" for n in range(1, len(bound_levels) + 1)]

  plot_central_potential_panel(
    ax2,
    r=r,
    V=V_plot,
    bound_levels=bound_levels,
    bound_labels=labels,
    level_kwargs={
      "start": 0.55,
      "extent": 3.6,
      "text_offset": 0.18,
      "linewidth": 2.4,
      "label_size": 13,
      "fallback_extent": 3.6,
    },
    potential_kwargs={"lw": 2.5, "color": "black"},
    continuum_span=(0.0, 1.15),
    continuum_kwargs={"color": "0.85", "alpha": 0.8},
    show_zero_line=True,
    zero_line_kwargs={"lw": 1.8, "ls": "dashed", "color": "0.4"},
    xlim=(0.15, 8.0),
    ylim=(-0.65, 1.15),
  )

  ax2.text(
    5.2, 0.55,
    "continuous\nscattering spectrum",
    ha="center", va="center", fontsize=14
  )
  ax2.text(7.95, 0.03, r"$E=0$", ha="right", va="bottom", color="0.35", fontsize=13)

  ax2.text(
    0.03, 0.96,
    "Spherically symmetric atom\n"
    r"$V(r)\sim -1/r$",
    transform=ax2.transAxes,
    ha="left", va="top", fontsize=15
  )
  ax2.text(
    0.03, 0.08,
    "Translation sym. broken,\nrotation sym. survives\n"
    r"$E<0$: discrete bound states" "\n"
    r"$E\geq 0$: continuous scattering states",
    transform=ax2.transAxes,
    ha="left", va="bottom"
  )

  ax2.set(
    xlabel=r"radius $r$",
    ylabel="Energy",
    title="Central potential: discrete bound spectrum + continuum",
  )

  fig.suptitle(
    "Free particle vs spherically symmetric atom:\n"
    "continuous momentum-labeled spectrum versus discrete bound levels",
    fontsize=18
  )

  return fig

def plot_free_particle_paraboloid():
  """3D paraboloid for the free-particle dispersion surface."""
  fig = plt.figure(figsize=(8, 6.5), constrained_layout=True)
  ax = fig.add_subplot(111, projection="3d")
  assert isinstance(ax, Axes3D)

  plot_free_paraboloid_panel(
    ax,
    kmax=2.8,
    ngrid=180,
    prefactor=0.22,
    surface_kwargs={"rstride": 2, "cstride": 2, "linewidth": 0, "antialiased": True, "alpha": 0.95},
    ring_radii=[0.9, 1.6, 2.3],
    ring_kwargs={"color": "black"},
    ring_offset=0.0,
    title=fr"Free particle: {inline_latex(_Ek_prop)}",
    labels=(r"$k_x$", r"$k_y$", "Energy"),
    view=(28, -52),
  )

  return fig



def make_free_vs_central_potential_figure(
  kmax: float = 2.4,
  ngrid: int = 300,
):
  r"""Free-particle surface beside a central potential with discrete levels."""
  fig = plt.figure(figsize=(13.5, 6.5), constrained_layout=True)
  gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0])

  # =========================
  # Left panel: free particle
  # =========================
  ax1 = fig.add_subplot(gs[0, 0], projection="3d")
  assert isinstance(ax1, Axes3D)

  plot_free_paraboloid_panel(
    ax1,
    kmax=kmax,
    ngrid=ngrid,
    prefactor=0.5,
    ring_energy=1.2,
    ring_offset=0.02,
    ring_kwargs={"lw": 3},
    title="Free particle",
    labels=(r"$k_x$", r"$k_y$", r"$E(k_x,k_y)$"),
    text_lines=[
      inline_latex(_Ek_eq),
      "equal-energy sets are circles in $k$-space",
    ],
    view=(28, -54),
  )

  # =========================================
  # Right panel: central potential + spectrum
  # =========================================
  ax2 = fig.add_subplot(gs[0, 1])

  plot_soft_coulomb_panel(ax2, depth=1.7, softening=onp.sqrt(0.18),
    panel_kwargs={
      "level_kwargs": {"text_offset": 0.15},
      "continuum_text": "scattering continuum\n$E>0$",
      "bound_region_text": "discrete bound states\n$E<0$",
    },
  )
  ax2.text(7.7, 0.03, r"$E=0$", ha="right", va="bottom", color="0.35")

  ax2.set(title="Spherically symmetric attractive potential", xlabel=r"$r$", ylabel="Energy")

  fig.suptitle(
    "From free-space continuum to discrete bound levels:\n"
    "translation sym. broken, rotational sym. retained"
  )

  return fig



def make_three_panel_symmetry_spectrum_figure():
  """Free particle vs atom vs crystal (bands) in a single three-panel figure."""
  fig = plt.figure(figsize=(16, 5.8), constrained_layout=True)
  gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.15])

  # ============================================================
  # Panel 1: Free particle
  # ============================================================
  ax1 = fig.add_subplot(gs[0, 0], projection="3d")
  assert isinstance(ax1, Axes3D)

  plot_free_paraboloid_panel(
    ax1,
    kmax=2.3,
    ngrid=220,
    prefactor=0.5,
    surface_kwargs={"alpha": 0.92},
    ring_energy=1.15,
    ring_kwargs={"lw": 3},
    title="Free particle",
    labels=(r"$k_x$", r"$k_y$", r"$E$"),
    text_lines=[
      inline_latex(_Ek_eq),
      "continuous momentum labels",
      "full translation + rotation symmetry",
    ],
    text_loc=(0.02, 0.96),
    view=(28, -55),
  )

  # ============================================================
  # Panel 2: Spherically symmetric attractive potential
  # ============================================================
  ax2 = fig.add_subplot(gs[0, 1])

  plot_soft_coulomb_panel(
    ax2,
    panel_kwargs={
      "continuum_span": (0.0, 0.68),
      "zero_line_kwargs": {"lw": 1.8, "ls": "--", "color": "0.45"},
      "bound_region_text": "discrete bound levels\n$E<0$",
    },
  )
  ax2.text(7.85, 0.03, r"$E=0$", ha="right", va="bottom", color="0.35")

  ax2.set(title="Central attractive potential", xlabel=r"$r$", ylabel="Energy")

  # ============================================================
  # Panel 3: Periodic potential + bands
  # ============================================================
  subgs = gs[0, 2].subgridspec(2, 1, height_ratios=[1.0, 1.05], hspace=0.02)
  ax3a = fig.add_subplot(subgs[0, 0])
  ax3b = fig.add_subplot(subgs[1, 0])

  # Top: periodic potential in real space
  a = 1.0
  n_cells = 8
  x = onp.linspace(-0.5 * a, (n_cells - 0.5) * a, 2600)
  site_positions = onp.arange(n_cells) * a

  Vlat = onp.zeros_like(x)
  for xs in site_positions:
    Vlat -= 1.15 * gaussian(x, xs, 0.16)
  Vlat += 0.07 * onp.cos(2 * onp.pi * x / a)

  ax3a.plot(x, Vlat, lw=2.4)
  ax3a.scatter(site_positions, onp.interp(site_positions, x, Vlat), s=24, zorder=5)

  for xs in site_positions:
    ax3a.axvline(xs, lw=0.8, alpha=0.18)

  _clean_axes(ax3a, xlim=(x.min(), x.max()), xticks=[], yticks=[])
  ax3a.set(title="Periodic lattice potential")
  ax3a.text(0.02, 0.93, r"$V(x+a)=V(x)$" "\ndiscrete translation symmetry", transform=ax3a.transAxes, ha="left", va="top")

  for spine in ["top", "right", "left"]:
    ax3a.spines[spine].set_visible(False)

  # Bottom: schematic band structure
  k = onp.linspace(-onp.pi, onp.pi, 600)

  # Three schematic bands
  band1 = -1.25 + 0.55 * (1 - onp.cos(k))
  band2 =  0.25 + 0.45 * (1 + onp.cos(k))
  band3 =  1.55 + 0.35 * (1 - onp.cos(k))

  ax3b.plot(k, band1, lw=2.5)
  ax3b.plot(k, band2, lw=2.5)
  ax3b.plot(k, band3, lw=2.5)

  # Shade band gaps
  ax3b.axhspan(band1.max(), band2.min(), alpha=0.10)
  ax3b.axhspan(band2.max(), band3.min(), alpha=0.10)

  # High-symmetry / zone-edge guides
  for kval in [-onp.pi, 0, onp.pi]:
    ax3b.axvline(kval, lw=0.9, alpha=0.2)

  _clean_axes(ax3b, xlim=(-onp.pi, onp.pi), xticks=[-onp.pi, 0, onp.pi], hide_spines=("top", "right"))
  ax3b.set(xlabel=r"$k$", ylabel="Energy", xticklabels=[r"$-\pi/a$", r"$0$", r"$\pi/a$"],)
  ax3b.text(0.02, 0.92, "Bloch bands + gaps", transform=ax3b.transAxes, ha="left", va="top")

  for spine in ["top", "right"]:
    ax3b.spines[spine].set_visible(False)

  fig.suptitle("From free-space continuum to atomic bound states to crystal bands", y=1.02)

  return fig


def make_mnt_plots(kmax: float = 2.4):
  """Produce a small bundle of figures for talks/notes in one shot."""
  E0 = 1.25
  r, V = soft_coulomb_profile()
  created = []
  outdir = resolve_outdir()


  # 1) Free-particle paraboloid
  fig = plt.figure(figsize=(8.5, 6.5), constrained_layout=True)
  ax = fig.add_subplot(111, projection="3d")
  assert isinstance(ax, Axes3D)
  plot_free_paraboloid_panel(
    ax,
    kmax=kmax,
    ngrid=240,
    prefactor=0.5,
    surface_kwargs={"alpha": 0.92},
    ring_energy=E0,
    ring_kwargs={"lw": 3},
    title="Free-particle dispersion",
    text_lines=[
      inline_latex(_Ek_eq),
      "Fixing $E$ fixes only the radius in $k$-space.",
    ],
    view=(28, -55),
  )
  created.extend(save(fig, "01_free_particle_paraboloid", outdir=outdir))

  # 2) Constant-energy circle
  fig, ax = plt.subplots(figsize=(7.5, 7.0), constrained_layout=True)
  R = 1.9
  res = 500
  th = 2 * onp.pi * onp.linspace(0, 1, res)
  ax.plot(R * onp.cos(th), R * onp.sin(th), lw=2.5)
  angles = 2 * onp.pi * onp.linspace(0, 1, 12, endpoint=False)
  for ang in angles:
    x = R * onp.cos(ang)
    y = R * onp.sin(ang)
    ax.arrow(0, 0, 0.92 * x, 0.92 * y, width=0.012, head_width=0.10, head_length=0.14,
            length_includes_head=True, alpha=0.85)
    ax.scatter([x], [y], s=28, zorder=5)
  lbl = r"One fixed energy $E_0$\n$\implies$ one fixed radius $|\mathbf{k}|=k_0$\nbut infinitely many directions."
  ax.text(0.03, 0.96, lbl,
      transform=ax.transAxes, va="top")
  ax.set(title="Degeneracy at fixed free-particle energy", xlabel=r"$k_x$", ylabel=r"$k_y$", 
         aspect="equal", xlim=(-2.4, 2.4), ylim=(-2.4, 2.4))
  ax.grid(alpha=0.22)
  created.extend(save(fig, "02_constant_energy_circle", outdir=outdir))

  # 3) Plane-wave patterns for different directions
  fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.8), constrained_layout=True, squeeze=False)

  x = 4.0 * onp.linspace(-1, 1, res)
  y = 4.0 * onp.linspace(-1, 1, res)
  X, Y = onp.meshgrid(x, y)
  angles = [0.0, onp.pi/4, onp.pi/2.7]
  titles = [r"$\theta=0$", r"$\theta=\pi/4$", r"$\theta\approx 1.16$"]
  for ax, ang, title in zip(axs.flat, angles, titles):
    kx0, ky0 = onp.cos(ang), onp.sin(ang)
    Z = onp.cos(2.6 * (kx0 * X + ky0 * Y))
    ax.imshow(Z, extent=[x.min(), x.max(), y.min(), y.max()], origin="lower", cmap="RdBu_r")
    ax.set(title=title, xticks=[], yticks=[])
  fig.suptitle("Directions on E shell give plane-wave patterns")
  created.extend(save(fig, "03_plane_waves_different_directions", outdir=outdir))

  panel_kwargs = {
    "level_kwargs": {"linewidth": 2.5},
    "potential_kwargs": {"lw": 2.5, "color": "black"},
    "continuum_span": (0.0, 0.65),
    "zero_line_kwargs": {"lw": 1.7, "ls": "--", "color": "0.45"},
    "continuum_text": "scattering continuum\n$E>0$",
    "bound_region_text": "discrete bound levels\n$E<0$",
    "ylim": (-1.85, 0.65),
  }

  # 4) Central potential with discrete/continuum split
  fig, ax = plt.subplots(figsize=(7.8, 5.8), constrained_layout=True)
  plot_soft_coulomb_panel(
    ax,
    r=r,
    V=V,
    panel_kwargs=panel_kwargs,
  )
  ax.text(0.03, 0.96, "Translation broken,\nrotation retained.", transform=ax.transAxes, va="top")
  ax.set(title="Central attractive potential", xlabel=r"$r$", ylabel="Energy")
  created.extend(save(fig, "04_central_potential_spectrum", outdir=outdir))

  # 5) Compact comparison slide
  fig = plt.figure(figsize=(12.5, 5.3), constrained_layout=True)
  gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0])
  ax1 = fig.add_subplot(gs[0, 0], projection="3d")
  assert isinstance(ax1, Axes3D)
  plot_free_paraboloid_panel(
    ax1,
    kmax=kmax,
    ngrid=240,
    prefactor=0.5,
    surface_kwargs={"alpha": 0.92},
    ring_energy=E0,
    ring_kwargs={"lw": 3},
    title="Free particle",
    text_lines=["continuous labels", "continuous energies"],
    view=(28, -55),
  )
  panel_kwargs.update({
    "continuum_text": "continuum",
    "bound_region_text": "discrete levels",
  })
  ax2 = fig.add_subplot(gs[0, 1])
  plot_soft_coulomb_panel(
    ax2,
    r=r,
    V=V,
    panel_kwargs=panel_kwargs,
  )
  ax2.set(title="Central potential", xlabel=r"$r$", ylabel="Energy")
  fig.suptitle("Symmetry labels states; binding discretizes energies")
  created.extend(save(fig, "05_free_vs_central_compact", outdir=outdir))

  print("\n".join(created))
  return created

# ---------------------------
# Batch generation
# ---------------------------

def generate_all_assets(out_dir: str | Path = "presentation_assets", style: PlotStyle=PlotStyle()) -> dict[str, Path]:
  apply_matplotlib_style(style)
  out_dir = resolve_outdir(out_dir)

  outputs: dict[str, Path] = {}
  outputs["shift_operator"] = make_shift_operator_animation(out_dir)
  outputs["discrete_laplacian"] = make_discrete_laplacian_figure(out_dir)
  outputs["monatomic_dispersion"] = make_monatomic_dispersion_figure(out_dir)
  outputs["diatomic_dispersion"] = make_diatomic_dispersion_figure(out_dir)
  outputs["gap_opening"] = make_gap_opening_animation(out_dir)
  outputs["mnt_plots"] = make_mnt_plots(out_dir)
  return outputs


if __name__ == "__main__":
  outputs = generate_all_assets("presentation_assets", STYLE)
  for name, path in outputs.items():
    print(f"{name:>20s} -> {path}")
