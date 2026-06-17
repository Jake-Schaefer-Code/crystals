# plotting_utils.py
from __future__ import annotations
import dataclasses as dcls
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as onp
from matplotlib.axes import Axes
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.mplot3d import Axes3D

from scipy.spatial import ConvexHull, Delaunay

from src import crystal_funcs as cfuncs
from src import coordinates as cconv
from src import planar_geometry as pgeom
import src.symmetry as sym

from src.core import NumericNDArray as NDArray



# ---- common defs
_2π = 2 * onp.pi
ACCENT = "#0f172a"
_cmap = plt.get_cmap('afmhot').copy()
_cmap.set_bad(color='whitesmoke')


# --- complex -> RGB helper: hue=phase, value=normalized magnitude
def complex_to_rgb(z, mag_gamma=0.8, eps=1e-12):
  phase = (onp.angle(z) + onp.pi) / _2π           # [0,1]
  mag = onp.abs(z)
  mag = mag / (mag.max() + eps)
  mag = mag ** mag_gamma
  hsv = onp.stack([phase, onp.ones_like(phase), mag], axis=-1)
  return hsv_to_rgb(hsv)


def subplots_list(nrows: int, ncols: int, **kwargs):
  fig, _axs = plt.subplots(nrows, ncols, **kwargs)
  axs: list[Axes] = onp.array(_axs).flatten().tolist()
  return fig, axs



# ---------------------------
# Configuration
# ---------------------------

@dcls.dataclass(frozen=True)
class PlotStyle:
  figure_dpi: int = 180
  save_dpi: int = 220
  lw_main: float = 2.2
  lw_aux: float = 1.3
  marker_size: float = 36.0
  alpha_aux: float = 0.35
  title_size: int = 15
  label_size: int = 12
  tick_size: int = 10
  legend_size: int = 10
  fig_title_size: int = 18
  fontset: str = "stix"
  font_family: str = "STIXGeneral"


def _merge_kwargs(defaults: dict | None, overrides: dict | None = None) -> dict:
  """Return a shallowly merged kwargs dict without mutating either input."""
  merged = dict(defaults or {})
  if overrides:
    merged.update(overrides)
  return merged


def inline_latex(x: str) -> str:
  """Wrap a plain LaTeX fragment in inline math delimiters."""
  text = x.strip()
  if text.startswith("$") and text.endswith("$"):
    return text
  return fr"${text}$"


def apply_matplotlib_style(style: PlotStyle = PlotStyle()) -> None:
  """Apply the shared presentation plotting defaults."""
  plt.rcParams.update({
    "figure.dpi": style.figure_dpi,
    "savefig.dpi": style.save_dpi,
    "font.size": style.label_size,
    "axes.titlesize": style.title_size,
    "axes.labelsize": style.label_size,
    "xtick.labelsize": style.tick_size,
    "ytick.labelsize": style.tick_size,
    "legend.fontsize": style.legend_size,
    "figure.titlesize": style.fig_title_size,
    "mathtext.fontset": style.fontset,
    "font.family": style.font_family,
    "lines.linewidth": style.lw_main,
  })


def _clean_axes(
  ax: Axes,
  *,
  xlim=None,
  ylim=None,
  aspect=None,
  xticks=None,
  yticks=None,
  hide_spines=("top", "right"),
  grid: bool | None = None,
) -> Axes:
  """Apply common axis cleanup used by presentation figures."""
  if xlim is not None:
    ax.set_xlim(xlim)
  if ylim is not None:
    ax.set_ylim(ylim)
  if aspect is not None:
    ax.set_aspect(aspect)
  if xticks is not None:
    ax.set_xticks(xticks)
  if yticks is not None:
    ax.set_yticks(yticks)
  if grid is not None:
    ax.grid(grid)
  for spine in hide_spines:
    if spine in ax.spines:
      ax.spines[spine].set_visible(False)
  return ax


def _panel_text(
  ax: Axes,
  text: str,
  *,
  xy: tuple[float, float] = (0.03, 0.95),
  align: str = "top",
  **kwargs,
):
  """Place small explanatory text in axes-relative coordinates."""
  defaults = {
    "ha": "left",
    "va": "bottom" if align == "bottom" else "top",
    "fontsize": kwargs.pop("fontsize", 12),
  }
  defaults.update(kwargs)
  return ax.text(xy[0], xy[1], text, transform=ax.transAxes, **defaults)


def resolve_outdir(outdir: str | Path | None = None) -> Path:
  """Resolve and create the default figure output directory."""
  path = Path(outdir or "media/figures")
  path.mkdir(parents=True, exist_ok=True)
  return path


def save_path(
  fig,
  path: str | Path,
  *,
  dpi: int | None = None,
  close: bool = True,
  tight_layout: bool = False,
  **savefig_kwargs,
) -> Path:
  """Save a figure to an explicit path, creating parent directories as needed."""
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  if tight_layout:
    fig.tight_layout()
  kwargs = {"bbox_inches": "tight"}
  kwargs.update(savefig_kwargs)
  fig.savefig(path, dpi=dpi or PlotStyle().save_dpi, **kwargs)
  if close:
    plt.close(fig)
  return path


def save(
  fig,
  name: str,
  *,
  outdir: str | Path | None = None,
  formats: tuple[str, ...] = ("png",),
  dpi: int | None = None,
  close: bool = True,
  **savefig_kwargs,
) -> list[str]:
  """Save a figure under one or more formats and return created paths."""
  out_path = resolve_outdir(outdir)
  stem = Path(name)
  if stem.suffix:
    paths = [out_path / stem.name]
  else:
    paths = [out_path / f"{name}.{fmt.lstrip('.')}" for fmt in formats]

  created = []
  for path in paths:
    save_path(fig, path, dpi=dpi, close=False, **savefig_kwargs)
    created.append(str(path))
  if close:
    plt.close(fig)
  return created


def normalize_to_unit_range(y, eps: float = 1e-12):
  """Scale an array so its max absolute value is 1, with an epsilon guard."""
  y = onp.asarray(y, dtype=float)
  m = onp.max(onp.abs(y))
  return y / max(m, eps)



def setup_2D_axes(ax:Axes, title=None, bounds=None):
  """
  Parameters
  ----------------
  """

  if title is not None:
    ax.set_title(title)
  if bounds is not None:
    ax.set(xlim=bounds, ylim=bounds)
  ax.set(aspect='equal', xlabel='X', ylabel='Y')
  ax.grid(False)

def setup_3D_axes(ax:Axes3D, view_angle=[30,30,0], title=None, bounds=None):
  """
  Parameters
  ----------------
  """
  ax.view_init(*view_angle)
  if title is not None:
    ax.set_title(title)
  if bounds is not None:
    ax.set(xlim=bounds, ylim=bounds, zlim=bounds)
  ax.set(aspect='equal', proj_type='persp', xlabel='X', ylabel='Y', zlabel='Z')
  ax.grid(False)


def plot_contours(X, Y, center, data, fund_unit, ax: Axes, **kwargs):
  """Plot scalar data over a translated 2D fundamental unit outline."""
  levels = kwargs.get("levels", 100)
  contour_kwargs = kwargs.get("contour_kwargs", {})
  patch_kwargs = {
    "fill": False,
    "edgecolor": kwargs.get("edgecolor", "k"),
  }
  patch_kwargs.update(kwargs.get("patch_kwargs", {}))

  cf = ax.contourf(X + center[0], Y + center[1], data, levels=levels, **contour_kwargs)
  patch = plt.Polygon(fund_unit, **patch_kwargs)
  ax.add_patch(patch)
  return cf, patch


def plot_density(coords, weights, fund_unit, ax: Axes, **kwargs):
  """Plot a weighted KDE over a 2D fundamental unit outline."""
  from mpl_toolkits.axes_grid1 import make_axes_locatable
  from scipy.stats import gaussian_kde

  coords = onp.asarray(coords)
  weights = onp.asarray(weights)
  grid_size = kwargs.get("grid_size", 100)
  bounds = kwargs.get("bounds", (0, 1))
  levels = kwargs.get("levels", 100)
  cmap = kwargs.get("cmap", _cmap)

  try:
    if coords.shape[0] <= coords.shape[1]:
      raise ValueError("not enough samples for a full-rank KDE")
    kde = gaussian_kde([*coords.T], weights=weights)
    X_grid, Y_grid = onp.meshgrid(
      onp.linspace(bounds[0], bounds[1], grid_size),
      onp.linspace(bounds[0], bounds[1], grid_size),
    )
    Z_grid = kde([X_grid.ravel(), Y_grid.ravel()]).reshape(X_grid.shape)
    density_artist = ax.contourf(X_grid, Y_grid, Z_grid, levels=levels, cmap=cmap)
  except ValueError:
    density_artist = ax.scatter(
      coords[:, 0],
      coords[:, 1],
      c=weights,
      cmap=cmap,
      s=kwargs.get("s", 20),
      alpha=kwargs.get("alpha", 0.8),
    )

  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  plt.colorbar(density_artist, cax=cax, orientation="vertical")
  patch = plt.Polygon(fund_unit, fill=False, edgecolor=kwargs.get("edgecolor", "k"))
  ax.add_patch(patch)
  return density_artist, patch


def plot_symmetry_op(
  X: onp.ndarray,
  Y: onp.ndarray,
  domain,
  group_data,
  fund_unit,
  dirichlet_pts,
  dirichlet_weights,
  n_samples,
  group_name,
  axes,
  *,
  rng=None,
):
  """Plot a symmetry field and an induced weighted sample density."""
  try:
    from sklearn.mixture import GaussianMixture
  except ModuleNotFoundError:
    GaussianMixture = None

  rng = onp.random.default_rng(rng)
  domain = onp.asarray(domain)
  fund_unit = onp.asarray(fund_unit)
  center = 0.5 * (domain[:, 1] - domain[:, 0])
  coords = onp.vstack([X.ravel(), Y.ravel()]).T + center
  cart_pts = cconv.barycentric_to_cartesian_2D(dirichlet_pts, fund_unit)

  inside_coords = coords[pgeom.isinside(coords, fund_unit)]
  weights = cfuncs.distance_weights(inside_coords, cart_pts, w=dirichlet_weights)
  probabilities = onp.asarray(group_data, dtype=float).ravel() ** 2
  probabilities /= onp.sum(probabilities)
  dist = rng.choice(len(coords), size=n_samples, p=probabilities)
  if GaussianMixture is None:
    samples = coords[dist]
  else:
    n_comp = 4 if group_name == "p4gm" else 1
    gmm = GaussianMixture(n_components=n_comp, max_iter=1000)
    gmm.fit(coords[dist])
    samples, _ = gmm.sample(n_samples)
  inside_samples = samples[pgeom.isinside(samples, fund_unit)]
  inside_weights = cfuncs.distance_weights(inside_coords, inside_samples)
  total_weights = weights * inside_weights
  total = onp.sum(total_weights)
  if total > 0:
    total_weights /= total
  filtered_coords = inside_coords[total_weights > 0]
  filtered_weights = total_weights[total_weights > 0]
  if filtered_coords.size == 0:
    filtered_coords = inside_coords
    filtered_weights = weights

  plot_contours(X, Y, center, group_data, fund_unit, axes[0])
  plot_density(filtered_coords, filtered_weights, fund_unit, axes[1])
  for ax in axes:
    setup_2D_axes(ax, title=group_name, bounds=(0, 0.99))


def plot_trimap_result(mapper, idx: int, **kwargs):
  """Convenience wrapper for plotting a stored ``TriMap`` candidate result."""
  return plot_result(
    mapper.maps[idx],
    mapper.matrices[idx],
    mapper.rotated_polygons[idx],
    mapper.triangulations[idx],
    mapper.distribution,
    mapper.distribution_weights,
    **kwargs,
  )


def plot_simplex(simplex, 
                ax: Axes3D, 
                simplices = None, 
                show_faces = False, 
                **kwargs):
  """
  Parameters
  ----------------
  simplex, 
  ax: Axes3D, 
  simplices = None, 
  show_faces = False, 
  **kwargs
  """

  data_kwargs = {
    "color": kwargs.get("color", "k"),
    "linestyle": kwargs.get("linestyle", "-"),
    "marker": kwargs.get("marker", "o"),
    "alpha": kwargs.get("alpha", 0.5),
    "ms":  kwargs.get("ms", 4),
  }

  face_kwargs = {
    "s": kwargs.get("s", 5),
    "c": kwargs.get("c", kwargs.get("color", "k")),
    "alpha": kwargs.get("face_alpha", 0.05),
  }
  
  if simplices is None:
      simplices = ConvexHull(simplex).simplices
  
  for face in simplices:
      vs = onp.array(list(simplex[face]) + [simplex[face][0]])
      ax.plot(*vs.T, **data_kwargs)

      # parametrizes each face on the simplex and plots that parametrization
      # effectively shades all the faces of the simplex for better visualization
      if show_faces:
          parametrization = cfuncs.parametrize_triangle(*simplex[face], num_points=100)
          ax.scatter(*parametrization.T, **face_kwargs)


def plot_weighted_dist(f_sym, 
                      coords, 
                      simplex, 
                      n_samples=10000,
                      **kwargs):
  """
  Parameters
  ----------------
  f_sym, 
  coords, 
  simplex, 
  n_samples=10000
  """
  dist, dist_weights = cfuncs.weighted_distribution(
      f_sym,
      coords,
      simplex,
      n_samples=n_samples,
      rng=kwargs.get("rng"),
  )
  dir_points = cconv.calculate_barycentric_coordinates(simplex, dist)
  dir_weights = cconv.barycentric_weights(dir_points)

  fig = plt.figure(figsize=(12,12))
  axes = []

  subplot_kw = {"projection": "3d"} if len(coords) == 3 else {}
  scatter_kw = {
      "alpha": kwargs.get("alpha", 0.1),
      "s": kwargs.get("s", 10),
      "cmap": kwargs.get("cmap", _cmap),
  }
  show_faces = False

  ax1:Axes3D = fig.add_subplot(211, **subplot_kw)
  ax1.set_title('Func on Fundamental Domain')
  scatter = ax1.scatter(*dist.T, c=dist_weights, **scatter_kw)
  fig.colorbar(scatter, ax=ax1)
  plot_simplex(simplex, ax1, show_faces=show_faces)

  ax2:Axes3D = fig.add_subplot(212, **subplot_kw)
  ax2.set_title('Func * Dirichlet on Fundamental Domain')
  scatter = ax2.scatter(*dist.T, c=dist_weights*dir_weights, **scatter_kw)
  fig.colorbar(scatter, ax=ax2)
  plot_simplex(simplex, ax2, show_faces=show_faces)

  for ax in[ax1, ax2]:
    ax.set(aspect='equal', xlim=(0, 1), ylim=(0, 1))
    if len(coords) == 3:
      assert isinstance(ax, Axes3D)
      ax.set(zlim=(0, 1), proj_type='persp')

  plt.gca().set_aspect('equal', adjustable='box')
  plt.show()


# TODO plot contours of 90% intervals?
def plot_func(func, *xi: NDArray, center: NDArray, sym_ops: list[list[sym.AffineOperation]]):
  r""" """
  # add the identity for just the plot of the original function
  ops = [[sym.identity(len(xi))]] + sym_ops
  num = len(ops)
  assert len(center) == len(xi)
  # Flatten grid coordinates for scatter; also translate to the plotted frame.
  coords = [(x + center[i]).ravel() for i, x in enumerate(xi)]
  
  kwargs = {}
  if len(xi) == 3:
    subplot_kw={"projection": "3d"}
    kwargs.update(subplot_kw)

  fig = plt.figure(figsize=(num * 4,8))
  axes = [fig.add_subplot(1, num, i+1, **kwargs) for i in range(num)]    
  for ax, op in zip(axes, ops):
    value = sym.symmetrize(func, *xi, sym_ops=op)
    value = sym._snap(value)
    scatter = ax.scatter(*coords, c=value.ravel(), cmap=_cmap, alpha=0.5)
    fig.colorbar(scatter, ax=ax)
    if isinstance(op, sym.FiniteGroupAction):
      title = op.name
    else:
      title = f"[{','.join(o.label for o in op)}]"
    # ax.set_proj_type('persp')
    ax.set(aspect='equal', title=title)
    # ax.set(xlim=(-1,1), ylim=(-1,1))
  plt.show()


def plot_result(
  best_map: NDArray, 
  best_mat: NDArray, 
  best_poly: NDArray, 
  best_tri: Delaunay, 
  dist, 
  dist_weights,
  **kwargs
):

  ndim = best_poly.shape[-1]
  show_faces = (ndim == 3)

  dist_kwargs = {
    "s":kwargs.get("s", 10),
    "alpha":kwargs.get("alpha", 0.5),
    "cmap":kwargs.get("cmap", _cmap),
    "edgecolors":kwargs.get("edgecolors", 'none'),
    "linewidths":kwargs.get("linewidths", 0),
  }
  
  quiver_kwargs = {
    "angles": 'xy',
    "scale_units": 'xy',
    "scale": 1,
    "alpha": 0.5
  } if ndim == 2 else {
    "length": 1,
    "arrow_length_ratio": 0.1
  }

  subplot_kw = {"projection": "3d"} if ndim == 3 else {}

  with plt.style.context('seaborn-v0_8-paper'):
    fig = plt.figure(figsize=(12,12))
    axes = [fig.add_subplot(1, 2, i+1, **subplot_kw) for i in range(2)]

    inv_mats = onp.array([onp.linalg.inv(M) for M in best_mat])
    dst_idxs = best_tri.find_simplex(dist)
    outside = dst_idxs < 0
    if onp.any(outside):
      raise ValueError(
          f"{outside.sum()} distribution points are outside the triangulation"
      )
    affine_map = onp.array([cfuncs.apply_affine_mat(pt, inv_mats[idx]) for pt, idx in zip(dist, dst_idxs)])

    # Plot original distribution
    axes[0].scatter(*dist.T, c=dist_weights, **dist_kwargs)
    plot_simplex(best_poly, axes[0], color='c', alpha=0.7, show_faces=show_faces)
    plot_simplex(best_map, axes[0], color='k')  

    end_pts = best_map-best_poly
    axes[0].quiver(*best_poly.T, *end_pts.T, **quiver_kwargs)

    # Plot Transformed distribution
    axes[1].scatter(*affine_map.T, c=dist_weights, **dist_kwargs)

    if ndim == 2:
      for t in best_tri.simplices:
        plot_simplex(best_poly[t], axes[1], alpha=0.1, edgecolor='k')
        plot_simplex(best_map[t], axes[0], alpha=0.1, color='r', linestyle='')   
    else:    
      plot_simplex(best_map, axes[0], simplices=ConvexHull(best_poly).simplices, color='r')    
      for tet in best_tri.simplices:
        faces = ConvexHull(best_map[tet]).simplices
        plot_simplex(best_poly, axes[0], simplices=faces, color='c', linestyle='')   
        plot_simplex(best_map, axes[0], simplices=faces, color='c', linestyle='')   

      for face in ConvexHull(best_poly).simplices:
        vs = onp.array(list(best_poly[face]) + [best_poly[face][0]])
        axes[1].plot(*vs.T, 'ko-')
        parametrization = cfuncs.parametrize_triangle(*best_poly[face], num_points=100)
        axes[1].scatter(*parametrization.T, c='k', s=10, alpha=0.025)
    

    for ax in axes:
      ax.set(aspect='equal', xlim=(-1.1, 1.1), ylim=(-1.1, 1.1))
      if ndim == 3:
        assert isinstance(ax, Axes3D)
        ax.set(zlim=(-1.1, 1.1), proj_type='persp')

    plt.show()
