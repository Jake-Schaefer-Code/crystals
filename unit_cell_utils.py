"""Shared geometry and plotting helpers for unit-cell sketches."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as onp
from numpy import cos, sin
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, Polygon,Circle
from matplotlib.text import TextPath
from matplotlib.transforms import Affine2D
import src.symmetry as sym

NDArray = onp.ndarray

_2π = 2 * onp.pi



def rhombus_unit_cell(origin: Sequence[float] = (0.0, 0.0), size: float = 1.0) -> NDArray:
  """Return a 60/120-degree rhombic unit cell."""
  origin = onp.asarray(origin, dtype=float)
  c, s = onp.cos(onp.pi / 3), onp.sin(onp.pi / 3)
  return onp.array([
    [0.0, 0.0],
    [c, s],
    [1.0 + c, s],
    [1.0, 0.0],
  ]) * size + origin



def add_polygon(
  ax: Axes,
  points: NDArray,
  *,
  closed: bool = True,
  edgecolor: str = "blue",
  facecolor="none",
  fill=None,
  linewidth: float = 2.0,
  **kwargs,
) -> Polygon:
  """Add a Matplotlib polygon patch and return it."""
  polygon = Polygon(
    points,
    closed=closed,
    edgecolor=edgecolor,
    facecolor=facecolor,
    fill=fill,
    linewidth=linewidth,
    **kwargs,
  )
  ax.add_patch(polygon)
  return polygon


def add_line(ax: Axes, p1: NDArray, p2: NDArray, **kwargs) -> Line2D:
  """Add a line between two points and return it."""
  line = Line2D(*zip(p1, p2), **kwargs)
  ax.add_line(line)
  return line


def add_text_path(
  ax: Axes,
  text: str,
  xy: Sequence[float],
  *,
  size: float = 0.2,
  rotation: float = 0.0,
  color: str = "black",
  **kwargs,
) -> PathPatch:
  """Add vector text transformed by rotation and translation."""
  path = TextPath((0, 0), text, size=size)
  transform = Affine2D().rotate(rotation).translate(*xy)
  patch = PathPatch(path.transformed(transform), color=color, **kwargs)
  ax.add_patch(patch)
  return patch


def triangular_grid_triangles(size: float, nx: int, ny: int) -> list[NDArray]:
  """Generate up/down equilateral triangles on the grid used by unit_cell.py."""
  triangles: list[NDArray] = []
  from src.lattice_geometry import equilateral_triangle_centered
  for dx in range(-nx, nx):
    for dy in range(-ny, ny):
      offset = (
        dx * onp.array([1.5 * size, 0.5 * onp.sqrt(3.0) * size])
        + dy * onp.array([0.0, onp.sqrt(3.0) * size])
      )
      tri = equilateral_triangle_centered(size, theta0=-5 * onp.pi / 6) + offset
      triangles.append(tri)

      v = tri[2] - tri[1]
      normal = onp.array([-v[1], v[0]])
      triangles.append(sym.reflection(normal).apply(tri))
  return triangles


def spiral_points(n: int) -> list[tuple[int, int]]:
  """Generate integer spiral coordinates for placing repeated unit cells."""
  x, y, dx, dy = 0, 0, 0, -1
  points: list[tuple[int, int]] = []
  for _ in range(n):
    if (-n / 2 < x <= n / 2) and (-n / 2 < y <= n / 2):
      points.append((x, y))
    if x == y or (x < 0 and x == -y) or (x > 0 and x == 1 - y):
      dx, dy = -dy, dx
    x, y = x + dx, y + dy
  return points


def dedupe_legend(ax: Axes) -> None:
  """Remove repeated legend labels from an axes."""
  handles, labels = ax.get_legend_handles_labels()
  by_label = dict(zip(labels, handles))
  ax.legend(by_label.values(), by_label.keys())


# ---- visuals

def plot_unit_cell(ax, origin, size):
  from src.lattice_geometry import equilateral_triangle_centered
  points = equilateral_triangle_centered(size, theta0=-5 * onp.pi / 6) + origin
  add_polygon(ax, points, edgecolor='blue', fill=None, linewidth=2)
  return points


def plot_asymmetric_units(ax, points, size):
  # Asymmetric unit, typically a sixth of the equilateral triangle
  centroid = onp.mean(points, axis=0)

  # Adding an "F" to the asymmetric unit
  add_text_path(ax, "F", centroid - onp.array([0.05, 0]), rotation=-onp.pi / 2)
  add_polygon(ax, points, edgecolor='red', fill=None, linestyle='--', linewidth=1.5)
  # Compute the centroid of the triangle for placing the "F"
  centroid = onp.mean(points, axis=0)
  add_text_path(ax, "F", centroid, rotation=-onp.pi / 3, lw=0.5)

def plot_all_asymmetric_units(ax, verts):
  centroid = onp.mean(verts, axis=0)
  centroid = onp.broadcast_to(centroid, verts.shape)

  for direction in (1, -1):
    # TODO is this ccw?
    midpts = 0.5 * (verts + onp.roll(verts, direction, axis=0))
    midpts2 = 0.5 * (verts + midpts)
    midpts3 = 0.5 * (verts + onp.roll(verts, -direction, axis=0))

    for p1, p2 in zip(midpts3, midpts2):
      ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="gray", linewidth=1, linestyle="--", zorder=-1,)

    # vertices, midpoints, and the centroid
    tris = onp.stack([verts, midpts, centroid], axis=1)

    for tri in tris:
      add_polygon(ax, tri, edgecolor="gray", fill=None, linewidth=2, linestyle="--")
      plot_F(ax, tri)

  edge_midpts = 0.5 * (verts + onp.roll(verts, -1, axis=0))
  add_polygon(ax, edge_midpts, edgecolor="gray", fill=None, linewidth=1, linestyle="--")


# thing to show orientation in unit cell
def make_F_in_tri(tri):
  x_values, y_values = [], []
  top = 0.3 * (tri[2] - tri[1])
  long = 0.3 * (- tri[0] + tri[1])
  ratio = onp.linalg.norm(long) / onp.linalg.norm(top)
  # base
  b = ratio * 0.5 * top + 0.5 * long + tri[1]

  l30 = 0.25 * long + b
  x_values = [
    [b[0], long[0] + b[0]],
    [b[0], top[0] + b[0]],
    [l30[0], 0.5 * top[0] + l30[0]],
  ]
  y_values = [
    [b[1], long[1] + b[1]],
    [b[1], top[1] + b[1]],
    [l30[1], 0.5 * top[1] + l30[1]],
  ]
  return x_values, y_values


def plot_F(ax, tri):
  x_values, y_values = make_F_in_tri(tri)
  for xv, yv in zip(x_values, y_values):
    ax.plot(xv, yv, color='k')




# ---- plotting Wyckoff positions
def plot_wyckoff_positions(ax: Axes, points, size, rot_order: int = 6):
  num = len(points)
  # Plotting Wyckoff positions 1a, 2c, and 3b
  ax.add_artist(Circle(points[0], 0.05, color='green', label='1a'))
  midpoints = [(points[i] + points[(i+1) % num]) / 2 for i in range(num)]
  for pt in midpoints:
    ax.add_artist(Circle(pt, 0.05, color='purple', label='2c'))
  center = onp.mean(points, axis=0)
  ax.scatter(*center, color='pink', s=20, marker='d')

  dθ = _2π / rot_order
  offsets = [onp.array([cos(dθ * i), sin(dθ * i)]) * size / 3 for i in range(rot_order)]
  for offset in offsets:
    ax.add_artist(Circle(center + offset, 0.05, color='red', label='6g'))
  # ax.add_artist(Circle(center, 0.05, color='orange', label='3b'))


def plot_wyckoff_positions(ax: Axes, points, size):
  # Plotting Wyckoff positions 1a, 2c, and 3b
  ax.add_artist(Circle(points[0], 0.05, color='green', label='1a'))
  for i in range(1, 3):
    ax.add_artist(Circle(points[i], 0.05, color='red', label='2c'))
  centroid = onp.mean(points, axis=0)
  ax.add_artist(Circle(centroid, 0.05, color='orange', label='3b'))



# ---- plotting mirror planes (TODO)
def plot_mirror_lines(ax: Axes, points, size):
  num = len(points)
  p1, p2, p3, p4 = points
  # Mirror lines in p6mm: along the axes and diagonals of the rhombus
  midpts = [(points[i] + points[(i+1) % num]) / 2 for i in range(num)]

  mirror_lines = [(midpts[0], midpts[2]), (midpts[1], midpts[3]), (p1, p3), (p2, p4),]
  for line in mirror_lines:
    add_line(ax, line[0], line[1], color="gray", linestyle='--')


def plot_mirror_lines(ax: Axes, points):
  num = len(points)
  # Points of the (triangle)
  A, B, C = points[0], points[1], points[2]
  # Centroid of the (triangle)
  centroid = onp.sum(points, axis=0) / num
  
  # Mirror lines from each vertex to the midpoint of the opposite side
  mirror_lines = [
    [A, (B + C) / 2],
    [B, (A + C) / 2],
    [C, (A + B) / 2]
  ]
  
  # Plotting each mirror line
  for line in mirror_lines:
    add_line(ax, line[0], line[1], color="k", linewidth=3, zorder=-1)
    add_line(ax, line[0], line[1], color="w", linewidth=1, zorder=-1)

  # Additional mirror lines from each vertex through the centroid
  for p in [A, B, C]:
    add_line(ax, p, centroid, color="k", linewidth=3, zorder=-1)
    add_line(ax, p, centroid, color="w", linewidth=1, zorder=-1)

  midpts = (points + onp.roll(points, 1, axis=0)) * 0.5
  for p in midpts:
    ax.scatter(p[0], p[1], marker="d", color='w', s=80,  ec='k', zorder=1)



def plot_cell(ax: Axes, tri):
  plot_all_asymmetric_units(ax, tri)
  add_polygon(ax, tri, edgecolor='k', fill=None, linewidth=3, facecolor='none')
  add_polygon(ax, tri, edgecolor='w', fill=None, linewidth=1, facecolor='none')
  plot_mirror_lines(ax, tri)
  centroid = onp.mean(tri, axis=0)
  ax.scatter(centroid[0], centroid[1], marker="^", color='w', s=80,  ec='k', zorder=1)
  ax.scatter(*tri.T, color='w', marker='H', s=80,  ec='k', zorder=1)


