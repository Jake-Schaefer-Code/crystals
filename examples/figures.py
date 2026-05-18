# figures.py
from __future__ import annotations

import dataclasses as dcls
from collections.abc import Callable
import matplotlib.pyplot as plt
import numpy as onp
from numpy import sin
import tyro
from matplotlib.axes import Axes
from matplotlib.animation import FuncAnimation
from matplotlib.patches import  Polygon

from unit_cell_utils import (
  add_polygon,
  add_text_path,
  dedupe_legend,
  rhombus_unit_cell,
  plot_cell,
  plot_asymmetric_units,
  plot_mirror_lines,
  plot_wyckoff_positions,
  plot_unit_cell,
)
from src.lattice_geometry import (
  translated_motif_cells,
  canonical_basis_2d,
  hexagon_centered,
  square_centered,
  triangle_pair_motif,
  tile_space,
  equilateral_triangle_centered,
  Bravais2D
)
import src.symmetry as sym
NDArray = onp.ndarray
# ---- common defs
_2π = 2 * onp.pi
# sqrt(3) = sin(60)


def unique_points(points, tol=1e-6):
  # Sort points primarily by the first column, and secondarily by the second column
  points = points[onp.lexsort((points[:,1], points[:,0]))]
  
  # Compute the differences between consecutive points
  delta = onp.diff(points, axis=0)
  dist = onp.linalg.norm(delta, axis=1)
  
  # Identify indices where points differ by more than the tolerance
  unique_indices = onp.where(dist > tol)[0] + 1
  
  # Always include the first point and append indices found
  unique_indices = onp.r_[0, unique_indices]
  
  return points[unique_indices]


def init_polygon_pool(
  ax: Axes,
  count: int,
  nverts: int,
  *,
  edgecolor: str = "blue",
  facecolor: str = "none",
  linewidth: float = 1.5,
) -> list[Polygon]:
  r""" INIT polygons """
  polys = [
    Polygon(
      onp.zeros((nverts, 2)),
      closed=True,
      edgecolor=edgecolor,
      facecolor=(onp.random.rand(3,) if facecolor == "random" else facecolor),
      linewidth=linewidth,
      visible=False,
    )
    for _ in range(count)
  ]
  for poly in polys:
    ax.add_patch(poly)
  return polys

def animate_tiling(
  ax: Axes,
  frame_to_cells: Callable[[int], list[NDArray]],
  *,
  max_cells: int,
  nverts: int,
  xlim: tuple[float, float],
  ylim: tuple[float, float],
  edgecolor: str = "blue",
  facecolor: str = "none",
  linewidth: float = 1.5,
):
  polys = init_polygon_pool(
    ax,
    max_cells,
    nverts,
    edgecolor=edgecolor,
    facecolor=facecolor,
    linewidth=linewidth,
  )
  # Initial plot elements: draw one unit cell
  # Function to initialize the animation
  def init():
    # Set the limits of the plot
    ax.set(aspect="equal", xlim=xlim, ylim=ylim)
    # axes off for visibility
    ax.axis("off")
    for poly in polys:
      poly.set_visible(False)
    return polys

  # Animation update function
  def update(frame):  
    # Generate multiple cells based on the frame number
    # get cells which are visible this frame
    cells = frame_to_cells(frame)
    for idx, cell in enumerate(cells):
      polys[idx].set_xy(cell)
      polys[idx].set_visible(True)
    # Hide any leftover artists from larger previous frames.
    for idx in range(len(cells), len(polys)):
      polys[idx].set_visible(False)
    return polys

  return init, update




def tile_hexagonal_lattice(ax: Axes, r=1, num_layers=3, theta0: float=0.0):
  r"""
  hexagonal/triangular Bravais lattice:  L ~= Z^2
  sixfold rotation point group:          C6
  sixfold with mirrors:                  D6
  space group action:                    L semidirect C6 or L semidirect D6  
  """
  nl = int(num_layers)

  n = 6
  h = sin(_2π / n)
  orbit = hexagon_centered(r, theta0)
  A = canonical_basis_2d(Bravais2D.HEXAGONAL, scale=r, theta0=theta0)
  motif = [orbit]
  frame_to_cells = lambda frame: translated_motif_cells(frame, motif, A)
  max_cells = len(frame_to_cells(nl))
  # Set limits a bit larger to accommodate all hexagons
  xlim = (-nl * r * 2, nl * r * 2)
  ylim = (-nl * r * 2 * h, nl * r * 2 * h)
  return animate_tiling(ax, frame_to_cells, max_cells=max_cells, nverts=n, xlim=xlim, ylim=ylim)


def tile_square_lattice(ax: Axes, a: float=1.0, max_layers: int = 6, theta0: float = 0.0):
  n = 4
  orbit = square_centered(a, theta0)
  A = canonical_basis_2d(Bravais2D.SQUARE, scale=a, theta0=theta0)
  motif = [orbit]
  frame_to_cells = lambda frame: translated_motif_cells(frame, motif, A)
  max_cells = len(frame_to_cells(max_layers))
  return animate_tiling(ax, frame_to_cells, max_cells=max_cells, nverts=n, xlim=(-4, 4), ylim=(-4, 4))


def tile_triangular_lattice(ax: Axes, side: float = 1.0, max_layers: int = 4, theta0: float = 0.0):
  """
  Animate a triangular tiling built from a single triangle `unit_cell` (3x2 array).

  Key points:
  - Uses lattice vectors a1 = v1-v0 and a2 = v2-v0 (more correct than using absolute vertices).
  - Precomputes rotated + mirrored variants once.
  - Reuses Polygon artists instead of clearing/recreating every frame.
  """
  n = 3
  motif, A = triangle_pair_motif(side, theta0)
  frame_to_cells = lambda frame: translated_motif_cells(frame, motif, A)
  return animate_tiling(ax, frame_to_cells, max_cells=len(frame_to_cells(max_layers)), 
                        nverts=n, xlim=(-4, 4), ylim=(-4, 4))


def tile_triangular_lattice2(ax: Axes, max_layers: int = 6):
  n = 3
  unit_cell = equilateral_triangle_centered(1.0, theta0=-5 * onp.pi / 6)
  frame_to_cells = lambda frame: tile_space(unit_cell, frame + 1)
  max_cells = len(frame_to_cells(max_layers-1))
  return animate_tiling(ax, frame_to_cells, max_cells=max_cells, nverts=n, 
                        xlim=(-2, 2), ylim=(-1, 2), edgecolor='k', facecolor='random')




# Function to plot a rhombic unit cell
def plot_unit_cell_rhombus(ax: Axes, origin, size):
  points = rhombus_unit_cell(origin, size)
  add_polygon(ax, points, edgecolor='blue', fill=None, linewidth=2)
  # Return vertices for additional plotting (e.g., Wyckoff positions)
  return points


# Function to plot asymmetric units
def plot_asymmetric_units_rhombus(ax: Axes, points, size):
  p1, p2, p3, p4 = points
  # Each rhombus can be split into 6 triangles for p6mm
  tris = [
    [p1, (p1 + p2) / 2, (p1 + p4) / 2],
    [p2, (p1 + p2) / 2, (p2 + p3) / 2]
  ]
  for i, tri in enumerate(tris):
    add_polygon(ax, tri, edgecolor='red', fill=None, linestyle='--', linewidth=1.5)
    # Compute the centroid of the triangle for placing the "F"
    centroid = onp.mean(tri, axis=0)
    angle = -onp.pi / 3 * i  # Rotation angle to align "F" with triangle
    add_text_path(ax, "F", centroid, size=0.2, rotation=angle, lw=0.5)



@dcls.dataclass
class Config:
  shape: str = "hexagon" # "hexagon" | "square" | "triangle"

  # Define the unit cell dimensions (for a square lattice)
  a: float = 1  # Lattice constant


  # plotting
  interval: int = 1000
  repeat: bool = True





def main(cfg: Config):
  # Create the figure and axis for the animation
  fig, ax = plt.subplots()
  fargs = None
  blit = False

  origin = [0, 0]


  if cfg.shape == "hexagon":
    init, update = tile_hexagonal_lattice(ax)
    frames = onp.arange(4)
  elif cfg.shape == "square":
    # Define the unit cell vertices for a simple square lattice
    init, update = tile_square_lattice(ax, cfg.a)
    frames = onp.arange(1, 5)
    blit = True
  elif cfg.shape == "triangle":
    init, update = tile_triangular_lattice(ax)
    frames=range(5)
  elif cfg.shape == "triangle2":
    init, update = tile_triangular_lattice2(ax)
    frames = 6

  elif cfg.shape == "rhombus":
    ax.axis('off')  # Hide axes for better visualization
    # Parameters
    size = 2

    # Plotting the unit cell and its components
    cell_points = plot_unit_cell_rhombus(ax, origin, size)
    plot_asymmetric_units_rhombus(ax, cell_points, size)
    plot_wyckoff_positions(ax, cell_points, size)
    ax.set(aspect='equal', xlim=(-size / 2, size * 2), ylim=(-size / 2, size * 2))
    dedupe_legend(ax)
  elif cfg.shape == "wyckoff_plot":
    ax.axis('off')
    size = 2
    cell_points = plot_unit_cell(ax, origin, size)
    plot_asymmetric_units(ax, cell_points, size)
    plot_mirror_lines(ax, cell_points)
    plot_wyckoff_positions(ax, cell_points, size)
    ax.set(
      aspect='equal',
      xlim=(-size / 2, size * 1.5),
      ylim=(-size / 2, size * 1.5),
    )
    dedupe_legend(ax)
  elif cfg.shape == "unit_cell":
    def init():
      ax.set(aspect='equal', xlim=(-1, 2), ylim=(-1, 2))
      ax.axis('off')
      return

    frames=6
    unit_cell = equilateral_triangle_centered(1.0, theta0=-5 * onp.pi / 6)

    def update(frame):
      num_tiles = frame+1
      triangles = tile_space(unit_cell, num_tiles)
      for triangle in triangles:
        plot_cell(ax, triangle)
      return


    name = 'spiral_animation2.gif'
    # ani.save(name, writer='ffmpeg', fps=360)

  try:
    # Create the animation
    ani = FuncAnimation(fig, update, frames=frames, init_func=init, fargs=fargs,
                          interval=cfg.interval, repeat=cfg.repeat, blit=blit)
  except:
    pass

  plt.show()
    


if __name__ == "__main__":
  main(tyro.cli(Config))
