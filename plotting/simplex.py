# plotting/simplex.py
r"""Drawing on the three-state probability simplex; the geometry is ``physics.simplex``."""
from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes

from physics.simplex import SIMPLEX_VERTICES


def format_prob(point, *, latex: bool = False) -> str:
  r""" ``(0.25, 0.50, 0.25)``; with ``latex=True`` the math-mode form with thin spaces. """
  entries = [f"{x:.2f}" for x in np.asarray(point)]
  if latex:
    return r"\left(" + r",\,".join(entries) + r"\right)"
  return "(" + ", ".join(entries) + ")"


def draw_simplex_frame(ax: Axes, color="0.25", lw=1.0):
  verts = SIMPLEX_VERTICES[[0, 1, 2, 0]]
  ax.plot(verts[:, 0], verts[:, 1], color=color, lw=lw, zorder=2)


def add_corner_labels(ax: Axes, fontsize=9, color="0.2", *, inside: bool = False):
  r""" Label the corners ``(1,0,0)``, ``(0,1,0)``, ``(0,0,1)``; ``inside`` tucks them into the triangle. """
  V = SIMPLEX_VERTICES
  if inside:
    ax.text(V[0, 0] + 0.035, V[0, 1] + 0.025, r"$(1,0,0)$", ha="left", va="bottom", fontsize=fontsize, color=color)
    ax.text(V[1, 0] - 0.035, V[1, 1] + 0.025, r"$(0,1,0)$", ha="right", va="bottom", fontsize=fontsize, color=color)
    ax.text(V[2, 0], V[2, 1] - 0.035, r"$(0,0,1)$", ha="center", va="top", fontsize=fontsize, color=color)
  else:
    ax.text(V[0, 0] - 0.03, V[0, 1] - 0.00, r"$(1,0,0)$", ha="right", va="top", fontsize=fontsize, color=color)
    ax.text(V[1, 0] + 0.03, V[1, 1] - 0.00, r"$(0,1,0)$", ha="left", va="top", fontsize=fontsize, color=color)
    ax.text(V[2, 0], V[2, 1] + 0.03, r"$(0,0,1)$", ha="center", va="bottom", fontsize=fontsize, color=color)
