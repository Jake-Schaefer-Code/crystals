# physics/simplex.py
r"""The probability simplex on three states: barycentric coordinates, meshes, and boundary regularization.

A distribution ``(p_0, p_1, p_2)`` is drawn at the corresponding convex combination of the corners of
an equilateral triangle (``SIMPLEX_VERTICES``). ``plotting.simplex`` draws the frame and labels.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

SIMPLEX_VERTICES = np.array([
  [0.0, 0.0],
  [1.0, 0.0],
  [0.5, np.sqrt(3.0) / 2.0],
])
r""" Corners of the plotted triangle, one row per pure state ``e_0, e_1, e_2``. """


def simplex_to_cartesian(points):
  r""" Map barycentric points of shape ``(..., 3)`` to plane coordinates of shape ``(..., 2)``. """
  return np.asarray(points) @ SIMPLEX_VERTICES


def sample_simplex(levels=45):
  r""" Regular mesh of ``(levels + 1)(levels + 2) / 2`` points ``(i, j, k) / levels`` with ``i + j + k = levels``. """
  points = []
  for i in range(levels + 1):
    for j in range(levels + 1 - i):
      k = levels - i - j
      points.append((i / levels, j / levels, k / levels))
  return jnp.array(points, dtype=jnp.float32)


def regularize_simplex(points, eps=1e-3):
  # Keep KL terms finite near the simplex boundary without changing the plot mesh.
  points = jnp.clip(points, eps, None)
  return points / jnp.sum(points, axis=-1, keepdims=True)


def normalize_simplex_point(point):
  r""" Rescale a nonnegative vector to sum to one. """
  point = np.asarray(point)
  return point / point.sum()
