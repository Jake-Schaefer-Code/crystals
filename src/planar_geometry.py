from __future__ import annotations

import numpy as np
import numpy.linalg as npla
from scipy.spatial import Delaunay

NDArray = np.ndarray


def edge_points(v1, v2, num_pts=20, *, include_start=False, include_end=False):
  """Return evenly spaced points along the edge from ``v1`` to ``v2``."""
  if num_pts < 0:
    raise ValueError("num_pts must be non-negative")

  v1, v2 = np.broadcast_arrays(np.asarray(v1), np.asarray(v2))
  dtype = np.result_type(v1, v2, float)
  if num_pts == 0:
    return np.empty((0,) + v1.shape, dtype=dtype)

  if include_start and include_end:
    ts = np.linspace(0, 1, num_pts)
  elif include_start:
    ts = np.linspace(0, 1, num_pts + 1)[:-1]
  elif include_end:
    ts = np.linspace(0, 1, num_pts + 1)[1:]
  else:
    ts = np.linspace(0, 1, num_pts + 2)[1:-1]

  ts = ts.reshape((num_pts,) + (1,) * v1.ndim)
  return (1 - ts) * v1 + ts * v2


def add_points_to_edge(v1, v2, num_pts=2):
  return list(edge_points(v1, v2, num_pts))


def edge_func(v1, v2, num_pts=20):
  if num_pts < 0:
    raise ValueError("num_pts must be non-negative")

  return edge_points(v2, v1, max(num_pts - 1, 0), include_end=True)


def edge_path_points(verts, pts_per_edge=2, *, closed=True, include_vertices=True):
  """Return evenly spaced points along each edge in a polygon or path."""
  if pts_per_edge < 0:
    raise ValueError("pts_per_edge must be non-negative")

  verts = np.asarray(verts)
  num_verts = len(verts)
  num_edges = num_verts if closed else max(num_verts - 1, 0)
  dtype = np.result_type(verts, float)

  if num_edges == 0 or pts_per_edge == 0:
    return np.empty((0,) + verts.shape[1:], dtype=dtype)

  samples = [
    edge_points(
      verts[i],
      verts[(i + 1) % num_verts],
      pts_per_edge,
      include_start=include_vertices,
    )
    for i in range(num_edges)
  ]
  return np.concatenate(samples, axis=0)


def add_points(polygon, pts_per_edge=2):
  """Create evenly spaced samples along polygon edges."""
  return edge_path_points(polygon, pts_per_edge)


def get_triangles(polygon, pts):
  """Return containing simplex indices from a Delaunay triangulation."""
  tris = Delaunay(polygon[:, :2])
  return tris.find_simplex(pts)


def get_triangles2(triangulation, pts):
  truths = np.array([isinside(pts, triangle) for triangle in triangulation])
  return np.argmax(truths, axis=0)


def area(vertices):
  """Signed polygon area; vertices are assumed to be ordered counterclockwise."""
  return 0.5 * np.sum(
    vertices[..., 0] * np.roll(vertices[..., 1], 1, axis=-1)
    - vertices[..., 1] * np.roll(vertices[..., 0], 1, axis=-1),
    axis=-1,
  )


def isinside(points, triangle):
  """Check whether 2D points lie inside a triangle using area partitioning."""
  shape = (len(points), 3, 2)
  triangle_area = np.abs(area(triangle))

  total_area = 0
  for i in range(3):
    tri = np.full(shape, triangle)
    tri[:, i] = points
    total_area = total_area + np.abs(area(tri))

  return triangle_area == total_area


def centroid(polygon):
  return np.sum(polygon, axis=0) / len(polygon)


def unique_triples(polygon):
  indices = np.arange(0, len(polygon))
  x, y, z = np.meshgrid(indices, indices, indices)
  triples_idxs = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
  triples_idxs = triples_idxs[
    (triples_idxs[:, 0] != triples_idxs[:, 1])
    & (triples_idxs[:, 1] != triples_idxs[:, 2])
    & (triples_idxs[:, 2] != triples_idxs[:, 0])
  ]
  return np.unique(np.sort(triples_idxs, axis=1), axis=0)


def orientation(p1, p2, p3):
  val = np.cross(p1[:2] - p2[:2], p3[:2] - p2[:2])
  if val == 0:
    return 0
  if val > 0:
    return 1
  return 2


def on_segment(p, q, r):
  return (
    min(p[0], r[0]) <= q[0] <= max(p[0], r[0])
    and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])
  )


def segments_intersect(p1, q1, p2, q2):
  o1 = orientation(p1, q1, p2)
  o2 = orientation(p1, q1, q2)
  o3 = orientation(p2, q2, p1)
  o4 = orientation(p2, q2, q1)
  return (
    (o1 != o2 and o3 != o4)
    or (o1 == 0 and on_segment(p1, p2, q1))
    or (o2 == 0 and on_segment(p1, q2, q1))
    or (o3 == 0 and on_segment(p2, p1, q2))
    or (o4 == 0 and on_segment(p2, q1, q2))
  )


def is_self_intersecting(polygon):
  n = len(polygon)
  for i in range(n):
    for j in range(i + 2, n):
      if i == 0 and j == n - 1:
        continue
      if segments_intersect(
        polygon[i],
        polygon[(i + 1) % n],
        polygon[j],
        polygon[(j + 1) % n],
      ):
        return True
  return False


def dist_to_line(points, lines_start, lines_end):
  line_vecs = lines_end - lines_start
  line_vecs_ext = line_vecs[:, None, :]
  pt_vecs = points - lines_start[:, None, :]
  line_lengths_squared = np.sum(line_vecs ** 2, axis=1)
  line_lengths_squared[line_lengths_squared == 0] = np.inf
  line_lengths_squared_ext = line_lengths_squared[:, None]
  t = np.sum(pt_vecs * line_vecs_ext, axis=2) / line_lengths_squared_ext
  t = np.clip(t, 0, 1)
  nearest_points = lines_start[:, None, :] + t[:, :, None] * line_vecs_ext
  return npla.norm(pt_vecs - nearest_points, axis=2)


def get_smallest_angles(poly: NDArray):
  """Return polygon vertex indices sorted by interior angle size."""
  num = poly.shape[0]
  angles = np.zeros(num)
  for i in range(num):
    v1 = poly[(i - 1) % num] - poly[i]
    v2 = poly[(i + 1) % num] - poly[i]
    v1 = v1 / npla.norm(v1)
    v2 = v2 / npla.norm(v2)
    angles[i] = np.arccos(np.dot(v1, v2))
  return np.argsort(angles)


__all__ = [
  "NDArray",
  "add_points",
  "add_points_to_edge",
  "area",
  "centroid",
  "dist_to_line",
  "edge_func",
  "edge_path_points",
  "edge_points",
  "get_smallest_angles",
  "get_triangles",
  "get_triangles2",
  "is_self_intersecting",
  "isinside",
  "on_segment",
  "orientation",
  "segments_intersect",
  "unique_triples",
]
