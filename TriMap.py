# TriMap.py
from __future__ import annotations

import numpy as onp
from scipy.spatial import Delaunay

from src import crystal_funcs as cfuncs
from src import coordinates as cconv
from src import planar_geometry as pgeom

NDArray = onp.ndarray


class TriMap:
  def __init__(self, polygon:NDArray, triangle:NDArray=None) -> None:
    """
    polygon : array of vertices of polygon
    """
    self.polygon = polygon
    self.dim = polygon.shape[-1]
    self.triangle = triangle
    self._setup_shapes()

  def _setup_shapes(self) -> None:
    if self.triangle is None:
      self.triangle = onp.array([(-0.5,0), (0.5,0), (0, onp.sqrt(3)/2)])
    self.polygon = self.polygon - pgeom.centroid(self.polygon.copy())
    self.triangle = self.triangle - pgeom.centroid(self.triangle.copy())
    self.unique_triples = pgeom.unique_triples(self.polygon)

  def define_distribution(self, distribution=None, distribution_weights=None, alpha=None, nsamples=10000, rng=None) -> None:
    """

    Parameters
    ----------------
    distribution

    distribution_weights

    alpha: list, default: uniform Dirichlet

    nsamples: int, default: 10000
    """
    alpha = onp.ones(self.dim + 1) if alpha is None else onp.array(alpha)
    rng = onp.random.default_rng(rng)

    if distribution is None:
      dirichlet_points = rng.dirichlet(alpha, nsamples)
      # points = onp.dot(dirichlet_points, self.triangle)
      points = cconv.bary_to_cart(dirichlet_points, self.triangle)
      distribution_weights = cconv.barycentric_weights(dirichlet_points)
      keep = distribution_weights > 0
      self.distribution = points[keep]
      self.distribution_weights = distribution_weights[keep]
      self.distribution_weights = self.distribution_weights / onp.sum(self.distribution_weights)
    else:
      if distribution.shape[-1] != self.dim + 1:
        raise ValueError(
          f"distribution must be barycentric with last dimension {self.dim + 1}; "
          f"got shape {distribution.shape}"
        )
      self.distribution = distribution
      if distribution_weights is None:
        self.distribution_weights = onp.full(distribution.shape[0], 1 / distribution.shape[0])
      else:
        self.distribution_weights = onp.asarray(distribution_weights, dtype=float)
        if not onp.all(onp.isfinite(self.distribution_weights)):
          raise ValueError("distribution_weights contains non-finite values")
        total_weight = onp.sum(self.distribution_weights)
        if total_weight <= 0:
          raise ValueError("distribution_weights must sum to a positive value")
        self.distribution_weights = self.distribution_weights / total_weight

  def create_mapping(self, lerp_kind:int=0):
    self.kind = lerp_kind
    maps, matrices, rot_polys, triangulations = [], [], [], []
    num = 12
    for _ in range(num):
      for indices in self.unique_triples:
        for i in range(3):
          dst_pts = self.map_polygon_to_triangle(onp.roll(self.triangle.copy(), i, axis=0), indices)
          if dst_pts is None:
            continue
          polygon_wc = list(self.polygon)
          dst_wc = list(dst_pts)
          polygon_wc.append(onp.array([0,0]))
          dst_wc.append(onp.array([0,0]))
          polygon_wc = onp.asarray(polygon_wc)
          dst_wc = onp.asarray(dst_wc)
          piecewise_tforms, triangulation = self._triangulation_method(polygon_wc, dst_wc)
          matrices.append(piecewise_tforms)
          triangulations.append(triangulation)
          
          maps.append(dst_wc)
          rot_polys.append(polygon_wc.copy())
      self.polygon = self.polygon @ cfuncs.rot_mat(onp.pi/num)

    # print([len(m) for m in maps])
    self.maps = onp.asarray(maps)
    # print([M.shape for M in matrices])
    self.matrices = onp.asarray(matrices)
    self.triangulations = onp.asarray(triangulations)
    self.rotated_polygons = onp.asarray(rot_polys)
    return self.maps, self.matrices, self.triangulations, self.rotated_polygons

  def map_polygon_to_triangle(self, triangle:NDArray, tri_indices:NDArray):
    dst_pts = self.polygon.copy()
    dst_pts[tri_indices] = triangle
    if pgeom.is_self_intersecting(dst_pts): 
      return None
    last_index = -1
    v1_idx, v2_idx = tri_indices[last_index%3], tri_indices[(last_index+1)%3]
    src_v1, src_v2 = self.polygon[v1_idx], self.polygon[v2_idx]
    dst_v1, dst_v2 = dst_pts[v1_idx], dst_pts[v2_idx]
    for j, p in enumerate(self.polygon):
      if j in tri_indices:
        last_index += 1
        v1_idx, v2_idx = tri_indices[last_index%3], tri_indices[(last_index+1)%3]
        src_v1, src_v2 = self.polygon[v1_idx], self.polygon[v2_idx]
        dst_v1, dst_v2 = dst_pts[v1_idx], dst_pts[v2_idx]
      else:
        dst_pts[j] = cfuncs.interp_vert(p, src_v1, src_v2, dst_v1, dst_v2, kind=self.kind)
    return dst_pts

  def _triangulation_method(self, src_pts, dst_pts):
    tri_poly_dst = Delaunay(dst_pts)
    piecewise_matrices = cfuncs.piecewise_tforms(src_pts, dst_pts, tri_poly_dst.simplices)
    return piecewise_matrices, tri_poly_dst

  def dirichlet_energy(self, matrices:NDArray, mapping:NDArray, triangulation:Delaunay, polygon):
    areasA = onp.abs(pgeom.area(polygon[triangulation.simplices]))
    areasB = onp.abs(pgeom.area(mapping[triangulation.simplices]))
    
    spectral_norms = onp.array([max(onp.linalg.svd(J)[1])**2 * B + max(onp.linalg.svd(onp.linalg.inv(J))[1])**2 * A for J, A, B in zip(matrices, areasA, areasB)])
    return onp.sum(spectral_norms)

  def minimize_jacobian(self):
    args = zip(self.matrices, self.maps, self.triangulations, self.rotated_polygons)
    energies = onp.array([self.dirichlet_energy(*args) for args in args])
    # optimal index
    return onp.argmin(energies)
