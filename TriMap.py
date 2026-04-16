# TriMap.py
from __future__ import annotations
import dataclasses as dcls
import numpy as onp
from scipy.spatial import Delaunay

from src import crystal_funcs as cfuncs
from src import coordinates as cconv
from src import planar_geometry as pgeom

NDArray = onp.ndarray

@dcls.dataclass(frozen=True)
class TriMapConfig:
  num_rotations: int = 12
  interpolation_kind: int = 0
  append_centroid: bool = True

@dcls.dataclass(frozen=True)
class TriMapParams:
  polygon: NDArray
  triangle: NDArray
  dim: int
  kind: int
  maps: NDArray
  matrices: NDArray
  triangulations: NDArray

@dcls.dataclass(frozen=True)
class CandidateMap:
  src_polygon: NDArray
  dst_polygon: NDArray
  anchor_indices: tuple[int, int, int]
  triangle_perm: tuple[int, int, int]
  triangulation: Delaunay | None
  jacobians: NDArray | None
  energy: float = onp.inf

class TriMap:
  def __init__(self, polygon:NDArray, triangle:NDArray=None) -> None:
    """
    polygon : array of vertices of polygon
    """
    self.polygon = polygon
    self.dim = polygon.shape[-1]
    self.triangle = triangle
    self._setup_shapes()
    self.cfg = TriMapConfig(12, 0, True)

  def _setup_shapes(self) -> None:
    if self.triangle is None:
      self.triangle = onp.array([(-0.5,0), (0.5,0), (0, onp.sqrt(3)/2)])
    
    self.polygon = self.polygon - pgeom.centroid(self.polygon.copy())
    self.triangle = self.triangle - pgeom.centroid(self.triangle.copy())
    self.unique_triples = pgeom.unique_triples(self.polygon)


  def create_mapping(self, lerp_kind:int=0):
    self.cfg.interpolation_kind = lerp_kind
    maps, matrices, rot_polys, triangulations = [], [], [], []
    rpoly = self.polygon.copy()
    for _ in range(self.cfg.num_rotations):
      for indices in self.unique_triples:
        for i in range(3):
          dst_pts = self.map_polygon_to_triangle(onp.roll(self.triangle.copy(), i, axis=0), indices)
          if dst_pts is None:
            continue
          rpoly_wc = onp.asarray(list(rpoly) + [onp.array([0,0])])
          dst_wc = onp.asarray(list(dst_pts) + [onp.array([0,0])])

          triangulation = Delaunay(dst_wc)
          piecewise_tforms = cfuncs.piecewise_tforms(rpoly_wc, dst_pts, triangulation.simplices)

          matrices.append(piecewise_tforms)
          triangulations.append(triangulation)
          candidate_map = CandidateMap(
            src_polygon=rpoly_wc, 
            dst_polygon=dst_wc, 
            anchor_indices=indices, 
            triangle_perm=i, 
            triangulation=triangulation, 
            jacobians=piecewise_tforms, 
            energy=self.dirichlet_energy(piecewise_tforms, dst_wc, triangulation, rpoly_wc)
          )
      self.polygon = self.polygon @ cfuncs.rot_mat(onp.pi/self.cfg.num_rotations)

    # print([len(m) for m in maps])
    self.maps = onp.asarray(maps)
    # print([M.shape for M in matrices])
    self.matrices = onp.asarray(matrices)
    self.triangulations = onp.asarray(triangulations)
    self.rotated_polygons = onp.asarray(rot_polys)
    return self.maps, self.matrices, self.triangulations, self.rotated_polygons

  def map_polygon_to_triangle(self, triangle:NDArray, indices:NDArray):
    r"""indices: triangle indices to map to"""
    poly = self.polygon
    dst_pts = poly.copy()
    dst_pts[indices] = triangle.copy()
    if pgeom.is_self_intersecting(dst_pts): 
      return None
    last_index = -1
    v0_idx, v1_idx = indices[last_index%3], indices[(last_index+1)%3]
    src_v1, src_v2 = poly[v0_idx], poly[v1_idx]
    dst_v1, dst_v2 = dst_pts[v0_idx], dst_pts[v1_idx]
    for j, p in enumerate(poly):
      if j in indices:
        last_index += 1
        v0_idx, v1_idx = indices[last_index%3], indices[(last_index+1)%3]
        src_v1, src_v2 = poly[v0_idx], poly[v1_idx]
        dst_v1, dst_v2 = dst_pts[v0_idx], dst_pts[v1_idx]
      else:
        dst_pts[j] = cfuncs.interp_vert(p, src_v1, src_v2, dst_v1, dst_v2, kind=self.kind)
    return dst_pts


  def dirichlet_energy(self, matrices:NDArray, mapping:NDArray, triangulation:Delaunay, polygon:NDArray):
    areasA = onp.abs(pgeom.area(polygon[triangulation.simplices]))
    areasB = onp.abs(pgeom.area(mapping[triangulation.simplices]))
    
    spectral_norms = onp.array([max(onp.linalg.svd(J)[1])**2 * B + max(onp.linalg.svd(onp.linalg.inv(J))[1])**2 * A for J, A, B in zip(matrices, areasA, areasB)])
    return onp.sum(spectral_norms)

  def minimize_jacobian(self):
    args = zip(self.matrices, self.maps, self.triangulations, self.rotated_polygons)
    energies = onp.array([self.dirichlet_energy(*args) for args in args])
    # optimal index
    return onp.argmin(energies)



def define_distribution(
  dim: int,
  triangle: NDArray,
  distribution: NDArray | None = None, 
  distribution_weights: NDArray | None = None, 
  alpha: NDArray | None = None, 
  nsamples: int = 10000, 
  rng: onp.random.Generator | None = None
) -> tuple[NDArray, NDArray]:
  """

  Parameters
  ----------------
  distribution

  distribution_weights

  alpha: list, default: uniform Dirichlet

  nsamples: int, default: 10000
  """
  alpha = onp.ones(dim + 1) if alpha is None else onp.array(alpha)
  rng = onp.random.default_rng(rng)

  if distribution is None:
    dirichlet_points = rng.dirichlet(alpha, nsamples)
    # points = onp.dot(dirichlet_points, self.triangle)
    points = cconv.bary_to_cart(dirichlet_points, triangle)
    dist_wts = cconv.barycentric_weights(dirichlet_points)
    keep = dist_wts > 0
    dist = points[keep]
    dist_wts = dist_wts[keep]
    dist_wts = dist_wts / onp.sum(dist_wts)
  else:
    dist = onp.asarray(distribution)
    if dist.shape[-1] != dim + 1:
      raise ValueError(
        f"distribution must be barycentric with last dimension {dim+ 1}; "
        f"got shape {dist.shape}"
      )
    if distribution_weights is None:
      dst_wts = onp.full(dist.shape[0], 1 / dist.shape[0])
    else:
      dst_wts = onp.asarray(distribution_weights, dtype=float)
      if not onp.all(onp.isfinite(dst_wts)):
        raise ValueError("distribution_weights contains non-finite values")
      total_weight = onp.sum(dst_wts)
      if total_weight <= 0:
        raise ValueError("distribution_weights must sum to a positive value")
      dst_wts = dst_wts / total_weight
  
  return dist, dst_wts