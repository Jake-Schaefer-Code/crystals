# src/crystal_funcs.py
from __future__ import annotations
import itertools
from math import comb
import numpy as np
import numpy.linalg as npla
from scipy.spatial import Delaunay
from scipy.stats import gaussian_kde

from src import coordinates as cconv
from src import geo_ops_utils as gops
from src.planar_geometry import (
  add_points,
  add_points_to_edge,
  area,
  centroid,
  dist_to_line,
  edge_func,
  edge_path_points,
  edge_points,
  get_smallest_angles,
  get_triangles,
  get_triangles2,
  is_self_intersecting,
  isinside,
  on_segment,
  orientation,
  segments_intersect,
  unique_triples,
)

NDArray = np.ndarray

# Backward-compatible aliases used by notebooks and TriMap.
rot_mat = gops.rot_mat
q_rot_mat = gops.q_rot_mat
generate_rotation_matrices = gops.generate_rotation_matrices
rotate_polyhedron = gops.rotate_polyhedron
rotate_about_pivot = gops.rotate_about_pivot
rotate_around_axis = gops.rotate_around_axis


def is_inside_tetrahedron(points, tetrahedron):
  """TODO do this with barycentric coords"""
  hull = Delaunay(tetrahedron)
  return hull.find_simplex(points) >= 0

def tetrahedron_volume(tetrahedrons):
  return np.abs(np.sum(
        np.cross(tetrahedrons[..., 1, :] - tetrahedrons[..., 0, :], 
                 tetrahedrons[..., 2, :] - tetrahedrons[..., 0, :]) * 
                (tetrahedrons[..., 3, :] - tetrahedrons[..., 0, :]), axis=-1)) / 6

def simplex_centroid(simplices: NDArray) -> NDArray:
  return np.sum(simplices, axis=-2)/simplices.shape[-2]

def polyhedron_centroid(vertices: NDArray) -> NDArray:
  Δs = vertices[Delaunay(vertices).simplices]
  volumes = tetrahedron_volume(Δs)
  centroids = simplex_centroid(Δs)
  return np.sum(centroids * volumes[:,None], axis=0) / np.sum(volumes)

def unique_tuples(num_pts, tuple_len):
  indices = np.arange(0, num_pts)
  combinations = np.array(np.meshgrid(*[indices for _ in range(tuple_len)])).T.reshape(-1, tuple_len)
  combinations.sort(axis=1)
  unique_mask = np.all(np.diff(combinations, axis=1) != 0, axis=1)
  return np.unique(combinations[unique_mask], axis=0)


# Mapping and Transformations

def get_parameter(p, v1, v2, t1, t2, kind):  
  vec = v2 - v1
  dist_p_v1 = npla.norm(p - v1)
  dist_p_v2 = npla.norm(p - v2)
  if kind == 0:
    param = (dist_p_v1 / (dist_p_v1 + dist_p_v2))
  elif kind == 1:
    param = np.dot(p-v1, vec) / np.dot(vec, vec)
  elif kind == 2:
    area_a = npla.norm(np.cross(p-v1, p - t1)) 
    area_b = npla.norm(np.cross(p-v2 , p - t2)) 
    param = (area_a / (area_a + area_b)) * (dist_p_v1 / (dist_p_v1 + dist_p_v2))
  return np.clip(param, 0.01, 0.99)

def interp_vert(p, v1, v2, t1, t2, kind):    
  param = get_parameter(p, v1, v2, t1, t2, kind)
  return t1 + param * (t2-t1)

def planar_interp(p, t1, t2, t3, v1, v2, v3):
  r""" interpolation to plane """
  dist_p_v1 = npla.norm(p - v1, axis=1, keepdims=True)
  dist_p_v2 = npla.norm(p - v2, axis=1, keepdims=True)
  dist_p_v3 = npla.norm(p - v3, axis=1, keepdims=True)
  p1 = (dist_p_v1 / (dist_p_v1 + dist_p_v2)) # param1
  p2 = (dist_p_v2 / (dist_p_v2 + dist_p_v3)) # param2
  return ((t1 + p1 * (t2-t1)) + (t2 + p2 * (t3-t2)))/2

def slerp(v0, v1, t):
  """ sinusoidal linear interpolation (for stability?) """
  dot = np.dot(v0, v1)
  theta = np.arccos(dot) * t
  s0 = np.sin((1 - t) * theta)
  s1 = np.sin(t * theta)
  return  (s0 * v0 + s1 * v1) / np.sin(theta)

def orthogonal_complement(q1, q2, q3):
  r"""!!! TODO """
  matrix = np.array([q1, q2, q3])
  u, s, vh = npla.svd(matrix)
  return vh[-1]

# instead of doing this, rotate orthogonal to q
def ortho_proj(p, q):
  """Return the unit component of p orthogonal to q."""
  p = np.asarray(p, dtype=float)
  q = np.asarray(q, dtype=float)
  q_norm_sq = np.dot(q, q)
  if np.isclose(q_norm_sq, 0.0):
    raise ValueError("q must be non-zero")

  projection = p - (np.dot(p, q) / q_norm_sq) * q
  projection_norm = npla.norm(projection)
  if np.isclose(projection_norm, 0.0):
    raise ValueError("p must not be parallel to q")
  return projection / projection_norm


def _monomial_exponents(ndim, max_degree):
  """Yield exponent tuples with total degree <= max_degree."""
  if ndim <= 0:
    raise ValueError("ndim must be positive")
  if max_degree < 0:
    raise ValueError("max_degree must be non-negative")

  def build(prefix, remaining_dims, remaining_degree):
    if remaining_dims == 1:
      for exponent in range(remaining_degree + 1):
        yield tuple(prefix + [exponent])
      return

    for exponent in range(remaining_degree + 1):
      yield from build(
        prefix + [exponent],
        remaining_dims - 1,
        remaining_degree - exponent,
      )

  yield from build([], ndim, max_degree)


def polynomial_basis(p: NDArray, max_degree: int=10):
  r"""Return all monomials in p with total degree up to max_degree."""
  coords = np.broadcast_arrays(*[np.asarray(coord) for coord in p])
  dtype = np.result_type(*coords, float)

  terms = []
  # tuples of exponents
  for exponents in _monomial_exponents(len(coords), max_degree):
    # 
    term = np.ones_like(coords[0], dtype=dtype)
    # 
    for coord, exponent in zip(coords, exponents):
      if exponent:
        term = term * (coord ** exponent)
    terms.append(term)
  return np.array(terms)



def compute_coefficients(points_src: NDArray, points_dst: NDArray, max_degree: int=10):
  if len(points_src) != len(points_dst):
    raise IndexError(f"Must have the same amount of source points and destination points: {len(points_src)} != {len(points_dst)}")
  dim = points_src.shape[1]

  A = np.zeros((len(points_src), comb(max_degree + dim, dim)))
  for i, p in enumerate(points_src):
    A[i, :] = polynomial_basis(p, max_degree)

  return [npla.lstsq(A, points_dst[:, i], rcond=None)[0] for i in range(dim)]

def affine_mat(src: NDArray, dst: NDArray):
  """
  src: source points
  dst: destination points
  """
  A = np.vstack([src.T, np.ones((src.shape[0],1)).T]).T
  B = np.vstack([dst.T, np.ones((dst.shape[0],1)).T]).T
  return npla.lstsq(A, B, rcond=None)[0].T

def apply_affine_mat(p, M) -> NDArray:
  return (M @ np.append(p, 1))[:-1]


def piecewise_tforms(src, dst, indices):
  """
  indices: triangle indices
  """
  return np.array([affine_mat(src[s2], dst[s2]) for s2 in indices])



def geodesic(p1, p2, num_points=100):
  p1 /= npla.norm(p1)
  p2 /= npla.norm(p2)
  t = np.linspace(0, 1, num_points)[:, None]
  theta = np.arccos(np.dot(p1, p2))
  sin_theta = np.sin(theta)
  if sin_theta==0: 
    print(f"geodesic(): divide by zero: sin(theta)=0, cos(theta)={np.dot(p1, p2)}")
    sin_theta=1
  
  return (np.sin((1 - t) * theta) / sin_theta) * p1 + (np.sin(t * theta) / sin_theta) * p2

def dist_to_geodesic(p,v1,v2):
  p /= npla.norm(p)
  n = np.cross((v1 / npla.norm(v1)), (v2 / npla.norm(v2)))
  n /= npla.norm(n)
  theta = np.arccos(np.dot(p,n))
  return np.pi / 2 - theta 

def find_vertex_angles(polyhedron, adjacency_list):
  """
  Parameters
  ----------------
  polyhedron

  adjacency_list
  """
  angles = np.zeros(len(polyhedron))
  for i, v in enumerate(polyhedron):
    edge_vecs = polyhedron[adjacency_list[i]] - v[None, :]
    edge_vecs /= npla.norm(edge_vecs, axis=1, keepdims=True)
    pairs = list(itertools.combinations(range(len(edge_vecs)), 2))
    thetas = np.arccos(np.sum(edge_vecs[pairs][:,0] * edge_vecs[pairs][:,1], axis=-1))
    angles[i] = np.sum(thetas)
  return np.argsort(angles)


def parametrize_geodesic_triangle(q1, q2, q3, num_points: int=10):
  s, t = np.meshgrid(np.linspace(0,1,num_points+1), np.linspace(0,1,num_points+1))
  mask = (s + t <= 1)
  s, t = s[mask][:,None], t[mask][:,None]
  points = s * q1 + t * q2 + (1-s-t)*q3
  return points / npla.norm(points, axis=1, keepdims=True)

def parametrize_triangle(p1, p2, p3, num_points=10):
  α, β = np.meshgrid(np.linspace(0,1,num_points+1), np.linspace(0,1,num_points+1))
  α = α.ravel()
  β = β.ravel()
  mask = (α + β <= 1)
  α, β = α[mask], β[mask]
  γ = 1 - α - β
  points = sum([α[:, None] * np.array(p1), β[:, None] * np.array(p2), γ[:, None] * np.array(p3)])
  return points

def scale_poly(points: NDArray) -> NDArray:
  max_dist = np.asarray(np.max(npla.norm(points, axis=1)))
  scaled = points / max_dist
  return scaled.astype(points.dtype)


def dirichlet_energy(matrices: NDArray, mapping: NDArray, triangulation: NDArray, polyhedron):

  # TODO dont need to calculate volumes every time (in class)

  vol0 = tetrahedron_volume(polyhedron[triangulation])
  vol1 = tetrahedron_volume(mapping[triangulation])
  if (vol0 == 0).any() or (vol1==0).any():
    return np.inf
  try:
    # B * len(triangulation) / np.sum(vol1)
    spectral_norms = np.array([max(npla.svd(J)[1])**2*B + max(npla.svd(npla.inv(J))[1])**2*A
                            for J, A, B in zip(matrices, vol0, vol1)])
    return np.sum(spectral_norms)
  except:
    print("singular matrices")
    return np.inf

def minimize_jacobian(matrices, maps, triangulations, rotated_polygons):
  energies = np.array([dirichlet_energy(*x) for x in zip(matrices, maps, triangulations, rotated_polygons)])
  return np.argmin(energies)


def distance_weights(coords, points, w=None):
  """
  weights based on distances from points

  Parameters
  ----------------
  coords, 
  
  points, 
  
  w=None
  """
  distances = npla.norm(coords[:, None, :] - points, axis=2)
  closest_indices = np.argmin(distances, axis=0)
  weight_sums = np.bincount(closest_indices, minlength=coords.shape[0])
  if w is None:
      return weight_sums / np.sum(weight_sums)
  weights = np.bincount(closest_indices, weights=w, minlength=coords.shape[0])
  # Mean weight value at each coordinate. Coordinates that receive no points
  # should remain zero instead of becoming nan from division by zero.
  valid = weight_sums > 0
  weights = weights.astype(float)
  weights[valid] /= weight_sums[valid]
  weights[~valid] = 0
  # norm to 1
  total = np.sum(weights)
  if total == 0:
      return weights
  return weights / total


def weighted_distribution(f_sym: NDArray, 
                        coordinates: NDArray, 
                        simplex: NDArray, 
                        alpha: NDArray = None, 
                        n_samples: int = 10000,
                        rng=None):
  """
  coordinates must be of shape (ndim, npts)

  Parameters
  ----------------
  f_sym : NDArray, 

  coordinates : NDArray, 

  simplex : NDArray, 

  alpha : NDArray = None, 

  n_samples : int = 10000
  """

  alpha = np.ones(len(simplex)) if alpha is None else np.array(alpha)
  rng = np.random.default_rng(rng)

  dir_points = rng.dirichlet(alpha, n_samples)
  dir_points_cart = cconv.bary_to_cart(dir_points, simplex)

  f_values = np.nan_to_num(np.asarray(f_sym, dtype=float).ravel())
  f_values = f_values - np.min(f_values)
  total = np.sum(f_values)
  if total == 0:
    f_weights = np.full_like(f_values, 1 / len(f_values), dtype=float)
  else:
    f_weights = f_values / total

  kde = gaussian_kde(coordinates, weights=f_weights, bw_method=0.05)
  sampled_f: NDArray = kde(dir_points_cart.T)
  assert isinstance(sampled_f, np.ndarray)
  sampled_f_total = np.sum(sampled_f)
  if sampled_f_total == 0:
    return dir_points_cart, np.full(n_samples, 1 / n_samples, dtype=float)
  sampled_f /= sampled_f_total

  # probabilities
  sampled_range = np.max(sampled_f) - np.min(sampled_f)
  if sampled_range == 0:
    probs = np.full(n_samples, 1 / n_samples, dtype=float)
  else:
    probs = (sampled_f - np.min(sampled_f)) / sampled_range
  # normalize
  probs = probs / np.sum(probs)

  # probabilities *= barycentric_weights(dir_points)
  return dir_points_cart, probs


def collect_matrices(
  src_tetra: NDArray, 
  dst_ply: NDArray, 
  num_rots: int = 12,
) -> tuple[list[NDArray], list[Delaunay], list[list[float]], list[list[float]]]:
  """
  Ex: src_ply is the fund domain simplex with defined distribution, 
  dst_ply is the destination polytope
  """
  dst_w_cent = np.array(list(dst_ply) + [polyhedron_centroid(dst_ply)])

  # TODO look at this more - something is up with the way it isnt covering the top of the sphere
  mats = gops.generate_rotation_matrices(num_rots)
  rotated_plys = gops.rotate_polyhedron(dst_w_cent, mats)
  combos = unique_tuples(dst_ply.shape[0], 4)
  # piecewise_matrices, dst, triangulations
  pw_mats, dst, tris, src, simps = [], [], [], [], []
  for combo in combos:
    rotated_tetras = rotated_plys[:, combo, :]
    mask = np.ones(dst_w_cent.shape[0], dtype=bool)
    mask[combo] = False
    mask[-1] = False
    # other pts
    others = rotated_plys[:, mask, :]
    dst_pts = np.zeros_like(rotated_plys)
    dst_pts[:, combo, :] = src_tetra
    
    # bary coords for other pts
    bary_others = np.array([cconv.calculate_barycentric_coordinates(src_tetra, rot) for rot in others])
    # Should I do abs. val of 
    closest_face = np.argsort(bary_others, axis=-1)[:,:, -3:]
    
    rows = np.arange(others.shape[0])[:, None, None]
    tetra_faces = src_tetra[closest_face]
    faces_poly = rotated_tetras[rows, closest_face]
    interped_pts = np.array([
      planar_interp(
        pts, 
        # *np.transpose(t_face, (1, 0))[:3],
        t_face[:,0], t_face[:,1], t_face[:,2], 
        p_face[:, 0], p_face[:, 1], p_face[:, 2]
      ) 
      for pts, t_face, p_face in zip(others, tetra_faces, faces_poly)
    ])
    
    dst_pts[:, mask, :] = interped_pts

    tri_dst = [Delaunay(pt) for pt in dst_pts]
    simplices = [tri.simplices for tri in tri_dst]

    tris.extend(tri_dst)
    simps.extend(simplices)
    dst.extend(list(dst_pts.copy()))
    src.extend(list(rotated_plys.copy()))


    pw_mats.extend(
      list(map(
        lambda x: piecewise_tforms(*x), 
        zip(rotated_plys, dst_pts, simplices)
      ))
    )
      
  return pw_mats, tris, simps, dst, src

