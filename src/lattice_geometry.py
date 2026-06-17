# lattice_geometry.py
from __future__ import annotations
from enum import Enum, auto
import numpy as onp
from numpy import sin, cos, sqrt
import src.symmetry as sym
from src import crystal_funcs as cfuncs
import src.symmetry.named_groups as groups

NDArray = onp.ndarray

_2π = 2 * onp.pi


class Bravais2D(Enum):
  SQUARE = auto()
  TRIANGULAR = auto()
  HEXAGONAL = auto()
  HONEYCOMB = auto()



def basis_matrix(*vectors: NDArray) -> NDArray:
  """Stack basis vectors as columns of a matrix ``A = [a1 a2 ...]``."""
  if not vectors:
    raise ValueError("basis_matrix requires at least one basis vector")

  cols = tuple(onp.asarray(v, dtype=float) for v in vectors)
  if any(col.ndim != 1 for col in cols):
    raise ValueError("basis vectors must be 1D arrays")

  dim = cols[0].shape[0]
  if any(col.shape != (dim,) for col in cols[1:]):
    raise ValueError("all basis vectors must have the same dimension")

  return onp.column_stack(cols)


def as_basis_matrix(A: NDArray, a2: NDArray | None = None, *, dim: int | None = None) -> NDArray:
  """Normalize either a basis matrix or a pair of basis vectors to columns."""
  if a2 is None:
    A = onp.asarray(A, dtype=float)
    if A.ndim != 2:
      raise ValueError(f"basis matrix must be 2D, got shape {A.shape}")
  else:
    A = basis_matrix(A, a2)

  if dim is not None and A.shape != (dim, dim):
    raise ValueError(f"expected a ({dim}, {dim}) basis matrix, got shape {A.shape}")
  return A


def basis_vectors(A: NDArray) -> tuple[NDArray, ...]:
  """Return the column basis vectors of ``A``."""
  A = as_basis_matrix(A)
  return tuple(A[:, idx].copy() for idx in range(A.shape[1]))



# ---- tiling
def lattice_sites(N: int, A: NDArray, a2: NDArray | None = None) -> list[NDArray]:
  A = as_basis_matrix(A, a2, dim=2)
  a1, a2 = basis_vectors(A)
  return [i * a1 + j * a2 for i in range(-N, N + 1) for j in range(-N, N + 1)]

def translated_motif_cells(
  N: int,
  motif: list[NDArray],
  A: NDArray,
  a2: NDArray | None = None,
) -> list[NDArray]:
  cells = []
  for site in lattice_sites(N, A, a2):
    for poly in motif:
      cells.append(poly + site)
  return cells


# ---- lattice basis

# TODO how to find basis vectors using symmetry groups rather than hard-coding?

def canonical_basis_2d(kind: Bravais2D, scale: float = 1.0, theta0: float = 0.0) -> NDArray:
  if kind is Bravais2D.SQUARE:
    A0 = basis_matrix(
      onp.array([1.0, 0.0]),
      onp.array([0.0, 1.0]),
    )
  elif kind is Bravais2D.TRIANGULAR:
    r""" 
    Direct lattice:
      a1 = a (1, 0)
      a2 = a (1/2, sqrt(3)/2)
    """
    A0 = basis_matrix(
      onp.array([1.0, 0.0]),
      onp.array([cos(_2π/6), sin(_2π/6)]),
    )
  elif kind is Bravais2D.HEXAGONAL:
    A0 = basis_matrix(
      onp.array([1.5, sin(_2π / 6)]),
      onp.array([0.0, 2 * sin(_2π / 6)]),
    )
    # flat-top: θ_0 = 0, pointy-top: θ_0 = π / 
  elif kind is Bravais2D.HONEYCOMB:
    """ Honeycomb lattice basis vectors. """
    A0 = basis_matrix(
      onp.array([2 * sin(_2π/6), 0.0]),
      onp.array([sin(_2π/6), 1.5 * sin(_2π/6)]),
    )
  else:
    raise ValueError(f"unsupported Bravais class: {kind}")

  R = sym.rotation2d(theta0).matrix
  return R @ (scale * A0)


def lattice_action(A: NDArray, op: sym.AffineOperation) -> NDArray:
  A = as_basis_matrix(A)
  return onp.linalg.solve(A, op.matrix @ A)

def preserves_lattice(
  A: NDArray,
  op: sym.AffineOperation,
  tol: float = 1e-10,
) -> bool:
  M = lattice_action(A, op)
  M_round = onp.round(M)
  det = round(onp.linalg.det(M_round))
  return onp.allclose(M, M_round, atol=tol) and abs(det) == 1

def body_centered_cubic_lattice_basis(a: float = 1.0, theta0: float = 0.0):
  a1 = a * onp.array([1.0, 0.0, 0.0])
  a2 = a * onp.array([0.0, 1.0, 0.0])
  a3 = a * onp.array([0.0, 0.0, 1.0])
  R = sym.rotation3d(theta0, axis=onp.array([1.0, 1.0, 1.0])).matrix
  return basis_matrix(R @ a1, R @ a2, R @ a3)





# ---- centered lattices

def equilateral_triangle_centered(side: float = 1.0, theta0: float = 0.0):
  R = side / sqrt(3.0)  # circumradius
  C3 = sym.cyclic_group(3, dim=2)
  v0 = R * onp.array([cos(theta0), sin(theta0)])
  return C3.orbit(v0)  # (3, 2), centroid = 0

def hexagon_centered(r: float = 1.0, theta0: float = 0.0):
  C6 = sym.cyclic_group(6, dim=2)
  v0 = r * onp.array([cos(theta0), sin(theta0)])
  return C6.orbit(v0)

def square_centered(a: float = 1.0, theta0: float = 0.0):
  R = a / sqrt(2)
  C4 = sym.cyclic_group(4, dim=2)
  # bottom left quadrant and shifting up and right 
  # (equiv to starting at 0,0 but rotating about some point in upper right quad)
  # TODO calculate center instead??
  v0 = R * onp.array([cos(theta0 + _2π/8), sin(theta0 + _2π/8)])
  return C4.orbit(v0)





def make_triangular_lattice(n1: int, n2: int, a=1.0):
  """
  Triangular Bravais lattice with primitive vectors
    a1 = a (1, 0)
    a2 = a (1/2, sqrt(3)/2)
  """
  A = canonical_basis_2d(Bravais2D.TRIANGULAR, scale=a)
  sites = lattice_sites(max(n1, n2), A)
  return onp.array(sites, dtype=onp.float64), A


def triangle_pair_motif(side: float = 1.0, theta0: float = 0.0):
  A = canonical_basis_2d(Bravais2D.TRIANGULAR, side, theta0)
  a1, a2 = basis_vectors(A)
  # One rhombic/parallelogram primitive cell split into two triangles.
  tri_up = onp.array([
    [0.0, 0.0],
    a1,
    a2,
  ])
  tri_down = onp.array([
    a1 + a2,
    a1,
    a2,
  ])
  return [tri_up, tri_down], A



def two_triangle_motif(
  side: float = 1.0,
  theta0: float = 0.0,
  *,
  center_cell: bool = False,
) -> tuple[list[NDArray], NDArray]:
  motif, A = triangle_pair_motif(side, theta0)
  a1, a2 = basis_vectors(A)
  if center_cell:
    c = 0.5 * (a1 + a2)   # centroid of the rhombic cell
    motif = [tri - c for tri in motif]

  return motif, A


def triangle_d3_wedges(side: float = 1.0, theta0: float = 0.0):
  tri = equilateral_triangle_centered(side, theta0)
  v0, v1, v2 = tri
  c = tri.mean(axis=0)  # should already be ~0

  # one asymmetric unit of the triangle
  wedge = onp.array([
    c,
    v0,
    0.5 * (v0 + v1),
  ])

  D3 = groups.dihedral_group(3)
  if theta0 != 0.0:
    D3 = sym.conjugate_action(D3, sym.rotation2d(theta0))

  wedges = [g.apply(wedge) for g in D3]
  return tri, wedges




def tile_space(unit_cell, num_tiles, rot_order: int=6, reflect_every: int=2):
  r"""
  Ex (Hexagonal lattice):
    - local point-group orbit: C6 acting around one fixed center/pivot
    - global Bravais tiling: translations by lattice vectors
    - full symmetry: translations semidirect product with C6 or D6
  """
  tris = [unit_cell]
  pivot = unit_cell[2]  # pivot at the top point of the triangle

  # Generate the tiling
  for i in range(1, num_tiles):
    # rotation about a pivot p is an affine map:
    #     x -> R(x - p) + p

    # 60 degrees rotation for hexagonal symmetry
    θ = i * _2π / rot_order 
    R = sym.rotation2d(θ)
    rotated = R.apply(unit_cell - pivot) + pivot
    tris.append(rotated)
    # Add the reflected triangle
    if i % reflect_every == 0:  # reflect every second triangle to maintain symmetry
      # Vector along the line
      v = rotated[2] - rotated[0]
      # Normal to the line
      normal = onp.array([-v[1], v[0]])
      ref = sym.reflection(normal)
      tris.append(ref.apply(rotated))

  return tris


def make_honeycomb_lattice(n1: int, n2: int, a=1.0):
  """
  Honeycomb lattice as triangular Bravais lattice + two-point basis.

  Here a is the nearest-neighbor distance.
  """
  # Triangular Bravais lattice vectors
  A = canonical_basis_2d(Bravais2D.HONEYCOMB, scale=a)

  # Two-site basis
  bA = onp.array([0.0, 0.0])
  bB = onp.array([sin(_2π/6) * a, cos(_2π/6) * a])
  
  sites = onp.array(lattice_sites(max(n1, n2), A))
  ptsA = sites + bA
  ptsB = sites + bB
  return ptsA, ptsB, A, bA, bB


def first_bz_hexagon_vertices(B: NDArray) -> NDArray:
  r"""
  Vertices of the first Brillouin zone (regular hexagon)
    for the triangular reciprocal lattice.
    These are the K-points.
  """
  B = as_basis_matrix(B, dim=2)
  v3 = onp.array([[1, 0.5], [0.5, 1], [-0.5, 0.5]], dtype=onp.float32)
  v3b = v3 @ B.T
  verts = onp.concatenate((v3b, -v3b))
  return verts * (2 / 3)


def finite_triangular_patch(A: NDArray, N: int, a2: NDArray | None = None) -> NDArray:
  """
  Hexagonal-ish finite patch of lattice sites:
    c_l = n1 a1 + n2 a2
  with max(|n1|, |n2|, |n1+n2|) <= N
  """
  A = as_basis_matrix(A, a2, dim=2)
  a1, a2 = basis_vectors(A)
  sites = []
  for n1 in range(-N, N + 1):
    for n2 in range(-N, N + 1):
      if max(abs(n1), abs(n2), abs(n1 + n2)) <= N:
        sites.append(n1 * a1 + n2 * a2)
  return onp.array(sites)
