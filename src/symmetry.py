# src/symmetry.py
"""Finite symmetry actions with representation-theory oriented helpers.

The existing notebooks mostly represent symmetry operations as ``(M, t)``
pairs acting by ``x -> M x + t``.  This module keeps that convention while
adding a small amount of structure: named affine operations, finite group
actions, orbits, characters, and Reynolds/projector helpers.
"""

from __future__ import annotations
from collections.abc import Iterable, Iterator, Sequence
import dataclasses as dcls
from enum import Enum, auto
import math
import numpy as onp
from src import geo_ops_utils as gops
from src.core.static_types import ReplaceMixin

NDArray = onp.ndarray
_2PI = 2.0 * onp.pi

class Bravais2D(Enum):
  SQUARE = auto()
  TRIANGULAR = auto()
  HEXAGONAL = auto()
  HONEYCOMB = auto()



def recip_lattice(A: NDArray) -> NDArray:
  r"""
  Reciprocal lattice:
    b_i . a_j = 2 pi delta_ij
  """
  return _2PI * onp.linalg.inv(A).T # columns are b1, b2



@dcls.dataclass(frozen=True)
class Lattice:
  basis: NDArray        # columns = direct lattice basis vectors
  reciprocal_basis: NDArray
  metric: NDArray
  reciprocal_metric: NDArray

  @classmethod
  def from_basis(cls, basis: NDArray) -> "Lattice":
    A = onp.asarray(basis, dtype=float)
    B = recip_lattice(A)
    G = A.T @ A
    G_star = B.T @ B
    return cls(A, B, G, G_star)

@dcls.dataclass(frozen=True)
class SeitzOp:
  R: NDArray   # integer/rational in lattice basis when possible
  tau: NDArray # fractional translation
  label: str = ""
  def __iter__(self) -> Iterator[NDArray]:
    yield self.R
    yield self.tau

@dcls.dataclass(frozen=True)
class PointGroup:
  elements: tuple[NDArray, ...]
  ...

@dcls.dataclass(frozen=True)
class SpaceGroup:
  lattice: Lattice
  point_group: PointGroup
  coset_reps: tuple[SeitzOp, ...]   # or full generators + closure logic
  ...

@dcls.dataclass(frozen=True)
class Site:
  frac: NDArray
  species: str
  dof: object | None = None

@dcls.dataclass(frozen=True)
class Crystal:
  lattice: Lattice
  sites: tuple[Site, ...]
  space_group: SpaceGroup | None = None

@dcls.dataclass(frozen=True)
class Representation:
  group: PointGroup | SpaceGroup | object
  matrices: dict[object, NDArray]   # maps group element -> rho(g)
  name: str = ""

  def __post_init__(self) -> None:
    if not self.matrices:
      raise ValueError("Representation requires at least one matrix")

    normalized: dict[object, NDArray] = {}
    dim: int | None = None
    dtype = float
    for element, matrix in self.matrices.items():
      matrix = onp.asarray(matrix)
      if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
          "representation matrices must be square, "
          f"got shape {matrix.shape} for element {element!r}"
        )
      if dim is None:
        dim = matrix.shape[0]
      elif matrix.shape != (dim, dim):
        raise ValueError(
          "all representation matrices must have the same shape, "
          f"got {(dim, dim)} and {matrix.shape} for element {element!r}"
        )
      dtype = onp.result_type(dtype, matrix.dtype)
      normalized[element] = matrix

    dtype = onp.result_type(dtype, float)
    object.__setattr__(
      self,
      "matrices",
      {element: onp.asarray(matrix, dtype=dtype) for element, matrix in normalized.items()},
    )

  @property
  def elements(self) -> tuple[object, ...]:
    return tuple(self.matrices.keys())

  @property
  def order(self) -> int:
    return len(self.matrices)

  @property
  def dim(self) -> int:
    return next(iter(self.matrices.values())).shape[0]

  @property
  def matrix_stack(self) -> NDArray:
    return onp.stack(tuple(self.matrices.values()), axis=0)

  def character(self) -> NDArray:
    return onp.trace(self.matrix_stack, axis1=-2, axis2=-1)

  def projector(self, chi: NDArray | dict[object, complex], irrep_dim: int | float) -> NDArray:
    r"""Projection matrix ``d/|G| \sum_g \overline{\chi(g)} \rho(g)``."""

    if isinstance(chi, dict):
      missing = [element for element in self.elements if element not in chi]
      if missing:
        raise ValueError(f"character dictionary is missing values for elements {missing!r}")
      chi = [chi[element] for element in self.elements]

    chi = onp.asarray(list(chi), dtype=complex)
    if chi.shape != (self.order,):
      raise ValueError(f"character must have shape {(self.order,)}, got {chi.shape}")

    return irrep_dim / self.order * onp.einsum(
      "g,gij->ij",
      onp.conjugate(chi),
      self.matrix_stack,
    )

  def reynolds_projector(self) -> NDArray:
    r"""Projection matrix onto the invariant subspace."""
    return self.projector(onp.ones(self.order, dtype=complex), 1)

@dcls.dataclass(frozen=True)
class CharacterTable:
  group: PointGroup | SpaceGroup | object
  characters: dict[object, NDArray]   # maps group element -> chi(g)
  name: str = ""
  ...



@dcls.dataclass(frozen=True)
class AffineOperation(ReplaceMixin):
  """An affine symmetry operation ``x -> matrix @ x + translation``."""

  matrix: NDArray
  translation: NDArray | None = None
  label: str = ""

  def __post_init__(self) -> None:
    matrix = onp.asarray(self.matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
      raise ValueError(f"matrix must be square, got shape {matrix.shape}")

    dtype = onp.result_type(matrix.dtype, float)
    if self.translation is None:
      translation = onp.zeros(matrix.shape[0], dtype=dtype)
    else:
      translation = onp.asarray(self.translation)
      dtype = onp.result_type(dtype, translation.dtype)
      translation = onp.asarray(translation, dtype=dtype)

    if translation.shape != (matrix.shape[0],):
      raise ValueError(
        "translation must have shape "
        f"{(matrix.shape[0],)}, got {translation.shape}"
      )

    matrix = onp.asarray(matrix, dtype=dtype)
    object.__setattr__(self, "matrix", matrix)
    object.__setattr__(self, "translation", translation)

  def __iter__(self) -> Iterator[NDArray]:
    yield self.matrix
    yield self.translation

  @property
  def dim(self) -> int:
    return self.matrix.shape[0]

  @property
  def character(self) -> complex:
    """Character of the defining linear representation at this operation."""
    return onp.trace(self.matrix).item()

  @property
  def determinant(self) -> complex:
    return onp.linalg.det(self.matrix).item()

  def as_pair(self) -> tuple[NDArray, NDArray]:
    return self.matrix, self.translation

  def apply(self, points: NDArray) -> NDArray:
    """Apply the operation to row-vector points with final axis ``dim``."""

    dtype = onp.result_type(points, self.matrix, self.translation)
    points = onp.asarray(points, dtype=dtype)
    if points.shape[-1] != self.dim:
      raise ValueError(
        f"points must have final dimension {self.dim}, got {points.shape}"
      )
    return points @ self.matrix.T + self.translation

  def dual_matrix(self) -> NDArray:
    """Matrix for the contragredient action on reciprocal vectors."""
    return onp.linalg.inv(self.matrix).T

  def apply_dual(self, covectors: NDArray) -> NDArray:
    """Apply the dual action to row-vector reciprocal/covector data."""

    dtype = onp.result_type(covectors, self.matrix)
    covectors = onp.asarray(covectors, dtype=dtype)
    if covectors.shape[-1] != self.dim:
      raise ValueError(
        f"covectors must have final dimension {self.dim}, got {covectors.shape}"
      )
    return covectors @ self.dual_matrix().T

  def compose(self, other: AffineOperation, *, label: str | None = None) -> AffineOperation:
    """Return ``self after other`` as an affine operation."""

    if self.dim != other.dim:
      raise ValueError(f"dimension mismatch: {self.dim} != {other.dim}")
    matrix = self.matrix @ other.matrix
    translation = self.matrix @ other.translation + self.translation
    return AffineOperation(matrix, translation, self.label if label is None else label)

  def inverse(self, *, label: str | None = None) -> AffineOperation:
    matrix = onp.linalg.inv(self.matrix)
    translation = -matrix @ self.translation
    return AffineOperation(matrix, translation, self.label if label is None else label)


def as_affine_operation(op: AffineOperation | tuple[NDArray, NDArray]) -> AffineOperation:
  if isinstance(op, AffineOperation):
    return op
  matrix, translation = op
  return AffineOperation(matrix, translation)


@dcls.dataclass(frozen=True)
class FiniteGroupAction:
  """A finite group action represented by affine operations."""

  operations: Sequence[AffineOperation | tuple[NDArray, NDArray]]
  name: str = ""

  def __post_init__(self) -> None:
    operations = tuple(as_affine_operation(op) for op in self.operations)
    if not operations:
      raise ValueError("FiniteGroupAction requires at least one operation")

    dim = operations[0].dim
    if any(op.dim != dim for op in operations):
      raise ValueError("all operations must have the same dimension")

    object.__setattr__(self, "operations", operations)

  def __iter__(self) -> Iterator[AffineOperation]:
    return iter(self.operations)

  def __len__(self) -> int:
    return len(self.operations)

  @property
  def order(self) -> int:
    return len(self.operations)

  @property
  def dim(self) -> int:
    return self.operations[0].dim

  @property
  def matrices(self) -> NDArray:
    return onp.stack([op.matrix for op in self.operations], axis=0)

  @property
  def translations(self) -> NDArray:
    return onp.stack([op.translation for op in self.operations], axis=0)

  @property
  def characters(self) -> NDArray:
    return onp.array([op.character for op in self.operations])

  def as_pairs(self) -> list[tuple[NDArray, NDArray]]:
    return [op.as_pair() for op in self.operations]

  def orbit(self, points: NDArray, *, unique: bool = False, tol: float = 1e-10) -> NDArray:
    r""" Return the orbit of point(s) under the group action. """

    orb = onp.stack([op.apply(points) for op in self.operations], axis=0)
    if not unique:
      return orb

    if orb.ndim != 2:
      raise ValueError("unique=True is only supported for a single point")
    scaled = onp.round(orb / tol).astype(onp.int64)
    _, idx = onp.unique(scaled, axis=0, return_index=True)
    return orb[onp.sort(idx)]

  def reynolds_average(self, func, *xi: NDArray, domain: NDArray | None = None) -> NDArray:
    r""" Average a scalar function over the action, projecting to invariants. """
    return symmetrize(func, *xi, sym_ops=self, domain=domain)

  def linear_projector(
    self,
    character: Iterable[complex],
    *,
    irrep_dim: float | None = None,
  ) -> NDArray:
    r"""Projection matrix ``d/|G| sum_g conj(chi(g)) rho(g)``.

    This acts on the defining vector representation carried by
    ``self.matrices``.  For the trivial character this is the usual Reynolds
    projector onto invariant vectors.
    """

    chi = onp.asarray(list(character), dtype=complex)
    if chi.shape != (self.order,):
      raise ValueError(f"character must have shape {(self.order,)}, got {chi.shape}")

    dim = chi[0].real if irrep_dim is None else irrep_dim
    return dim / self.order * onp.einsum("g,gij->ij", onp.conjugate(chi), self.matrices)


# to handle numeric error near zero in generated group elements
def _snap(x, eps: float=1e-14):
  return onp.where(onp.abs(x) < eps, 0.0, x)

# TODO exploit properties of shift operator
# shift operator
def S(v: NDArray) -> NDArray:
  # (Sv)_n = v_{n-1}
  return onp.roll(v, 1)

def identity(dim: int = 2, *, inversion: bool = False, label: str = "e") -> AffineOperation:
  matrix = -onp.eye(dim) if inversion else onp.eye(dim)
  return AffineOperation(matrix, onp.zeros(dim), label)

def about(center: NDArray, op: AffineOperation):
  M, t = op.matrix, op.translation
  return AffineOperation(M, center - M @ center + t)


def rotation2d(theta: float, *, label: str | None = None) -> AffineOperation:
  return AffineOperation(_snap(gops.rot_mat(theta)), label=label or f"r({theta:g})")


def rotation3d(theta: float, axis: NDArray, *, label: str | None = None) -> AffineOperation:
  return AffineOperation(_snap(gops.q_rot_mat(theta, axis)), label=label or f"r({theta:g})")


def reflection(
  normal: NDArray,
  translation: NDArray | None = None,
  *,
  label: str = "s",
) -> AffineOperation:
  """
  R = I - 2nn^T
  
  Parameters:
  ----------------
  normal : arraylike
      The normal vector to the plane.
      
  Returns:
  --------
  R : NDArray, shape = (ndim, ndim)
  """
  normal = onp.asarray(normal, dtype=float)
  normal = normal / onp.linalg.norm(normal)
  matrix = onp.eye(len(normal)) - 2.0 * onp.outer(normal, normal)
  return AffineOperation(_snap(matrix), translation, label)


def glide_reflection(normal: NDArray, translation: NDArray, *, label: str = "g") -> AffineOperation:
  return reflection(normal, translation, label=label)

def screw_rotation(theta: float, axis: NDArray, translation: NDArray) -> AffineOperation:
  matrix = gops.q_rot_mat(theta, axis)
  translation = translation * axis
  return AffineOperation(_snap(matrix), translation)

def conjugate_action(action: FiniteGroupAction, op: AffineOperation):
  op_inv = op.inverse()
  return FiniteGroupAction(
    [op.compose(g).compose(op_inv) for g in action],
    name=action.name,
  )


def cyclic_group(
  order: int,
  *,
  dim: int = 2,
  axis: NDArray | None = None,
  name: str | None = None,
) -> FiniteGroupAction:
  if order <= 0:
    raise ValueError("order must be positive")
  if dim == 2:
    ops = [rotation2d(i * _2PI / order, label=f"r^{i}") for i in range(order)]
  elif dim == 3:
    if axis is None:
      raise ValueError("axis is required for a 3D cyclic group")
    ops = [rotation3d(i * _2PI / order, axis, label=f"r^{i}") for i in range(order)]
  else:
    raise ValueError(f"cyclic_group only supports dim=2 or dim=3, got {dim}")
  return FiniteGroupAction(ops, name=name or f"C{order}")


def dihedral_group(order: int, *, name: str | None = None) -> FiniteGroupAction:
  """Planar dihedral action generated by rotation and reflection across x-axis."""

  if order <= 0:
    raise ValueError("order must be positive")

  mirror = reflection(onp.array([0.0, 1.0]), label="s")
  ops: list[AffineOperation] = []
  for i in range(order):
    rot = rotation2d(i * _2PI / order, label=f"r^{i}")
    ops.append(rot)
  for i in range(order):
    rot = rotation2d(i * _2PI / order, label=f"r^{i}")
    ops.append(rot.compose(mirror, label=f"r^{i}s"))
  return FiniteGroupAction(ops, name=name or f"D{order}")


def tetrahedral_group(*, name: str | None = None, tol: float = 1e-10) -> FiniteGroupAction:
  r"""Return the proper rotational symmetry group of a tetrahedron (isomorphic to \(A_4\)).

  Implementation: close the group generated by a 3-fold rotation about the body-diagonal
  axis \((1,1,1)\) and a 2-fold rotation about the x-axis. This yields a 12-element
  subgroup of SO(3) (the chiral tetrahedral group).
  """

  axis_111 = onp.array([1.0, 1.0, 1.0])
  axis_111 = axis_111 / onp.linalg.norm(axis_111)
  a3 = rotation3d(_2PI / 3.0, axis_111, label="a3")  # order 3
  b2 = rotation3d(onp.pi, onp.array([1.0, 0.0, 0.0]), label="b2")  # order 2

  # A_4
  order = math.factorial(4)/2
  action = generated_group([a3, b2], name=name or "A4", tol=tol, max_order=order)
  if action.order != order:
    raise ValueError(f"tetrahedral_group closure produced order={action.order}, expected 12")
  return action




def generated_group(
  generators: Sequence[AffineOperation | tuple[NDArray, NDArray]],
  *,
  name: str = "",
  tol: float = 1e-10,
  max_order: int = 512,
) -> FiniteGroupAction:
  """Close a finite affine group generated by the supplied operations."""

  gens = tuple(as_affine_operation(g) for g in generators)
  if not gens:
    raise ValueError("at least one generator is required")
  dim = gens[0].dim
  if any(g.dim != dim for g in gens):
    raise ValueError("all generators must have the same dimension")

  def key(op: AffineOperation) -> tuple[float, ...]:
    data = onp.concatenate([op.matrix.ravel(), op.translation])
    return tuple(onp.round(data / tol).astype(onp.int64).tolist())

  ident = identity(dim)
  seen = {key(ident): ident}
  frontier = [ident]
  while frontier and ((current := frontier.pop()) or True):

  # while frontier:
  #   current = frontier.pop()
    for gen in gens:
      for candidate in (gen.compose(current), current.compose(gen)):
        candidate = candidate.replace(
          matrix=_snap(candidate.matrix), 
          translation=_snap(candidate.translation)
        )
        candidate_key = key(candidate)
        if candidate_key in seen:
          continue
        seen[candidate_key] = candidate
        frontier.append(candidate)
        if len(seen) > max_order:
          raise ValueError(
            "generated group exceeded max_order; check whether the generators "
            "produce an infinite affine group or raise max_order"
          )

  return FiniteGroupAction(tuple(seen.values()), name=name)




def apply_func(func, *xi, domain=None):
  """
  Parameters
  ----------------
  func, 

  *xi, 
  
  domain=None
  """
  if domain is None:
    # do not wrap if no domain
    return func(*xi)
  # if domain is None: domain = onp.array([(0,1) for _ in range(len(xi))])
  center = 0.5 * (domain[:,1] - domain[:,0])
  # modulo [0,1) domain, assuming that the coords were translated to be centred at zero
  xp = [(x + center[i]) % (2*center[i]) for i, x in enumerate(xi)]
  return func(*xp)

def _as_affine_pair(op: AffineOperation | tuple[NDArray, NDArray]) -> tuple[NDArray, NDArray]:
  if hasattr(op, "as_pair"):
    return op.as_pair()
  if hasattr(op, "matrix") and hasattr(op, "translation"):
    return op.matrix, op.translation
  return op

def symmetrize(func, *xi: NDArray, sym_ops, domain=None):
  """
  Parameters
  ----------------
  func, 
  
  *xi, 
  
  sym_ops, 
  
  domain=None
  """
  f_symm = onp.zeros_like(xi[0])
  for op in sym_ops:
    # get affine representation 
    M, t = _as_affine_pair(op)
    X_prime = onp.dot(M, onp.vstack([x.ravel() for x in xi])) + t[:, None]
    xp = [X_prime[i].reshape(x.shape) for i, x in enumerate(xi)]
    f_symm += apply_func(func, *xp, domain=domain)
  return f_symm/len(sym_ops)
