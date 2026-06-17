# src/symmetry.py
"""Finite symmetry actions with representation-theory oriented helpers.

The existing notebooks mostly represent symmetry operations as ``(M, t)``
pairs acting by ``x -> M x + t``.  This module keeps that convention while
adding a small amount of structure: named affine operations, finite group
actions, orbits, characters, and Reynbolds/projector helpers.
"""

from __future__ import annotations
from typing import TypeVar, Generic, Protocol
from collections.abc import Iterable, Iterator, Sequence
import dataclasses as dcls
import numpy as onp
from src.core.static_types import ReplaceMixin
from src.symmetry.core import Field, F

NDArray = onp.ndarray
_2PI = 2.0 * onp.pi


def recip_lattice(A: NDArray) -> NDArray:
  r"""
  Reciprocal lattice:
    b_i . a_j = 2 pi delta_ij
  """
  return _2PI * onp.linalg.inv(A).T # columns are b1, b2


# TODO
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


# TODO
@dcls.dataclass(frozen=True)
class SeitzOp:
  R: NDArray   # integer/rational in lattice basis when possible
  tau: NDArray # fractional translation
  label: str = ""
  def __iter__(self) -> Iterator[NDArray]:
    yield self.R
    yield self.tau


# TODO
@dcls.dataclass(frozen=True)
class PointGroup:
  elements: tuple[NDArray, ...]
  ...


# TODO
@dcls.dataclass(frozen=True)
class SpaceGroup:
  lattice: Lattice
  point_group: PointGroup
  coset_reps: tuple[SeitzOp, ...]   # or full generators + closure logic
  ...


# TODO
@dcls.dataclass(frozen=True)
class Site:
  frac: NDArray
  species: str
  dof: object | None = None


# TODO
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
      _chi = [chi[element] for element in self.elements]
    else:
      _chi = chi

    chi = onp.asarray(list(_chi), dtype=complex)
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


# TODO
@dcls.dataclass(frozen=True)
class CharacterTable:
  class_order: tuple[object, ...]
  class_sizes: dict[object, int]
  characters: dict[object, NDArray]   # maps row label -> values on conjugacy classes
  group: PointGroup | SpaceGroup | object | None = None
  name: str = ""

  def __post_init__(self) -> None:
    if not self.class_order:
      raise ValueError("CharacterTable requires a nonempty class_order")

    missing = [ct for ct in self.class_order if ct not in self.class_sizes]
    if missing:
      raise ValueError(f"class_sizes is missing entries for {missing!r}")

    normalized: dict[object, NDArray] = {}
    for label, row in self.characters.items():
      arr = onp.asarray(row, dtype=complex)
      if arr.shape != (len(self.class_order),):
        raise ValueError(
          "character rows must have shape "
          f"{(len(self.class_order),)}, got {arr.shape} for {label!r}"
        )
      normalized[label] = arr

    object.__setattr__(self, "class_order", tuple(self.class_order))
    object.__setattr__(self, "characters", normalized)

  @classmethod
  def from_rows(
    cls,
    *,
    class_order: Sequence[object],
    class_sizes: dict[object, int],
    row_labels: Sequence[object],
    rows: Sequence[Sequence[complex]],
    group: PointGroup | SpaceGroup | object | None = None,
    name: str = "",
  ) -> "CharacterTable":
    characters = {
      label: onp.asarray(row, dtype=complex)
      for label, row in zip(row_labels, rows, strict=True)
    }
    return cls(
      class_order=tuple(class_order),
      class_sizes=dict(class_sizes),
      characters=characters,
      group=group,
      name=name,
    )

  @property
  def order(self) -> int:
    return sum(self.class_sizes[ct] for ct in self.class_order)

  @property
  def n_classes(self) -> int:
    return len(self.class_order)

  @property
  def row_labels(self) -> tuple[object, ...]:
    return tuple(self.characters.keys())

  @property
  def weights(self) -> NDArray:
    return onp.asarray([self.class_sizes[ct] for ct in self.class_order], dtype=float) / self.order

  def __getitem__(self, label: object) -> NDArray:
    return self.characters[label]

  def matrix(self, labels: Sequence[object] | None = None) -> NDArray:
    labels = self.row_labels if labels is None else tuple(labels)
    if not labels:
      return onp.zeros((0, self.n_classes), dtype=complex)
    return onp.stack([self.characters[label] for label in labels], axis=0)

  def weighted_matrix(self, labels: Sequence[object] | None = None) -> NDArray:
    return onp.conjugate(self.matrix(labels)) * self.weights[None, :]

  def as_dict(self, label: object) -> dict[object, complex]:
    row = self[label]
    return {ct: row[i] for i, ct in enumerate(self.class_order)}

  def class_function(self, values: Sequence[complex] | dict[object, complex]) -> dict[object, complex]:
    row = self._coerce_row(values)
    return {ct: row[i] for i, ct in enumerate(self.class_order)}

  def inner_product(
    self,
    chi: object | Sequence[complex] | dict[object, complex],
    psi: object | Sequence[complex] | dict[object, complex],
  ) -> complex:
    u = self._coerce_row(chi)
    v = self._coerce_row(psi)
    return onp.dot(self.weights, onp.conjugate(u) * v)

  def norm(self, chi: object | Sequence[complex] | dict[object, complex]) -> complex:
    return self.inner_product(chi, chi)

  def decompose(
    self,
    chi: object | Sequence[complex] | dict[object, complex],
    *,
    labels: Sequence[object] | None = None,
  ) -> dict[object, complex]:
    labels = self.row_labels if labels is None else tuple(labels)
    return {label: self.inner_product(chi, label) for label in labels}

  def orthogonal_complement(
    self,
    labels: Sequence[object] | None = None,
    *,
    tol: float = 1e-10,
  ) -> tuple[NDArray, ...]:
    r"""Basis for the weighted orthogonal complement of the selected rows."""
    A = self.weighted_matrix(labels)
    if A.size == 0:
      return tuple(onp.eye(self.n_classes, dtype=complex)[i] for i in range(self.n_classes))

    _, s, vh = onp.linalg.svd(A, full_matrices=True)
    rank = int(onp.sum(s > tol))
    basis = vh[rank:]
    return tuple(onp.asarray(row, dtype=complex) for row in basis)

  def with_character(self, label: object, row: Sequence[complex]) -> "CharacterTable":
    updated = dict(self.characters)
    updated[label] = onp.asarray(row, dtype=complex)
    return CharacterTable(
      class_order=self.class_order,
      class_sizes=dict(self.class_sizes),
      characters=updated,
      group=self.group,
      name=self.name,
    )

  def _coerce_row(self, chi: object | Sequence[complex] | dict[object, complex]) -> NDArray:
    if isinstance(chi, dict):
      missing = [ct for ct in self.class_order if ct not in chi]
      if missing:
        raise ValueError(f"class function is missing values for {missing!r}")
      return onp.asarray([chi[ct] for ct in self.class_order], dtype=complex)

    try:
      return self.characters[chi]
    except KeyError:
      pass
    except TypeError:
      pass

    arr = onp.asarray(chi, dtype=complex)
    if arr.shape != (self.n_classes,):
      raise ValueError(f"class function must have shape {(self.n_classes,)}, got {arr.shape}")
    return arr



@dcls.dataclass(frozen=True)
class AffineOperation(ReplaceMixin):
  """An affine symmetry operation ``x -> matrix @ x + translation``."""

  matrix: NDArray
  # TODO fix
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
    object.__setattr__(self, "_transl", translation)

  def __iter__(self) -> Iterator[NDArray]:
    yield self.matrix
    yield self.transl

  @property
  def transl(self) -> NDArray:
    return self._transl


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
    return self.matrix, self.transl

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
    translation = - matrix @ self.transl
    return AffineOperation(matrix, translation, self.label if label is None else label)


def as_affine_operation(op: AffineOperation | tuple[NDArray, NDArray]) -> AffineOperation:
  if isinstance(op, AffineOperation):
    return op
  matrix, translation = op
  return AffineOperation(matrix, translation)


@dcls.dataclass(frozen=True)
class FiniteGroupAction:
  """A finite group action represented by affine operations."""

  operations: Sequence[AffineOperation]
  # operations: Sequence[AffineOperation | tuple[NDArray, NDArray]]
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
    return onp.stack([op.transl for op in self.operations], axis=0)

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
    data = onp.concatenate([op.matrix.ravel(), op.transl])
    packed = onp.concatenate([data.real, data.imag])
    return tuple(onp.round(packed / tol).astype(onp.int64).tolist())

  from src.symmetry.operations import identity
  ident = identity(dim)
  seen = {key(ident): ident}
  frontier = [ident]
  while frontier and ((current := frontier.pop()) or True):
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
    return op.matrix, op.transl
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


class GaloisGroup(FiniteGroupAction):
  pass

Gal = GaloisGroup

