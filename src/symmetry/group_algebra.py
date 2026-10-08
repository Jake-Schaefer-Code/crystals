# src/symmetry/group_algebra.py
r"""The group algebra ``C[G]`` of a finite group, generic in the type of its elements.

Young symmetrizers, Reynolds projectors and random-walk distributions on a group are all elements
of ``C[G]``: formal combinations ``sum_g a_g g`` multiplied by extending the group law linearly,

  (sum_g a_g g)(sum_h b_h h) = sum_{g,h} a_g b_h (gh).

This module is the one place that product lives. ``GroupAlgebra[G]`` is parametrized by the element
type ``G`` and needs only a ``FiniteGroup[G]`` (see ``src.symmetry.group_laws``)::

  GroupAlgebra[Permutation].symmetric(4)              # C[S_4]
  GroupAlgebra[Permutation].of(permutation_group)     # C[H] for H <= S_n
  GroupAlgebra[AffineOperation].of(dihedral_group(4)) # C[D_4] from point-group operations

Subscripting is a type-level statement, as for ``list[int]``: the subscripted class builds the same
object, and the checker then knows that ``x[g]`` takes a ``G`` and ``x.items()`` yields ``(G, coeff)``.

An element is a sparse map from group elements to coefficients with zero coefficients pruned.
Coefficients are stored against ``group.key(g)``, a hashable canonical form, so groups of
float-valued elements (``AffineOperation``) work the same way as permutations; ``terms`` exposes that
key-to-coefficient map, and ``items()`` recovers the group elements. For permutations the key is the
permutation, so ``terms`` is the plain ``{permutation: coeff}`` dict that ``young`` returns, and
``GroupAlgebra.element`` lifts such a dict in.

Separating the *carrier* (sparse coefficients over a basis of group elements) from the *rule* (the
group's multiplication) also makes the regular representation immediate. In the basis ``{e_g}`` the
product is ``e_a e_b = e_{ab}``, so the structure tensor

  T[a, b, c] = 1 if c = ab else 0

is a 0/1 array, and ``left_regular(x)`` is the matrix of left multiplication by ``x``. That
representation is faithful of dimension ``|G|`` and contains every irreducible with multiplicity
equal to its dimension. Any other representation ``rho`` extends linearly: ``x.represent(rho)``
is ``sum_g a_g rho(g)``.

Coefficients may be any Python number: ``int`` and ``Fraction`` keep results exact (dividing an
integer element by an integer yields ``Fraction`` coefficients), which is what idempotency checks
want. Zero pruning is an exact ``== 0`` test, so with ``float`` coefficients use ``allclose``
rather than ``==`` to compare results.

Conventions follow ``src.symmetry.permutations``: ``compose(p, q)[i] = p[q[i]]``.
"""
from __future__ import annotations

import dataclasses as dcls
import numbers
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping
from fractions import Fraction
from typing import Any, overload, Self

import numpy as onp

import src.symmetry.permutations as perms
from src.symmetry.group_laws import FiniteGroup, PermutationLaw, as_law
from src.symmetry.symmetry import AffineOperation, FiniteGroupAction

Permutation = perms.Permutation


@dcls.dataclass(frozen=True, eq=False)
class GroupAlgebra[G]:
  r"""``C[G]`` for a finite group whose elements have type ``G``."""
  group: FiniteGroup[G]
  # key -> a representative element, so that stored keys can be turned back into elements to compose
  _reps: dict = dcls.field(default_factory=dict, init=False, repr=False, compare=False)

  # -- constructors ------------------------------------------------------- #

  @classmethod
  def symmetric(cls, n: int) -> GroupAlgebra[Permutation]:
    r"""``C[S_n]``; no permutations are enumerated until ``basis`` is read."""
    return cls(PermutationLaw.symmetric(n))  # type: ignore[arg-type, return-value]

  @overload
  @classmethod
  def of(cls, group: perms.PermutationGroup) -> GroupAlgebra[Permutation]: ...
  @overload
  @classmethod
  def of(cls, group: FiniteGroupAction) -> GroupAlgebra[AffineOperation]: ...
  @overload
  @classmethod
  def of(cls, group: FiniteGroup[G]) -> GroupAlgebra[G]: ...
  @classmethod
  def of(cls, group) -> Self:  # pyright: ignore[reportInconsistentOverload]
    r"""Algebra of a ``PermutationGroup``, a ``FiniteGroupAction``, or any ``FiniteGroup``."""
    return cls(as_law(group))

  # -- the group ---------------------------------------------------------- #

  @property
  def order(self) -> int:
    return len(self.group.elements)

  @property
  def basis(self) -> tuple[G, ...]:
    r"""The group elements in a fixed order; this indexes ``structure_tensor`` and ``left_regular``."""
    return tuple(self.group.elements)

  def __eq__(self, other) -> bool:
    return isinstance(other, GroupAlgebra) and (
      self is other or self.group.signature == other.group.signature
    )

  def __hash__(self) -> int:
    return hash(self.group.signature)

  def _intern(self, g: G) -> Hashable:
    r"""Key of ``g``, remembering a representative so the key can be turned back into an element."""
    k = self.group.key(g)
    if k not in self._reps:
      self._reps[k] = g
    return k

  def _check(self, g: G) -> Hashable:
    if not self.group.contains(g):
      raise ValueError(f"{g!r} is not an element of the group")
    return self._intern(g)

  def _element_of(self, k: Hashable) -> G:
    return self._reps[k]

  def _require(self, x: GroupAlgebraElement[G]) -> None:
    if x.algebra is not self and x.algebra != self:
      raise ValueError("element belongs to a different group algebra")

  # -- construction of elements ------------------------------------------- #

  def element(self, terms: Mapping[G, Any] | Iterable[tuple[G, Any]] = ()) -> GroupAlgebraElement[G]:
    r"""Build an element from ``{g: coeff}`` or an iterable of ``(g, coeff)`` pairs.

    Group elements that are not hashable, such as ``AffineOperation``, must come as pairs.
    """
    pairs = terms.items() if isinstance(terms, Mapping) else terms
    out: dict = {}
    for g, c in pairs:
      k = self._check(g)
      out[k] = out[k] + c if k in out else c
    return GroupAlgebraElement(self, _prune(out))

  def zero(self) -> GroupAlgebraElement[G]:
    return GroupAlgebraElement(self, {})

  def unit(self) -> GroupAlgebraElement[G]:
    return GroupAlgebraElement(self, {self._intern(self.group.identity()): 1})

  def basis_element(self, g: G) -> GroupAlgebraElement[G]:
    return GroupAlgebraElement(self, {self._check(g): 1})

  # -- regular representation -------------------------------------------- #

  def structure_tensor(self) -> onp.ndarray:
    r"""``T[a, b, c] = 1`` iff ``basis[c] = basis[a] basis[b]``; an ``(N, N, N)`` int8 array.

    Dense and ``|G|^3`` in size (``S_5`` is 1.7 MB, ``S_6`` is 373 MB), so keep to small groups;
    ``left_regular`` is the ``|G|^2`` route for a single element.
    """
    group, basis = self.group, self.basis
    index = {group.key(g): i for i, g in enumerate(basis)}
    N = len(basis)
    T = onp.zeros((N, N, N), dtype=onp.int8)
    for a, g in enumerate(basis):
      for b, h in enumerate(basis):
        T[a, b, index[group.key(group.compose(g, h))]] = 1
    return T

  def left_regular(self, x: GroupAlgebraElement[G]) -> onp.ndarray:
    r"""Matrix of ``y -> x y`` in the basis ``self.basis``, so ``L(xy) = L(x) L(y)``.

    Column ``j`` holds the coordinates of ``x * basis[j]``. The dtype follows the coefficients and
    is ``object`` for ``Fraction``, keeping exactness through matrix products.
    """
    self._require(x)
    group, basis = self.group, self.basis
    index = {group.key(g): i for i, g in enumerate(basis)}
    N = len(basis)
    dtype = onp.asarray([c for c in x.terms.values()] or [0]).dtype
    M = onp.zeros((N, N), dtype=dtype)
    for j, h in enumerate(basis):
      for g, c in x.items():
        M[index[group.key(group.compose(g, h))], j] += c
    return M


@dcls.dataclass(frozen=True, eq=False)
class GroupAlgebraElement[G]:
  r"""An immutable element ``sum_g a_g g`` of a ``GroupAlgebra[G]``.

  ``terms`` maps ``group.key(g)`` to a nonzero coefficient.
  """
  algebra: GroupAlgebra[G]
  terms: dict

  # -- container protocol ------------------------------------------------- #

  def __getitem__(self, g: G):
    return self.terms.get(self.algebra.group.key(g), 0)

  def __iter__(self) -> Iterator[G]:
    return (self.algebra._element_of(k) for k in self.terms)

  def __len__(self) -> int:
    return len(self.terms)

  def __bool__(self) -> bool:
    return bool(self.terms)

  def items(self) -> Iterator[tuple[G, Any]]:
    r"""``(group element, coefficient)`` pairs."""
    return ((self.algebra._element_of(k), c) for k, c in self.terms.items())

  def __repr__(self) -> str:
    if not self.terms:
      return "0"
    return " + ".join(f"{c}*{k}" for k, c in sorted(self.terms.items(), key=repr))

  # -- ring structure ------------------------------------------------------ #

  def _lift(self, other) -> GroupAlgebraElement[G] | None:
    if isinstance(other, GroupAlgebraElement):
      self.algebra._require(other)
      return other
    return None

  def _with(self, terms: dict) -> GroupAlgebraElement[G]:
    return GroupAlgebraElement(self.algebra, _prune(terms))

  def __eq__(self, other) -> bool:
    if not isinstance(other, GroupAlgebraElement):
      return NotImplemented
    return (other.algebra is self.algebra or other.algebra == self.algebra) and self.terms == other.terms

  __hash__ = None  # mutable dict payload; not hashable  # pyright: ignore[reportAssignmentType]

  def __neg__(self) -> GroupAlgebraElement[G]:
    return GroupAlgebraElement(self.algebra, {k: -c for k, c in self.terms.items()})

  def __add__(self, other):
    other = self._lift(other)
    if other is None:
      return NotImplemented
    out = dict(self.terms)
    for k, c in other.terms.items():
      out[k] = out[k] + c if k in out else c
    return self._with(out)

  def __sub__(self, other):
    other = self._lift(other)
    if other is None:
      return NotImplemented
    return self + (-other)

  def __mul__(self, other):
    if isinstance(other, numbers.Number):
      return self._with({k: c * other for k, c in self.terms.items()})
    other = self._lift(other)
    if other is None:
      return NotImplemented
    A = self.algebra
    out: dict = {}
    for g, a in self.items():
      for h, b in other.items():
        k = A._intern(A.group.compose(g, h))
        out[k] = out[k] + a * b if k in out else a * b
    return self._with(out)

  def __rmul__(self, other):
    if isinstance(other, numbers.Number):
      return self * other
    return NotImplemented

  def __truediv__(self, other):
    if isinstance(other, numbers.Number):
      return self._with({k: _div(c, other) for k, c in self.terms.items()})
    return NotImplemented

  def __pow__(self, n: int):
    if not isinstance(n, int) or n < 0:
      raise ValueError("exponent must be a non-negative integer")
    out, base = self.algebra.unit(), self
    while n:
      if n & 1:
        out = out * base
      n >>= 1
      if n:
        base = base * base
    return out

  # -- the involution, representations and predicates ---------------------- #

  def star(self) -> GroupAlgebraElement[G]:
    r"""The anti-involution ``(sum a_g g)* = sum conj(a_g) g^{-1}``, so ``(xy)* = y* x*``."""
    A = self.algebra
    return GroupAlgebraElement(
      A, {A._intern(A.group.inverse(g)): c.conjugate() for g, c in self.items()}
    )

  def represent(self, rho: Callable[[G], Any]) -> onp.ndarray:
    r"""Image ``sum_g a_g rho(g)`` under the linear extension of a representation ``rho``.

    ``rho`` maps a group element to a matrix (or any array); the result is multiplicative in the
    algebra whenever ``rho`` is a homomorphism. ``Fraction`` coefficients are converted to ``float``
    against a floating-point ``rho`` and stay exact against an integer or object one. With ``rho = lambda g: g.matrix`` on a point group
    and the element ``(1/|G|) sum g`` this is the Reynolds projector onto invariant vectors.
    """
    total = None
    for g, c in self.items():
      m = onp.asarray(rho(g))
      if isinstance(c, Fraction) and m.dtype.kind in "fc":
        c = float(c)  # exact coefficient, floating-point representation: avoid an object array
      term = c * m
      total = term if total is None else total + term
    if total is None:
      raise ValueError(
        "cannot represent the zero element without knowing the dimension; "
        "apply rho to the unit and scale instead"
      )
    return total

  def allclose(self, other: GroupAlgebraElement[G], atol: float = 1e-12) -> bool:
    other = self._lift(other)
    keys = set(self.terms) | set(other.terms)
    return all(abs(self.terms.get(k, 0) - other.terms.get(k, 0)) <= atol for k in keys)

  def is_idempotent(self, atol: float | None = None) -> bool:
    r"""``x * x == x``; exact unless ``atol`` is given."""
    sq = self * self
    return sq == self if atol is None else sq.allclose(self, atol)

  def trace(self):
    r"""Coefficient of the identity, which is ``tr(L(x)) / |G|`` in the regular representation."""
    return self[self.algebra.group.identity()]


def _div(c, k):
  r""" True division that stays exact for integer data: ``int / int`` gives a ``Fraction``. """
  if isinstance(c, numbers.Integral) and isinstance(k, numbers.Integral):
    return Fraction(c, k)
  return c / k


def _prune(d: dict) -> dict:
  return {k: c for k, c in d.items() if c != 0}
