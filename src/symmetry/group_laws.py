# src/symmetry/group_laws.py
r"""Finite groups as a *law*: identity, composition, inverse and a canonical hashable key.

``GroupAlgebra[G]`` is generic in the type ``G`` of group elements and needs nothing more of a group
than the ``FiniteGroup[G]`` protocol below. Two implementations ship here:

- ``PermutationLaw`` for ``Permutation = tuple[int, ...]``, either all of ``S_n`` (enumerated only
  on demand) or an explicit ``PermutationGroup``;
- ``AffineLaw`` for ``AffineOperation``, i.e. point and space-group operations ``x -> Mx + t``.

The ``key`` is what makes this work for float-valued elements: an ``AffineOperation`` holds numpy
arrays and so is neither hashable nor exactly comparable, so group-algebra coefficients are stored
against ``key(g)``, a tuple of integers obtained by rounding on a tolerance grid, exactly as
``symmetry.generated_group`` already does. For permutations the key is the permutation itself.

Composition follows ``permutations.compose``: ``compose(a, b)`` is "``a`` after ``b``", so
``compose(p, q)[i] = p[q[i]]`` and ``AffineOperation.compose`` agree.
"""
from __future__ import annotations

import dataclasses as dcls
import functools
import itertools as it
from collections.abc import Hashable, Sequence
from typing import Protocol

import numpy as onp

import src.symmetry.permutations as perms
from src.symmetry.symmetry import AffineOperation, FiniteGroupAction, _snap

Permutation = perms.Permutation


class FiniteGroup[G](Protocol):
  r"""A finite group of elements of type ``G``, given by its law.

  ``key`` must be a faithful hashable fingerprint: ``key(a) == key(b)`` iff ``a`` and ``b`` are the
  same group element. ``signature`` identifies the group itself, so that two algebras built from
  equal groups are interchangeable.
  """

  @property
  def elements(self) -> Sequence[G]: ...
  @property
  def signature(self) -> Hashable: ...
  def identity(self) -> G: ...
  def compose(self, a: G, b: G) -> G: ...
  def inverse(self, a: G) -> G: ...
  def key(self, a: G) -> Hashable: ...
  def contains(self, a: G) -> bool: ...


def _factorial(n: int) -> int:
  out = 1
  for k in range(2, n + 1):
    out *= k
  return out


# --------------------------------------------------------------------------- #
# Permutations
# --------------------------------------------------------------------------- #

@dcls.dataclass(frozen=True, eq=False)
class PermutationLaw:
  r"""``S_n`` (``group=None``) or an explicit ``PermutationGroup``, acting on ``range(degree)``.

  With ``group=None`` nothing is enumerated until ``elements`` is read, so symmetrizers in ``S_8``
  need no listing of 40320 permutations.
  """
  degree: int
  group: perms.PermutationGroup | None = None

  @classmethod
  def symmetric(cls, n: int) -> "PermutationLaw":
    return cls(n, None)

  @classmethod
  def of(cls, group: perms.PermutationGroup) -> "PermutationLaw":
    return cls(group.degree, group)

  @functools.cached_property
  def _members(self) -> frozenset | None:
    return None if self.group is None else frozenset(self.group.elements)

  @functools.cached_property
  def elements(self) -> tuple[Permutation, ...]:
    if self.group is not None:
      return tuple(self.group.elements)
    return tuple(it.permutations(range(self.degree)))

  @property
  def order(self) -> int:
    return len(self.group) if self.group is not None else _factorial(self.degree)

  @property
  def signature(self) -> Hashable:
    return ("perm", self.degree, self._members)

  def identity(self) -> Permutation:
    return perms.identity_perm(self.degree)

  def compose(self, a: Permutation, b: Permutation) -> Permutation:
    return perms.compose_perm(a, b)

  def inverse(self, a: Permutation) -> Permutation:
    return perms.inverse_perm(a)

  def key(self, a: Permutation) -> Permutation:
    return tuple(a)

  def contains(self, a: Permutation) -> bool:
    a = tuple(a)
    if self._members is not None:
      return a in self._members
    return len(a) == self.degree and sorted(a) == list(range(self.degree))

  def __eq__(self, other) -> bool:
    return isinstance(other, PermutationLaw) and self.signature == other.signature

  def __hash__(self) -> int:
    return hash(self.signature)


# --------------------------------------------------------------------------- #
# Affine operations (point groups and space-group operations)
# --------------------------------------------------------------------------- #

@dcls.dataclass(frozen=True, eq=False)
class AffineLaw:
  r"""A finite group of ``AffineOperation`` closed under composition up to ``tol``.

  Closure and uniqueness are verified on construction (``O(|G|^2)`` compositions), so an action
  that is secretly infinite, such as a glide with no finite quotient, is rejected rather than
  silently producing a table with holes.
  """
  operations: tuple[AffineOperation, ...]
  tol: float = 1e-10

  @classmethod
  def of(cls, action: FiniteGroupAction, *, tol: float = 1e-10) -> "AffineLaw":
    return cls(tuple(action.operations), tol)

  def __post_init__(self) -> None:
    ops = tuple(self.operations)
    if not ops:
      raise ValueError("a group needs at least one operation")
    object.__setattr__(self, "operations", ops)
    keys = [self.key(op) for op in ops]
    if len(set(keys)) != len(keys):
      raise ValueError("operations contain duplicates up to tol")
    members = set(keys)
    if self.key(self.identity()) not in members:
      raise ValueError("the identity is not among the operations")
    for a in ops:
      if self.key(self.inverse(a)) not in members:
        raise ValueError("operations are not closed under inverse")
      for b in ops:
        if self.key(self.compose(a, b)) not in members:
          raise ValueError("operations are not closed under composition")

  @property
  def dim(self) -> int:
    return self.operations[0].dim

  @property
  def elements(self) -> tuple[AffineOperation, ...]:
    return self.operations

  @property
  def order(self) -> int:
    return len(self.operations)

  @functools.cached_property
  def _members(self) -> frozenset:
    return frozenset(self.key(op) for op in self.operations)

  @property
  def signature(self) -> Hashable:
    return ("affine", self.tol, self._members)

  def identity(self) -> AffineOperation:
    return AffineOperation(onp.eye(self.dim), onp.zeros(self.dim), "e")

  def compose(self, a: AffineOperation, b: AffineOperation) -> AffineOperation:
    out = a.compose(b)
    return out.replace(matrix=_snap(out.matrix), translation=_snap(out.translation))

  def inverse(self, a: AffineOperation) -> AffineOperation:
    out = a.inverse()
    return out.replace(matrix=_snap(out.matrix), translation=_snap(out.translation))

  def key(self, a: AffineOperation) -> tuple[int, ...]:
    data = onp.concatenate([a.matrix.ravel(), a.transl])
    packed = onp.concatenate([data.real, data.imag])
    return tuple(onp.round(packed / self.tol).astype(onp.int64).tolist())

  def contains(self, a: AffineOperation) -> bool:
    return a.dim == self.dim and self.key(a) in self._members

  def __eq__(self, other) -> bool:
    return isinstance(other, AffineLaw) and self.signature == other.signature

  def __hash__(self) -> int:
    return hash(self.signature)


def as_law(group) -> FiniteGroup:
  r"""Adapt the group types in this package to a ``FiniteGroup``; laws pass through unchanged."""
  if isinstance(group, perms.PermutationGroup):
    return PermutationLaw.of(group)
  if isinstance(group, FiniteGroupAction):
    return AffineLaw.of(group)
  return group
