# src/grassmann.py
r"""Plucker coordinates, the Grassmannian, and how far a fermionic state is from one determinant.

A Slater determinant depends only on the span of its occupied orbitals, not on the orbitals: any
``GL(N)`` mixing of the occupied set leaves the determinant unchanged up to a scalar. So the set
of Slater determinants is not a vector space but the Grassmannian ``Gr(N, M)`` of N-planes in the
M-dimensional single-particle space, embedded in projective space by the Plucker map::

  span(v_1, ..., v_N)  |-->  [v_1 ^ ... ^ v_N]  in  P(Lambda^N C^M)

whose homogeneous coordinates are exactly the N-by-N minors, i.e. the CI coefficients of the state
in a given basis. The image is cut out by the quadratic Plucker relations, which turns a physical
question into an algebraic one:

  "Is this N-fermion state a single Slater determinant, or genuinely correlated?"

has the answer "iff the Plucker quadrics vanish on its coefficient vector". The smallest case is
``Gr(2, 4)``, one quadric in ``P^5``, the Klein quadric -- and that quadric is the 4-by-4
Pfaffian, so the two-fermion decomposability test and the Pfaffian of pairing theory are the same
polynomial.

Nothing here is dimension-specific; it is a statement about the single-particle space.
"""
from __future__ import annotations

import itertools as it
from typing import Sequence

import numpy as onp
import jax.numpy as jnp


# --------------------------------------------------------------------------- #
# Index bookkeeping
# --------------------------------------------------------------------------- #

def plucker_indices(n_orb: int, n_occ: int) -> list:
  r"""Increasing index tuples of length ``n_occ`` from ``range(n_orb)``, in lexicographic order.

  These label the Plucker coordinates, equivalently the occupied-orbital configurations of an
  ``n_occ``-fermion state in ``n_orb`` orbitals. There are ``binom(n_orb, n_occ)`` of them.
  """
  return list(it.combinations(range(n_orb), n_occ))


def sort_sign(seq: Sequence[int]):
  r"""Return ``(sorted_tuple, sign)`` for a sequence of distinct ints, or ``(None, 0)`` on repeats.

  Plucker coordinates are antisymmetric in their indices, so an unsorted index set is read off the
  sorted one up to the sign of the sorting permutation, and vanishes when an index repeats.
  """
  s = list(seq)
  if len(set(s)) != len(s):
    return None, 0
  sign = 1
  for i in range(len(s)):
    for j in range(i + 1, len(s)):
      if s[i] > s[j]:
        sign = -sign
  return tuple(sorted(s)), sign


# --------------------------------------------------------------------------- #
# The Plucker map
# --------------------------------------------------------------------------- #

def plucker_coordinates(C):
  r"""Plucker vector of the column span of ``C``, an ``(n_orb, n_occ)`` coefficient matrix.

  Entry ``I`` is the minor of the rows indexed by ``I``, ordered as ``plucker_indices``. Because
  the minors transform by ``det`` under ``GL(n_occ)`` acting on the columns, the resulting point
  of projective space depends only on the span -- which is the whole point.
  """
  C = jnp.asarray(C)
  n_orb, n_occ = C.shape
  idx = plucker_indices(n_orb, n_occ)
  return jnp.stack([jnp.linalg.det(C[list(I), :]) for I in idx])


def plucker_relations(p, n_orb: int, n_occ: int):
  r"""Residuals of the quadratic Plucker relations on a coordinate vector ``p``.

  For every ``I`` of size ``n_occ - 1`` and ``J`` of size ``n_occ + 1``::

    sum_k (-1)^k p_{I + j_k} p_{J - j_k}  =  0

  These generate the ideal of the Grassmannian. The returned vector is zero exactly when ``p`` is
  decomposable, i.e. when the state is a single Slater determinant. For ``(n_occ, n_orb) = (2, 4)``
  there is one relation and it is the Klein quadric.

  The ``(I, J)`` loop produces each relation several times, so the results are deduplicated up to
  an overall sign; for ``Gr(2, n)`` what survives is the classical set of ``binom(n, 4)``
  three-term relations.
  """
  p = jnp.asarray(p)
  symbolic = _relation_terms(n_orb, n_occ)
  if not symbolic:
    return jnp.zeros((0,), dtype=p.dtype)
  return jnp.stack([sum(c * p[a] * p[b] for c, a, b in terms) for terms in symbolic])


def _relation_terms(n_orb: int, n_occ: int) -> list:
  r"""Distinct Plucker relations as lists of ``(coefficient, index, index)`` triples.

  Each relation is a quadratic form in the coordinates; the pair of indices is unordered, so
  terms are canonicalized before deduplication and the whole relation is canonicalized up to an
  overall sign.
  """
  idx = plucker_indices(n_orb, n_occ)
  pos = {I: k for k, I in enumerate(idx)}

  seen = {}
  for I in it.combinations(range(n_orb), n_occ - 1):
    for J in it.combinations(range(n_orb), n_occ + 1):
      acc: dict = {}
      for k, j in enumerate(J):
        left, sgn = sort_sign(tuple(I) + (j,))
        if sgn == 0:
          continue
        right = tuple(x for x in J if x != j)
        a, b = sorted((pos[left], pos[right]))
        acc[(a, b)] = acc.get((a, b), 0) + (-1) ** k * sgn
      terms = tuple(sorted((c, a, b) for (a, b), c in acc.items() if c != 0))
      if not terms:
        continue
      negated = tuple(sorted((-c, a, b) for c, a, b in terms))
      seen.setdefault(min(terms, negated), terms)
  return [list(v) for _, v in sorted(seen.items())]


def decomposability_residual(p, n_orb: int, n_occ: int):
  r"""Scale-invariant distance from the Grassmannian: ``|relations| / |p|^2``.

  Zero iff the state is a single Slater determinant. The quadratic normalization makes it
  invariant under rescaling ``p``, so it is a function on projective space and comparable across
  states. Useful as a correlation measure: it answers "how far from mean field" in the one place
  where that question has an exact algebraic meaning.
  """
  p = jnp.asarray(p)
  rel = plucker_relations(p, n_orb, n_occ)
  return jnp.linalg.norm(rel) / jnp.sum(jnp.abs(p) ** 2)


# --------------------------------------------------------------------------- #
# Pfaffians and the two-fermion case
# --------------------------------------------------------------------------- #

def pfaffian(A):
  r"""Pfaffian of an even-dimensional antisymmetric matrix, by expansion along the first row.

  ``Pf(A)^2 = det(A)``. For a 4-by-4 matrix this is ``A01 A23 - A02 A13 + A03 A12``: the three
  perfect matchings of four points, the crossing one carrying the minus sign. That expression is
  also the Klein quadric, and also the pairing amplitude of a four-particle BCS state -- one
  polynomial wearing three hats.

  Recursive in Python over static shapes, so it jits; intended for small matrices.
  """
  A = jnp.asarray(A)
  n = A.shape[-1]
  if n % 2 != 0:
    raise ValueError(f"Pfaffian needs an even dimension, got {n}")
  if n == 0:
    return jnp.ones((), dtype=A.dtype)
  if n == 2:
    return A[0, 1]
  total = jnp.zeros((), dtype=A.dtype)
  rest = list(range(1, n))
  for k, j in enumerate(rest):
    minor_idx = [m for m in rest if m != j]
    sub = A[jnp.ix_(jnp.array(minor_idx), jnp.array(minor_idx))]
    total = total + ((-1) ** k) * A[0, j] * pfaffian(sub)
  return total


def klein_quadric(p):
  r"""``p12 p34 - p13 p24 + p14 p23`` on the six Plucker coordinates of ``Gr(2, 4)``.

  ``p`` is ordered as ``plucker_indices(4, 2)``, namely
  ``(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)``. Vanishes iff the two-fermion state in four
  orbitals is a single Slater determinant.
  """
  p = jnp.asarray(p)
  p01, p02, p03, p12, p13, p23 = [p[i] for i in range(6)]
  return p01 * p23 - p02 * p13 + p03 * p12


def coefficient_matrix(p, n_orb: int):
  r"""Antisymmetric ``(n_orb, n_orb)`` coefficient matrix of a two-fermion Plucker vector."""
  p = jnp.asarray(p)
  idx = plucker_indices(n_orb, 2)
  C = jnp.zeros((n_orb, n_orb), dtype=p.dtype)
  for k, (i, j) in enumerate(idx):
    C = C.at[i, j].set(p[k]).at[j, i].set(-p[k])
  return C


def slater_rank(p, n_orb: int, tol: float = 1e-10) -> int:
  r"""Half the rank of the two-fermion coefficient matrix: the number of occupied pair orbitals.

  A two-fermion state can always be brought to a sum of ``slater_rank`` disjoint Slater
  determinants (the Zumino / Slater decomposition). Rank 1 means a single determinant, so this is
  the integer-valued companion to ``decomposability_residual``.
  """
  C = onp.asarray(coefficient_matrix(p, n_orb))
  svals = onp.linalg.svd(C, compute_uv=False)
  return int(onp.sum(svals > tol)) // 2
