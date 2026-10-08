# src/symmetry/young.py
r"""Young tableaux, symmetrizers and characters of the symmetric group.

Partitions of n index both the irreducible representations of S_n and the irreducible polynomial
representations of GL, and the dictionary between them is what makes antisymmetry tractable:

- ``(1^n)`` is the sign representation, whose isotypic component in the polynomial ring is the
  Vandermonde times the invariants. That is the content of ``src.invariants``.
- For spin-1/2 fermions the total state is antisymmetric but the spatial factor is not; it carries
  the irrep conjugate to the spin irrep, and for spin 1/2 only partitions with at most two columns
  occur. A Young symmetrizer projects onto that component, which gives a spin-adapted spatial
  wavefunction of definite total S with the spin coordinates integrated out entirely.

Characters come from the Murnaghan-Nakayama rule on beta-numbers rather than from building matrix
representations, so the character table of S_n is reachable well past the point where the regular
representation is.

Conventions follow ``src.symmetry.permutations``: a permutation is a ``tuple[int, ...]`` with
``p[i] = g(i)`` on ``{0, ..., n-1}``, composition is ``compose(p, q)[i] = p[q[i]]``, and a
partition is a weakly decreasing ``tuple[int, ...]`` of positive parts. A tableau is a tuple of
rows, each a tuple of entries drawn from ``{0, ..., n-1}``.
"""
from __future__ import annotations

import functools
import itertools as it
import math
from fractions import Fraction
from typing import Iterator, Sequence

import numpy as onp

import src.symmetry.permutations as perms

Partition = tuple
Tableau = tuple


# --------------------------------------------------------------------------- #
# Partitions
# --------------------------------------------------------------------------- #

def partitions(n: int, max_part: int | None = None) -> Iterator[Partition]:
  r"""Generate the partitions of n in reverse lexicographic order."""
  if n == 0:
    yield ()
    return
  top = n if max_part is None else min(n, max_part)
  for first in range(top, 0, -1):
    for rest in partitions(n - first, first):
      yield (first,) + rest


def conjugate(lam: Sequence[int]) -> Partition:
  r"""The conjugate (transposed) partition."""
  if not lam:
    return ()
  return tuple(sum(1 for part in lam if part > j) for j in range(lam[0]))


def young_diagram(lam: Sequence[int]) -> tuple:
  r"""Cells of the diagram as ``(row, column)`` pairs, both zero-based."""
  return tuple((i, j) for i, part in enumerate(lam) for j in range(part))


def hook_lengths(lam: Sequence[int]) -> dict:
  r"""Hook length of each cell: arm + leg + 1."""
  conj = conjugate(lam)
  return {
    (i, j): (lam[i] - j - 1) + (conj[j] - i - 1) + 1
    for i, part in enumerate(lam) for j in range(part)
  }


def dimension(lam: Sequence[int]) -> int:
  r"""Dimension of the S_n irrep indexed by ``lam``, by the hook length formula.

  ``dim = n! / prod(hooks)``, which also counts the standard Young tableaux of shape ``lam``.
  """
  n = sum(lam)
  prod = 1
  for h in hook_lengths(lam).values():
    prod *= h
  return math.factorial(n) // prod


def standard_tableaux(lam: Sequence[int]) -> Iterator[Tableau]:
  r"""Generate the standard Young tableaux of shape ``lam``, entries ``0..n-1``.

  Standard means strictly increasing along rows and down columns. Entries are placed in order,
  each into the first available corner, so the count matches ``dimension(lam)``.
  """
  lam = tuple(lam)
  n = sum(lam)
  rows = [[] for _ in lam]

  def place(k):
    if k == n:
      yield tuple(tuple(r) for r in rows)
      return
    for i, part in enumerate(lam):
      if len(rows[i]) < part and (i == 0 or len(rows[i - 1]) > len(rows[i])):
        rows[i].append(k)
        yield from place(k + 1)
        rows[i].pop()

  yield from place(0)


def first_tableau(lam: Sequence[int]) -> Tableau:
  r"""The row-reading standard tableau: ``0..lam_0-1`` in row 0, then the next row, and so on."""
  out, k = [], 0
  for part in lam:
    out.append(tuple(range(k, k + part)))
    k += part
  return tuple(out)


# --------------------------------------------------------------------------- #
# Young symmetrizers
# --------------------------------------------------------------------------- #

def _subgroup_from_blocks(blocks, n: int) -> list:
  r"""All permutations of ``0..n-1`` permuting each block among itself."""
  out = []
  per_block = [list(it.permutations(b)) for b in blocks if len(b) > 1]
  fixed = [b for b in blocks if len(b) > 1]
  for choice in it.product(*per_block) if per_block else [()]:
    image = list(range(n))
    for block, images in zip(fixed, choice):
      for src_i, dst in zip(block, images):
        image[src_i] = dst
    out.append(tuple(image))
  return out


def row_group(tab: Tableau, n: int) -> list:
  r"""The row stabilizer of a tableau: permutations preserving each row setwise."""
  return _subgroup_from_blocks([tuple(r) for r in tab], n)


def column_group(tab: Tableau, n: int) -> list:
  r"""The column stabilizer of a tableau: permutations preserving each column setwise."""
  width = max((len(r) for r in tab), default=0)
  cols = [tuple(r[j] for r in tab if len(r) > j) for j in range(width)]
  return _subgroup_from_blocks(cols, n)


def young_symmetrizer(tab: Tableau, n: int | None = None) -> dict:
  r"""The Young symmetrizer ``c = sum_{q in C} sum_{p in R} sgn(q) q p`` as a group-algebra element.

  Returned as a dict from permutation to integer coefficient. ``c`` is idempotent up to the scalar
  ``n! / dim(lam)``; use ``normalized_symmetrizer`` for the honest projector.

  For ``lam = (1^n)`` the row group is trivial and this is the full antisymmetrizer; for
  ``lam = (n)`` the column group is trivial and this is the symmetrizer. Everything in between is
  a mixed-symmetry projector, which is what a spin-adapted spatial wavefunction needs.
  """
  if n is None:
    n = sum(len(r) for r in tab)
  out: dict = {}
  rows = row_group(tab, n)
  cols = column_group(tab, n)
  for q in cols:
    sq = perms.sgn(q)
    for p in rows:
      g = perms.compose_perm(q, p)
      out[g] = out.get(g, 0) + sq
  return {g: c for g, c in out.items() if c != 0}


def normalized_symmetrizer(tab: Tableau, lam: Sequence[int], n: int | None = None) -> dict:
  r"""``(dim(lam) / n!) c``, the genuine idempotent projecting onto the irrep component."""
  if n is None:
    n = sum(lam)
  scale = Fraction(dimension(lam), math.factorial(n))
  return {g: scale * c for g, c in young_symmetrizer(tab, n).items()}


def apply_symmetrizer(element: dict, f, x):
  r"""Apply a group-algebra element to a function of n arguments at a point.

  ``element`` maps permutations to coefficients, ``f`` takes a sequence of length n, and ``x`` is
  that sequence. Computes ``sum_g coeff(g) f(g . x)`` with the relabeling action
  ``(g . x)_i = x_{g[i]}``, matching ``physics.nodal``.
  """
  total = None
  for g, coeff in element.items():
    val = float(coeff) * f([x[g[i]] for i in range(len(g))])
    total = val if total is None else total + val
  return total


# --------------------------------------------------------------------------- #
# Characters (Murnaghan-Nakayama on beta-numbers)
# --------------------------------------------------------------------------- #

def _beta(lam: Partition, length: int) -> tuple:
  padded = list(lam) + [0] * (length - len(lam))
  return tuple(padded[i] + (length - 1 - i) for i in range(length))


def _from_beta(beta: Sequence[int]) -> Partition:
  b = sorted(beta, reverse=True)
  L = len(b)
  lam = [b[i] - (L - 1 - i) for i in range(L)]
  while lam and lam[-1] == 0:
    lam.pop()
  return tuple(lam)


@functools.lru_cache(maxsize=None)
def character(lam: Partition, rho: Partition) -> int:
  r"""The irreducible character ``chi^lam`` on the class of cycle type ``rho``.

  Murnaghan-Nakayama: strip one border strip of length ``rho[0]`` from ``lam`` in every possible
  way, weight by ``(-1)^height``, and recurse on the rest of ``rho``. Border strip removal is a
  single step on beta-numbers: replace ``beta_i`` by ``beta_i - r`` when that is non-negative and
  not already present, with the height counting the beta-numbers strictly between them.

  ``chi^{(1^n)}`` is the sign character and ``chi^{(n)}`` is the trivial one; both are useful
  sanity checks.
  """
  lam = tuple(lam)
  rho = tuple(rho)
  if not rho:
    return 1 if not lam else 0
  if not lam:
    return 0
  r, rest = rho[0], rho[1:]
  length = len(lam)
  beta = _beta(lam, length)
  bset = set(beta)
  total = 0
  for i, b in enumerate(beta):
    nb = b - r
    if nb < 0 or nb in bset:
      continue
    height = sum(1 for other in beta if nb < other < b)
    newbeta = list(beta)
    newbeta[i] = nb
    total += (-1) ** height * character(_from_beta(newbeta), rest)
  return total


def class_size(rho: Sequence[int]) -> int:
  r"""Size of the S_n conjugacy class of cycle type ``rho``: ``n! / prod_k k^{m_k} m_k!``."""
  n = sum(rho)
  counts: dict = {}
  for part in rho:
    counts[part] = counts.get(part, 0) + 1
  denom = 1
  for k, m in counts.items():
    denom *= (k ** m) * math.factorial(m)
  return math.factorial(n) // denom


def class_sign(rho: Sequence[int]) -> int:
  r"""Sign of any permutation of cycle type ``rho``: ``(-1)^{n - number of cycles}``."""
  return (-1) ** (sum(rho) - len(rho))


def character_table(n: int):
  r"""Return ``(shapes, table)`` with ``table[i, j] = chi^{shapes[i]}(shapes[j])``.

  Rows and columns are both indexed by the partitions of n in reverse lexicographic order, rows as
  irreps and columns as conjugacy classes. Integer dtype.
  """
  shapes = list(partitions(n))
  table = onp.array(
    [[character(lam, rho) for rho in shapes] for lam in shapes], dtype=onp.int64
  )
  return shapes, table


def character_function(n: int, lam: Sequence[int], max_order: int = 50_000):
  r"""``chi^lam`` as a ``CharG`` over the full group S_n, for interop with ``permutations``.

  Builds a value for every element, so it is only sensible for small n; ``max_order`` guards the
  factorial. For structural work prefer ``character_table``, which is indexed by class.
  """
  if math.factorial(n) > max_order:
    raise ValueError(f"S_{n} has order {math.factorial(n)} > max_order={max_order}")
  group = perms.PermutationGroup.symmetric(n)
  lam = tuple(lam)
  data = {g: character(lam, tuple(perms.cycle_type(g))) for g in group.elements}
  return perms.CharG(data, group)


# --------------------------------------------------------------------------- #
# Molien series
# --------------------------------------------------------------------------- #

def _series_product(a: list, b: list, dmax: int) -> list:
  out = [Fraction(0)] * (dmax + 1)
  for i, ai in enumerate(a):
    if ai == 0:
      continue
    for j, bj in enumerate(b):
      if i + j > dmax:
        break
      out[i + j] += ai * bj
  return out


def molien(n: int, chi, dmax: int) -> list:
  r"""Hilbert series of the ``chi``-isotypic component of the polynomial ring, to degree dmax.

  For S_n permuting n coordinates, a permutation of cycle type ``rho`` acts on the linear forms
  with ``det(1 - q g) = prod_i (1 - q^{rho_i})``, so Molien's formula reads::

    F(q) = (1 / n!) sum_rho |C_rho| chi(rho) / prod_i (1 - q^{rho_i})

  ``chi`` is a callable taking a cycle type (a partition of n) and returning an integer. Computed
  in exact rational arithmetic and returned as a list of ``dmax + 1`` integers; a non-integral
  coefficient means ``chi`` was not a character.
  """
  total = [Fraction(0)] * (dmax + 1)
  for rho in partitions(n):
    weight = Fraction(class_size(rho) * chi(rho), math.factorial(n))
    term = [Fraction(0)] * (dmax + 1)
    term[0] = Fraction(1)
    for part in rho:
      geom = [Fraction(1) if d % part == 0 else Fraction(0) for d in range(dmax + 1)]
      term = _series_product(term, geom, dmax)
    total = [t + weight * s for t, s in zip(total, term)]
  out = []
  for d, c in enumerate(total):
    if c.denominator != 1:
      raise ValueError(f"Molien coefficient at degree {d} is {c}, not an integer")
    out.append(int(c))
  return out


def molien_sign(n: int, dmax: int) -> list:
  r"""Dimensions of the degree-d antisymmetric polynomials in n variables, from characters.

  The same integers as ``src.invariants.antisymmetric_dimensions``, which gets them from the
  closed form ``q^{n(n-1)/2} / prod_k (1 - q^k)``. The two routes share no code: this one sums the
  sign character over conjugacy classes, that one uses freeness of the invariant ring. Agreement
  is a joint test of both modules.
  """
  return molien(n, class_sign, dmax)
