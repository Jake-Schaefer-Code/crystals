# src/permutations.py
from __future__ import annotations
from collections import defaultdict
from collections.abc import Sequence
import dataclasses as dcls
import math
import numpy as onp
import itertools as it
import re
from src.symmetry.core import Field, F

Permutation = tuple[int, ...]
Transposition = tuple[int, int]
Cycle = tuple[int, ...]


_CYCLE_BLOCK_RE = re.compile(r"\(([^()]*)\)")

# conversion to int utils
def int_vec(v):
  return tuple(onp.rint(onp.asarray(v)).astype(int).tolist())

def int_mat(M):
  return onp.rint(onp.asarray(M)).astype(int)

def identity_perm(n: int) -> Permutation:
  return tuple(range(n))

identity = identity_perm

def inverse_perm(p: Permutation) -> Permutation:
  inv = [0] * len(p)
  for i, j in enumerate(p):
    inv[j] = i
  return tuple(inv)

inverse = inverse_perm


def cycle_type(p: Permutation) -> tuple[int, ...]:
  r""" cycle type from permutation `p` """
  seen = set()
  parts = []
  for i in range(len(p)):
    if i in seen:
      continue
    j = i
    k = 0
    while j not in seen:
      seen.add(j)
      j = p[j]
      k += 1
    parts.append(k)
  return tuple(sorted(parts, reverse=True))


def perm_hom_tup(items, image_fn):
  r""" permutation represented as tuple - acts on indices """
  lookup = {item: i for i, item in enumerate(items)}
  return tuple(lookup[image_fn(item)] for item in items)

def perm_hom(perm):
  r""" canonical (I think?) permutation representation homomorphism """
  n = len(perm)
  M = onp.zeros((n, n), dtype=int)
  for j, i in enumerate(perm):
    # TODO why flipped?
    M[i, j] = 1
  return M



def perm_char(p: Permutation) -> int:
  return int(onp.trace(perm_hom(p)))

character = perm_char

def compose_perm(p, q) -> Permutation:
  return tuple(p[q[i]] for i in range(len(p)))

compose = compose_perm

def perm_from_cycles(
  cycles: Sequence[Cycle],
  *,
  degree: int | None = None,
  one_based: bool = True,
) -> Permutation:
  r"""Build a tuple permutation from cycle data.

  The returned permutation uses the existing internal convention
  ``p[i] = g(i)`` on ``{0, ..., n-1}``. Multiple cycles are composed in the
  order they are written, so ``[(1, 2), (2, 3)]`` means ``(1,2)(2,3)``.
  """

  cycles = tuple(tuple(int(label) for label in cycle) for cycle in cycles)
  if degree is None:
    labels = [label for cycle in cycles for label in cycle]
    if not labels:
      raise ValueError("degree is required when no nontrivial cycles are supplied")
    degree = max(labels)

  if degree < 0:
    raise ValueError("degree must be nonnegative")

  offset = 1 if one_based else 0
  lo = offset
  hi = degree - 1 + offset
  out = identity_perm(degree)

  for cycle in cycles:
    if len(cycle) <= 1:
      continue
    shifted = tuple(label - offset for label in cycle)
    if len(set(shifted)) != len(shifted):
      raise ValueError(f"cycle contains repeated labels: {cycle!r}")
    if any(label < lo or label > hi for label in cycle):
      raise ValueError(
        f"cycle labels must lie in [{lo}, {hi}], got {cycle!r}"
      )

    cycle_perm = list(range(degree))
    for src, dst in zip(shifted, shifted[1:] + shifted[:1]):
      cycle_perm[src] = dst
    out = compose_perm(out, tuple(cycle_perm))

  return out

from_cycles = perm_from_cycles

def perm_from_cycle_notation(
  notation: str,
  *,
  degree: int | None = None,
  one_based: bool = True,
) -> Permutation:
  r"""Parse cycle notation like ``"(1,2,3)(4,5)"`` into tuple form."""
  cycles: list[tuple[int, ...]] = []
  pos = 0
  for match in _CYCLE_BLOCK_RE.finditer(notation):
    if notation[pos:match.start()].strip():
      raise ValueError(f"invalid cycle notation: {notation!r}")
    body = match.group(1).strip()
    if body:
      pieces = [piece for piece in re.split(r"[\s,]+", body) if piece]
      cycles.append(tuple(int(piece) for piece in pieces))
    else:
      cycles.append(tuple())
    pos = match.end()

  if notation[pos:].strip() or not cycles:
    raise ValueError(f"invalid cycle notation: {notation!r}")

  return perm_from_cycles(cycles, degree=degree, one_based=one_based)

from_cycle_notation = perm_from_cycle_notation

def perm_to_cycles(
  p: Permutation,
  *,
  one_based: bool = True,
  include_fixed: bool = False,
) -> tuple[Cycle, ...]:
  r"""Convert tuple permutation form into disjoint cycle form."""

  p = tuple(int(i) for i in p)
  n = len(p)
  if tuple(sorted(p)) != tuple(range(n)):
    raise ValueError(f"invalid permutation: {p!r}")

  offset = 1 if one_based else 0
  seen = set()
  cycles = []


  for start in range(n):
    if start in seen:
      continue
    cycle = []
    j = start
    while j not in seen:
      seen.add(j)
      cycle.append(j + offset)
      j = p[j]
    if len(cycle) == 1 and not include_fixed:
      continue

    min_idx = min(range(len(cycle)), key=cycle.__getitem__)
    cycle = cycle[min_idx:] + cycle[:min_idx]
    cycles.append(tuple(cycle))

  return tuple(cycles)

to_cycles = perm_to_cycles

def perm_to_cycle_notation(
  p: Permutation,
  *,
  one_based: bool = True,
  include_fixed: bool = False,
) -> str:
  r"""Format tuple permutation form as cycle notation."""

  cycles = perm_to_cycles(p, one_based=one_based, include_fixed=include_fixed)
  if not cycles:
    return "()"
  return "".join("(" + ",".join(str(label) for label in cycle) + ")" for cycle in cycles)


to_cycle_notation = perm_to_cycle_notation


def to_transpositions(p: Permutation) -> tuple[Transposition, ...]:
  raise NotImplementedError

def generated_group(
  generators: Sequence[Permutation],
  *,
  name: str = "",
  max_order: int | None = None,
) -> "PermutationGroup":
  r"""Close the subgroup generated by tuple permutations."""

  gens = tuple(tuple(int(i) for i in gen) for gen in generators)
  if not gens:
    raise ValueError("at least one generator is required")

  degree = len(gens[0])
  expected = tuple(range(degree))
  for gen in gens:
    if len(gen) != degree:
      raise ValueError("all generators must have the same degree")
    if tuple(sorted(gen)) != expected:
      raise ValueError(f"invalid permutation generator: {gen!r}")

  ident = identity(degree)
  seen = {ident}
  elts = [ident]
  frontier = [ident]

  def parse_candidate(cand):
    if cand in seen:
      return
    seen.add(cand)
    elts.append(cand)
    frontier.append(cand)
    if max_order is not None and len(elts) > max_order:
      raise ValueError(
        "generated group exceeded max_order; "
        "check the generators or raise max_order"
      )

  while frontier:
    current = frontier.pop()
    for gen in gens:
      for candidate in (compose(gen, current), compose(current, gen)):
        parse_candidate(candidate)

  return PermutationGroup(tuple(elts), name=name, generators=gens)




def power(p, k):
  r""" get power of permuatation """
  out = identity_perm(len(p))
  # TODO scan/vmap
  for _ in range(k):
    out = compose(p, out)
  return out

perm_power = power

def sgn(p):
  r""" get sign of permuatation """
  inversions = 0
  for i in range(len(p)):
    for j in range(i + 1, len(p)):
      inversions += p[i] > p[j]
  return -1 if inversions % 2 else 1

sign_perm = sgn

def Fix(p: Permutation) -> int:
  return sum(i == p[i] for i in range(len(p)))

fixed_points = Fix

def conjugate_perm(p: Permutation, g: Permutation) -> Permutation:
  r""" Conjugate `p` by `g`, i.e., return `g p g^{-1}`. """
  ginv = inverse_perm(g)
  return compose_perm(g, compose_perm(p, ginv))


def commutator_perm(p: Permutation, q: Permutation) -> Permutation:
  r""" Return the commutator ``[p, q] = p q p^{-1} q^{-1}``."""
  if len(p) != len(q):
    raise ValueError("commutator requires permutations of the same degree")
  pinv = inverse_perm(p)
  qinv = inverse_perm(q)
  return compose(p, compose(q, compose(pinv, qinv)))


commutator = commutator_perm


def conjugacy_class(p: Permutation, group: tuple[Permutation, ...] | list[Permutation]|PermutationGroup) -> tuple[Permutation, ...]:
  r"""Conjugacy class of `p` inside the supplied permutation group."""
  seen = set()
  cls = []
  for g in group:
    q = conjugate_perm(p, g)
    if q in seen:
      continue
    seen.add(q)
    cls.append(q)
  return tuple(cls)

def conjugacy_class_representative(p: Permutation, group: tuple[Permutation, ...] | list[Permutation]|PermutationGroup) -> Permutation:
  r""" Get conjugacy class representative for a permutation given a group (sorts based on min) """
  return min(conjugacy_class(p, group))

# TODO this should be ordered with identity first
def conjugacy_classes(group: tuple[Permutation, ...] | list[Permutation]) -> tuple[tuple[Permutation, ...], ...]:
  r"""Partition a finite permutation group into subgroup conjugacy classes."""
  remaining = set(group)
  classes = []
  while remaining:
    p = next(iter(remaining))
    cls = conjugacy_class(p, group)
    classes.append(cls)
    # remove redundant class elements - removes all elements in cls from the 
    # set of permutation elements
    remaining.difference_update(cls)
  return tuple(classes)


def conjugacy_class_representatives(
  group: tuple[Permutation, ...] | list[Permutation],
) -> tuple[Permutation, ...]:
  r"""Deterministic representatives for subgroup conjugacy classes."""
  return tuple(sorted(min(cls) for cls in conjugacy_classes(group)))


def commutator_subgroup(
  group: tuple[Permutation, ...] | list[Permutation],
  *,
  generators: Sequence[Permutation] | None = None,
  name: str = "",
  max_order: int | None = None,
) -> "PermutationGroup":
  r"""Return the derived subgroup generated by commutators.

  If ``generators`` is supplied, we use the standard fact that ``[G, G]`` is
  the normal closure of the pairwise generator commutators ``[s, t]``. This
  is much cheaper than scanning all element pairs in large groups such as
  ``M11``.
  """
  group = tuple(tuple(int(i) for i in g) for g in group)
  if not group:
    raise ValueError("commutator_subgroup requires a nonempty group")

  degree = len(group[0])
  expected = tuple(range(degree))
  for g in group:
    if len(g) != degree:
      raise ValueError("all group elements must have the same degree")
    if tuple(sorted(g)) != expected:
      raise ValueError(f"invalid group element: {g!r}")

  if generators is None:
    gens = group
  else:
    # gens = tuple(map(lambda g: tuple(map(int, g)), generators))
    # gens = tuple(tuple(map(int, g)) for g in generators)
    gens = tuple(tuple(int(i) for i in g) for g in generators)

  if not gens:
    raise ValueError("generators must be nonempty when supplied")
  for g in gens:
    if len(g) != degree:
      raise ValueError("all generators must have the same degree as the group")
    if tuple(sorted(g)) != expected:
      raise ValueError(f"invalid generator: {g!r}")

  bound = len(group) if max_order is None else max_order

  if generators is None:
    seeds = []
    seen = set()
    for s in gens:
      for g in group:
        c = commutator_perm(s, g)
        if c in seen:
          continue
        seen.add(c)
        seeds.append(c)
    return generated_group(seeds, name=name, max_order=bound)

  seed_gens = []
  seed_seen = set()
  for s in gens:
    for t in gens:
      c = commutator_perm(s, t)
      if c in seed_seen:
        continue
      seed_seen.add(c)
      seed_gens.append(c)
  if not seed_gens:
    seed_gens = [identity(degree)]

  subgens = list(seed_gens)
  while True:
    subgroup = generated_group(subgens, name=name, max_order=bound)
    subgroup_set = set(subgroup.elements)

    new_gens = []
    new_seen = set()
    for s in gens:
      for h in subgens:
        c = conjugate_perm(h, s)
        if c in subgroup_set or c in new_seen:
          continue
        new_seen.add(c)
        new_gens.append(c)

    if not new_gens:
      return PermutationGroup(
        subgroup.elements,
        name=name,
        generators=tuple(subgens),
      )
    subgens.extend(new_gens)


def classes_by_cycle_type(
  group: tuple[Permutation, ...] | list[Permutation]
) -> dict[tuple[int, ...], tuple[Permutation, ...]]:
  r"""Group permutations by cycle type.

  For the full symmetric group `S_n`, these are exactly the conjugacy classes.
  For a subgroup, same cycle type does not necessarily imply conjugacy.
  """
  classes = defaultdict(list)
  for p in group:
    classes[cycle_type(p)].append(p)
  return {ct: tuple(perms) for ct, perms in classes.items()}


def class_size_symmetric(cycle_shape: tuple[int, ...]) -> int:
  r"""Size of the `S_n` conjugacy class with the given cycle type."""
  n = sum(cycle_shape)
  multiplicities = defaultdict(int)
  for part in cycle_shape:
    multiplicities[part] += 1

  denom = 1
  for part, mult in multiplicities.items():
    denom *= part ** mult
    denom *= math.factorial(mult)
  return math.factorial(n) // denom


def centralizer_size_symmetric(cycle_shape: tuple[int, ...]) -> int:
  r""" Size of the centralizer in `S_n` of a permutation with the given cycle type. """
  n = math.factorial(sum(cycle_shape))
  return n // class_size_symmetric(cycle_shape)


# TODO this should be registered as a pytree-dataclass for backprop eventually
# However, I will then need to change the method of composition, as this is not 
# really supported through JIT compilation
# @dcls.dataclass(frozen=True)
class PermElt(tuple):
  _name: str = ''
  def __new__(cls, *a, name: str=''):
    obj = tuple.__new__(PermElt, a)
    obj._name = name if name != '' else to_cycle_notation(obj)
    return obj

  def __rmul__(self, other: PermElt):
    r""" im thinking right-multiplication should be composition """
    return PermElt(compose_perm(other, self))

  def __mul__(self, other: PermElt):
    r""" TODO left-multiplication should be action??? But how to implement... """
    raise NotImplementedError

  def __call__(self, other: PermElt):
    r""" TODO or should __call__ be action? or call is just composition? 
    idk... gotta decide """
    return PermElt(compose(self, other))

  @property
  def name(self) -> str:
    return self._name
  # def __getitem__(self, key):
  #   return self.perm[key]
  
  # def __hash__(self):
  #   r""" hash the owned object """
  #   return hash(self.perm)

  def inv(self):
    return inverse(self)
  # def conj(self):
    # return conjugate_perm()


@dcls.dataclass(frozen=True)
class PermutationGroup:
  r""" permutation group structure """
  elements: tuple[Permutation, ...]
  name: str = ""
  generators: tuple[Permutation, ...] = ()

  @classmethod
  def symmetric(cls, n: int, *, name: str | None = None) -> "PermutationGroup":
    return cls(
      tuple(it.permutations(range(n))),
      name=name or f"S{n}",
      generators=_symmetric_generators(n),
    )

  # @staticmethod
  # def symmetric_stream()

  @classmethod
  def generated(
    cls,
    generators: Sequence[Permutation],
    *,
    name: str = "",
    max_order: int | None = None,
  ) -> "PermutationGroup":
    return generated_group(generators, name=name, max_order=max_order)

  def __iter__(self):
    return iter(self.elements)

  def __len__(self) -> int:
    return len(self.elements)

  @property
  def order(self) -> int:
    return len(self.elements)

  @property
  def degree(self) -> int:
    return len(self.elements[0]) if self.elements else 0

  def conjugacy_class(self, p: Permutation) -> tuple[Permutation, ...]:
    return conjugacy_class(p, self.elements)

  def conjugacy_classes(self) -> tuple[tuple[Permutation, ...], ...]:
    r""" conjugacy classes of permutation group """
    return conjugacy_classes(self.elements)

  def conjugacy_class_representatives(self) -> tuple[Permutation, ...]:
    return conjugacy_class_representatives(self.elements)

  def conjugacy_class_sizes(self) -> dict[Permutation, int]:
    classes = conjugacy_classes(self.elements)
    return {min(clg): len(clg) for clg in classes}

  def commutator_subgroup(
    self,
    *,
    generators: Sequence[Permutation] | None = None,
    name: str | None = None,
    max_order: int | None = None,
  ) -> "PermutationGroup":
    gens = self.generators if generators is None and self.generators else generators
    if name is None:
      name = f"[{self.name},{self.name}]" if self.name else ""
    return commutator_subgroup(
      self.elements,
      generators=gens,
      name=name,
      max_order=max_order,
    )

  def classes_by_cycle_type(self) -> dict[tuple[int, ...], tuple[Permutation, ...]]:
    return classes_by_cycle_type(self.elements)

  def class_sizes_by_cycle_type(self) -> dict[tuple[int, ...], int]:
    return {ct: len(cls) for ct, cls in self.classes_by_cycle_type().items()}

  def matrices(self):
    return {p: perm_hom(p) for p in self.elements}




def substitute_roots(sigma, roots):
  """Return (alpha_{sigma(0)}, ..., alpha_{sigma(n-1)})."""
  if len(sigma) != len(roots):
    raise ValueError("permutation and roots must have the same length")
  return tuple(roots[sigma[i]] for i in range(len(sigma)))

def act_on_indexed_data(sigma, data):
  """Left action on indexed data: (sigma · x)_i = x_{sigma^{-1}(i)}."""
  if len(sigma) != len(data):
    raise ValueError("permutation and data must have the same length")
  sigma_inv = inverse_perm(sigma)
  return tuple(data[sigma_inv[i]] for i in range(len(sigma)))


def _symmetric_generators(n: int) -> tuple[Permutation, ...]:
  if n < 2:
    return ()

  gens = [from_cycle_notation("(1,2)", degree=n)]
  if n >= 3:
    gens.append(from_cycle_notation("(" + ",".join(map(str, range(1, n + 1))) + ")", degree=n))
  return tuple(gens)


def _alternating_generators(n: int) -> tuple[Permutation, ...]:
  if n < 3:
    return ()
  return tuple(
    from_cycle_notation(f"(1,2,{k})", degree=n)
    for k in range(3, n + 1)
  )



def Cn(n):
  gen = from_cycle_notation("(" + ",".join(map(str, range(1, n + 1))) + ")", degree=n)
  return PermutationGroup.generated((gen,), name=f"C{n}")

def An(n):
  Sn = PermutationGroup.symmetric(n)
  return PermutationGroup(
    tuple(g for g in Sn if sgn(g) == 1),
    name=f"A{n}",
    generators=_alternating_generators(n),
  )


def dist(G, terms):
  out = {g: 0.0 for g in G}
  for c, g in terms:
    out[g] += float(c)
  return out

def convolve(G, P, Q):
  r""" Convolution of two distributions on ``G``: the product in the group algebra """
  from src.symmetry.group_algebra import GroupAlgebra  # group_algebra imports this module
  A = GroupAlgebra.of(G)
  prod = A.element(P) * A.element(Q)
  return {g: float(prod[g]) for g in G}

def power_dist(G, P, n):
  e = identity_perm(G.degree)
  out = {g: float(g == e) for g in G}
  base = P
  while n:
    if n & 1:
      out = convolve(G, out, base)
    n >>= 1
    if n:
      base = convolve(G, base, base)
  return out

def variation_distance(P, Q, G):
  if Q is None:
    Q = {g: 1.0 / G.order for g in G}
  return 0.5 * sum(abs(P[g] - Q[g]) for g in G)


class CharG[F]:
  r""" Group does not determine a character; it only determines the domain and conjugacy classes """
  def __init__(self, data: dict[Permutation, F], group: PermutationGroup):
    self.group = group
    self.class_reps = group.conjugacy_class_representatives()
    self.class_sizes = group.conjugacy_class_sizes()
    normed = {}
    for key, val in data.items():
      rep = conjugacy_class_representative(key, group)
      if rep in normed and normed[rep] != val:
        raise ValueError(f"inconsistent values on class {rep!r}")
      normed[rep] = val

    missing = [rep for rep in self.class_reps if rep not in normed]
    if missing:
      raise ValueError(f"missing values for classes {missing!r}")

    # get_rep = lambda p: perms.conjugacy_class_representative(p, group)
    # normed = dict(zip(map(get_rep, data.keys()), data.values()))
    self.data = normed

  def __getitem__(self, key) -> F:
    rep = conjugacy_class_representative(key, self.group)
    return self.data[rep]

  def __or__(self, other: 'CharG'):
    r""" Inner product on G!!! """
    return sum(
      self.class_sizes[rep] * onp.conj(self[rep]) * other[rep] 
      for rep in self.class_reps
    ) / self.group.order
  
  def values_on_classes(self):
    return [self[rep] for rep in self.class_reps]

  def decompose(self, irrep_table):
    return {
      name: (self|irrep)
      for name, irrep in irrep_table.items()
    }
  @classmethod
  def from_row(cls, row: list[F], group: PermutationGroup):
    reps = group.conjugacy_class_representatives()
    return cls(dict(zip(reps, row)), group)


