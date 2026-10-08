# physics/nodal.py
r"""Nodal surfaces of real antisymmetric wavefunctions: slices, node distances, nodal-cell census.

Conventions
-----------
A wavefunction is a callable ``psi(R) -> (sign, log|psi|)`` on one configuration ``R`` of shape
``(n_particles, dim)``: the ``(sign, logabsdet)`` convention of ``jnp.linalg.slogdet``. Particle
``i`` has spin ``spins[i]`` in {0, 1}, and psi must be antisymmetric under exchanging particles of
equal spin. Batched helpers accept configurations of shape ``(..., n_particles, dim)``. Enable
float64 (``jax.config.update("jax_enable_x64", True)``) for trustworthy signs near the node.

A jaxqmc ``Wavefunction`` returns ``(log|psi|, sign)`` and takes spins explicitly::

  psi = lambda R: wf(R, spins)[::-1]

Nodal cells
-----------
Relabeling by P in S = S_up x S_down gives psi(P R) = sgn(P) psi(R), so P maps nodal cells onto
nodal cells. The stabilizer G = {P : P Omega = Omega} of the cell Omega containing R0 lies in the
even subgroup S+ = ker(sgn), and the distinct cells P Omega correspond to the cosets S / G. For a
ground state the tiling theorem [Ceperley, J. Stat. Phys. 63, 1237 (1991)] says these are all the
cells, so n_cells = |S| / |G|. Ground states in dim >= 2 generically have G = S+, i.e. two cells
[Mitas, PRL 96, 240402 (2006)], while a product of up and down determinants has at least four.
In one dimension particles cannot pass each other and there are n_up! n_down! cells.

``cell_census`` exhibits elements of G as node-free exchange paths, which bounds the number of
cells from above; see its docstring.
"""
from __future__ import annotations

import collections
import dataclasses as dcls
import functools
import math

import numpy as onp
import jax
import jax.numpy as jnp

import src.symmetry.permutations as perms


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def _distance_fn(psi):
  grad_log = jax.grad(lambda R: psi(R)[1])
  return lambda R: 1.0 / jnp.linalg.norm(grad_log(R))


@functools.lru_cache(maxsize=64)
def _jit_psi(psi):
  return jax.jit(jax.vmap(psi))


@functools.lru_cache(maxsize=64)
def _jit_node_distance(psi):
  return jax.jit(jax.vmap(_distance_fn(psi)))


def _batched(fn, R, chunk=1 << 14):
  R = jnp.asarray(R)
  flat = R.reshape(-1, *R.shape[-2:])
  out = [fn(flat[i:i + chunk]) for i in range(0, flat.shape[0], chunk)]
  out = jax.tree.map(lambda *xs: jnp.concatenate(xs), *out)
  return jax.tree.map(lambda x: x.reshape(R.shape[:-2]), out)


def evaluate(psi, R):
  r"""``(sign, log|psi|)`` for configurations ``R`` of shape ``(..., n, dim)``."""
  return _batched(_jit_psi(psi), R)


def node_distance(psi, R):
  r"""First-order distance to the node, ``|psi| / |grad psi| = 1 / |grad log|psi||``.

  The distance from R to the zero set of the linearization of psi at R. It becomes exact as R
  approaches a smooth node, and under |psi|^2 sampling its density vanishes like d^2 at d = 0.
  """
  return _batched(_jit_node_distance(psi), R)


def metropolis(psi, R0, key, n_sweeps, step, *, fixed_node=False, n_sub=8):
  r"""Single-particle Metropolis sampling of |psi|^2.

  ``R0`` has shape ``(n_walkers, n, dim)`` (or ``(n, dim)``). Each sweep proposes
  ``r_k -> r_k + step * xi`` for every particle ``k`` in turn. Returns ``(samples, acceptance)``
  with one sample per sweep: ``samples.shape == (n_sweeps, n_walkers, n, dim)``.

  ``fixed_node=True`` confines each walker to the nodal cell it starts in: a move is also rejected
  unless, at ``n_sub`` interior points and both ends of the displacement, psi keeps its sign and
  ``node_distance`` exceeds twice the point spacing (the test ``cell_census`` applies to paths).
  The extra test is symmetric in (R, R'), so the chain still samples |psi|^2, restricted to the
  cell.
  """
  R0 = jnp.asarray(R0)
  R0 = R0[None] if R0.ndim == 2 else R0
  n_walkers, n, dim = R0.shape
  vpsi = jax.vmap(psi)
  vdist = jax.vmap(_distance_fn(psi))
  frac = jnp.arange(1, n_sub + 2) / (n_sub + 1)  # interior points, then the endpoint

  def move(carry, k):
    R, s, l, d, key = carry
    key, k_prop, k_acc = jax.random.split(key, 3)
    dr = step * jax.random.normal(k_prop, (n_walkers, dim))
    Rp = R.at[:, k].add(dr)
    if fixed_node:
      pts = R[None] + frac[:, None, None, None] * (Rp - R)[None]
      s_pts, l_pts = jax.vmap(vpsi)(pts)
      d_pts = jax.vmap(vdist)(pts)
      sp, lp = s_pts[-1], l_pts[-1]
      margin = 2.0 * jnp.linalg.norm(dr, axis=-1) / (n_sub + 1)
      safe = jnp.all(s_pts == s, axis=0) & jnp.all(d_pts > margin, axis=0) & (d > margin)
    else:
      sp, lp = vpsi(Rp)
      safe = True
    accept = safe & (jnp.log(jax.random.uniform(k_acc, (n_walkers,))) < 2.0 * (lp - l))
    if fixed_node:
      d = jnp.where(accept, d_pts[-1], d)
    R = jnp.where(accept[:, None, None], Rp, R)
    return (R, jnp.where(accept, sp, s), jnp.where(accept, lp, l), d, key), accept

  def sweep(carry, _):
    carry, accepted = jax.lax.scan(move, carry, jnp.arange(n))
    return carry, (carry[0], accepted.mean())

  @jax.jit
  def run(R0, key):
    s0, l0 = vpsi(R0)
    d0 = vdist(R0) if fixed_node else jnp.zeros(n_walkers)
    _, (samples, acc) = jax.lax.scan(sweep, (R0, s0, l0, d0, key), None, length=n_sweeps)
    return samples, acc.mean()

  return run(R0, key)


# --------------------------------------------------------------------------- #
# Nodal-cell census
# --------------------------------------------------------------------------- #

def _support(p):
  return frozenset(i for i, j in enumerate(p) if i != j)


def _local_exchanges(R, spins):
  r"""Exchanges of nearby particles in ``R``, as permutations acting by ``(P R)_i = R_{P[i]}``.

  For each particle ``a`` of a species with at least three members, the 3-cycle through ``a`` and
  its two nearest same-spin neighbours. If both species have at least two members, for each up
  particle ``a`` the double transposition (a b)(c e): ``b`` the nearest up neighbour of ``a``,
  ``c`` the down particle nearest their midpoint, ``e`` the nearest down neighbour of ``c``.
  """
  R = onp.asarray(R)
  n = len(spins)
  dist = onp.linalg.norm(R[:, None] - R[None], axis=-1)
  up, dn = (onp.flatnonzero(spins == s) for s in (0, 1))
  out = []
  for members in (up, dn):
    if len(members) < 3:
      continue
    for a in members:
      b, c = sorted((j for j in members if j != a), key=lambda j: dist[a, j])[:2]
      out.append(perms.from_cycles(((a, b, c),), degree=n, one_based=False))
  if len(up) >= 2 and len(dn) >= 2:
    for a in up:
      b = min((j for j in up if j != a), key=lambda j: dist[a, j])
      c = min(dn, key=lambda j: onp.linalg.norm(R[j] - 0.5 * (R[a] + R[b])))
      e = min((j for j in dn if j != c), key=lambda j: dist[c, j])
      out.append(perms.from_cycles(((a, b), (c, e)), degree=n, one_based=False))
  return list(dict.fromkeys(out))


def _bulged_paths(A, B, W, t):
  r"""Points of ``R(t) = (1 - t) A + t B + sin(pi t) W``: ``(m, n, dim)`` -> ``(m, len(t), n, dim)``."""
  t = t[None, :, None, None]
  return (1 - t) * A[:, None] + t * B[:, None] + jnp.sin(jnp.pi * t) * W[:, None]


def _node_free(psi, A, B, W, n_points, n_coarse=64, chunk=64):
  r"""Mask of bulged paths along which psi keeps its sign and ``node_distance`` exceeds twice the
  local point spacing at all ``n_points`` points, so that no pair of crossings hides between them."""
  ok = onp.zeros(len(A), dtype=bool)
  t = jnp.linspace(0.0, 1.0, n_coarse)
  for i in range(0, len(A), 1024):  # cheap sign-only filter
    s, _ = evaluate(psi, _bulged_paths(A[i:i + 1024], B[i:i + 1024], W[i:i + 1024], t))
    ok[i:i + 1024] = onp.asarray(jnp.all(s == s[:, :1], axis=1) & (s[:, 0] != 0))
  t = jnp.linspace(0.0, 1.0, n_points)
  survivors = onp.flatnonzero(ok)
  for i in range(0, len(survivors), chunk):
    idx = survivors[i:i + chunk]
    padded = onp.resize(idx, chunk)  # fixed batch shape -> a single compilation
    P = _bulged_paths(A[padded], B[padded], W[padded], t)
    s, _ = evaluate(psi, P)
    d = node_distance(psi, P)
    gap = jnp.linalg.norm(jnp.diff(P, axis=1).reshape(chunk, n_points - 1, -1), axis=-1)
    h = jnp.maximum(jnp.pad(gap, ((0, 0), (1, 0))), jnp.pad(gap, ((0, 0), (0, 1))))
    good = jnp.all(s == s[:, :1], axis=1) & jnp.all(d > 2.0 * h, axis=1)
    ok[idx] = onp.asarray(good)[:len(idx)]
  return ok


def _stabilizer_order(certified, spins, group_order, max_group_order):
  r"""``|H|`` for ``H = <certified>``, or None if H would have to be enumerated and is too big."""
  triples = [_support(p) for p in certified if len(_support(p)) == 3]
  parent = list(range(len(spins)))

  def find(i):
    while parent[i] != i:
      parent[i] = parent[parent[i]]
      i = parent[i]
    return i

  for a, *rest in map(tuple, triples):
    for b in rest:
      parent[find(b)] = find(a)
  touched = set().union(*triples)
  # 3-cycles generate the alternating group on each connected component of their supports
  sizes = collections.Counter(find(i) for i in touched)
  order = math.prod(math.factorial(k) // 2 for k in sizes.values())
  if len(triples) == len(certified):
    return order
  spanned = all(set(m) <= touched and len({find(i) for i in m}) == 1
                for m in (onp.flatnonzero(spins == s) for s in (0, 1)) if len(m) >= 3)
  if spanned:  # A_up x A_down has index 2 in S+, and a double transposition lies outside it
    return group_order // 2
  if group_order <= max_group_order:
    return perms.generated_group(list(certified)).order
  return None


@dcls.dataclass(frozen=True)
class CellCensus:
  r"""Outcome of ``cell_census``. Permutations act by relabeling, ``(P R)_i = R_{P[i]}``.

  ``certified`` exchanges are shown (up to path discretization) to map the reference cell onto
  itself. They generate a subgroup H of the cell's stabilizer G with ``|H| = stabilizer_order``,
  so there are at most ``n_cells_upper = |S| / |H|`` distinct cells P Omega(R0): every cell, for a
  ground state. ``attempts[kind] = (tried, certified)`` counts candidate exchanges; a kind that is
  never certified is evidence, not proof, that the corresponding cells are disconnected.
  ``perms.generated_group(census.certified)`` builds H itself when it is small.
  """
  certified: tuple[tuple[int, ...], ...]
  witnesses: dict[tuple[int, ...], onp.ndarray]  # one node-free path (n_points, n, dim) per P
  attempts: dict[str, tuple[int, int]]
  group_order: int
  stabilizer_order: int | None
  n_cells_lower: int
  walk_acceptance: float

  @property
  def n_cells_upper(self) -> int | None:
    return None if self.stabilizer_order is None else self.group_order // self.stabilizer_order

  def __str__(self) -> str:
    order = "?" if self.stabilizer_order is None else self.stabilizer_order
    lines = [f"S_up x S_down has order {self.group_order}; certified exchanges generate order {order}"]
    lines += [f"  {kind}: {ok}/{tried} certified" for kind, (tried, ok) in self.attempts.items()]
    upper = self.n_cells_upper
    if upper == self.n_cells_lower:
      lines.append(f"nodal cells: {upper} (exact)")
    else:
      lines.append(f"nodal cells: at most {'?' if upper is None else upper} (at least {self.n_cells_lower})")
    return "\n".join(lines)


def cell_census(psi, spins, R0, key, *, step, n_walkers=64, n_sweeps=200, n_paths=16,
                n_points=1024, max_group_order=100_000):
  r"""Bound the number of nodal cells of ``psi`` by certifying node-free particle exchanges.

  1. ``n_walkers`` fixed-node walkers run ``n_sweeps`` sweeps of ``metropolis`` (move size
     ``step``) from ``R0``; each final configuration R* is joined to R0 by a node-free path.
  2. From each R*, exchanges P of nearby particles (3-cycles within a species; double
     transpositions pairing an up pair with a down pair) are tried along ``n_paths`` random paths
     ``R(t) = (1 - t) R* + t P R* + sin(pi t) W``. A path certifies P if psi keeps its sign and
     ``node_distance`` exceeds twice the point spacing at all ``n_points`` points. Then
     R0 -> R* -> P R* -> P R0 is node-free, so P is in the stabilizer G of the cell of R0.
  3. Certified 3-cycles generate the alternating group on each connected component of their
     supports. If each species is spanned and a double transposition is certified, H = S+ and the
     bound is two cells.
  """
  spins = onp.asarray(spins)
  R0 = jnp.asarray(R0)
  n, dim = R0.shape
  if evaluate(psi, R0)[0] == 0:
    raise ValueError("R0 lies on the node; start from a configuration where psi != 0")
  counts = [int(onp.sum(spins == s)) for s in (0, 1)]
  group_order = math.factorial(counts[0]) * math.factorial(counts[1])
  k_walk, k_amp, k_spec, k_dir = jax.random.split(key, 4)

  samples, acc = metropolis(psi, jnp.broadcast_to(R0, (n_walkers, n, dim)), k_walk, n_sweeps, step,
                            fixed_node=True)
  starts = onp.asarray(samples[-1])
  jobs = [(w, p) for w in range(n_walkers) for p in _local_exchanges(starts[w], spins)]
  kinds = {3: "3-cycles", 4: "double exchanges"}
  attempts = {kinds[len(_support(p))]: [0, 0] for _, p in jobs}
  certified, witnesses = [], {}
  if jobs:
    m = len(jobs)
    A = starts[[w for w, _ in jobs]]                                     # R*
    B = onp.stack([a[list(p)] for a, (_, p) in zip(A, jobs)])           # P R*
    moving = onp.stack([onp.asarray(p) != onp.arange(n) for _, p in jobs])
    scale = onp.sqrt(onp.mean(onp.sum((A[:, :, None] - A[:, None]) ** 2, axis=-1), axis=(1, 2)))
    # bulge size log-uniform in [0.1, 1] x rms particle separation; spectators get a random fraction
    amp = scale[:, None] * jnp.exp(jax.random.uniform(k_amp, (m, n_paths), minval=math.log(0.1), maxval=0.0))
    weight = jnp.where(moving[:, None, :], 1.0, jax.random.uniform(k_spec, (m, n_paths, 1)))
    W = jax.random.normal(k_dir, (m, n_paths, n, dim)) * (amp[..., None] * weight)[..., None] / math.sqrt(n * dim)
    ok = _node_free(psi, onp.repeat(A, n_paths, 0), onp.repeat(B, n_paths, 0), W.reshape(-1, n, dim), n_points)
    ok = ok.reshape(m, n_paths)
    t = jnp.linspace(0.0, 1.0, n_points)
    for k, (_, p) in enumerate(jobs):
      tally = attempts[kinds[len(_support(p))]]
      tally[0] += 1
      if not ok[k].any():
        continue
      tally[1] += 1
      if p not in witnesses:
        j = int(onp.argmax(ok[k]))
        witnesses[p] = onp.asarray(_bulged_paths(A[k:k + 1], B[k:k + 1], W[k, j][None], t)[0])
        certified.append(p)
  return CellCensus(
    certified=tuple(certified),
    witnesses=witnesses,
    attempts={kind: tuple(v) for kind, v in attempts.items()},
    group_order=group_order,
    stabilizer_order=_stabilizer_order(certified, spins, group_order, max_group_order),
    n_cells_lower=2 if max(counts) >= 2 else 1,
    walk_acceptance=float(acc),
  )


# --------------------------------------------------------------------------- #
# Comparisons and checks
# --------------------------------------------------------------------------- #

def sign_agreement(psi_a, psi_b, R):
  r"""Fraction of configurations ``R (..., n, dim)`` on which psi_a and psi_b share a sign.

  Maximized over the arbitrary global sign, ``max(p, 1 - p)``. Draw ``R`` from |psi_b|^2 to weight
  by probability; 1 means the two nodal surfaces agree wherever the samples go.
  """
  s_a, _ = evaluate(psi_a, R)
  s_b, _ = evaluate(psi_b, R)
  p = float(jnp.mean(s_a == s_b))
  return max(p, 1.0 - p)


def permutation_equivariance_error(psi, R, elements, character):
  R = jnp.asarray(R)
  R = R.reshape(-1, *R.shape[-2:])
  elements = tuple(elements)
  if not elements:
    return 0, 0.0

  s, l = evaluate(psi, R)
  transformed = jnp.stack([
    R[:, jnp.asarray(p), :]
    for p in elements
  ])
  s_g, l_g = evaluate(psi, transformed)

  chi = jnp.asarray(list(map(character, elements)))

  sign_errors = jnp.sum(s_g != chi[:, None] * s[None, :])
  log_error = jnp.max(jnp.abs(l_g - l[None, :]))
  return int(sign_errors), float(log_error)


def antisymmetry_error(psi, R, spins):
  r"""Violations of ``psi(P_ij R) = -psi(R)`` over same-spin transpositions ``(i j)``.

  Returns ``(n_sign_errors, max_log_error)`` over configurations ``R (..., n, dim)``: the number of
  (configuration, transposition) pairs whose sign does not flip, and the largest
  ``| log|psi(P_ij R)| - log|psi(R)| |``.
  """
  spins = onp.asarray(spins)
  n = R.shape[-2]
  transpositions = [
    perms.from_cycles(((i, j),), degree=n, one_based=False)
    for i in range(n)
    for j in range(i + 1, n)
    if spins[i] == spins[j]
  ]
  return permutation_equivariance_error(
    psi, R, transpositions, perms.sgn
  )

# --------------------------------------------------------------------------- #
# Slices through configuration space
# --------------------------------------------------------------------------- #

def coordinate_plane(R, first, second):
  r"""``(base, d1, d2)`` for ``slice_grid`` with ``u, v`` the absolute coordinates ``R[first]`` and
  ``R[second]`` (each a ``(particle, axis)`` pair); all other coordinates stay frozen at ``R``.

  ``coordinate_plane(R, (i, 0), (i, 1))`` moves particle ``i`` in the xy-plane;
  ``coordinate_plane(R, (0, 0), (1, 0))`` is the (x_0, x_1) plane of a one-dimensional system.
  """
  R = jnp.asarray(R, dtype=float)
  base = R.at[first].set(0.0).at[second].set(0.0)
  return base, jnp.zeros_like(R).at[first].set(1.0), jnp.zeros_like(R).at[second].set(1.0)


def jacobi_plane(n):
  r"""Orthonormal directions ``(d1, d2)`` of shape ``(n, 1)``: the first two Jacobi coordinates of
  ``n`` particles on a line, both orthogonal to the centre-of-mass direction. For ``n = 3`` every
  coincidence plane x_i = x_j is a line through the origin, cutting the plane into 3! wedges."""
  d1 = onp.zeros((n, 1))
  d2 = onp.zeros((n, 1))
  d1[:2, 0] = onp.array([1.0, -1.0]) / onp.sqrt(2.0)
  d2[:3, 0] = onp.array([1.0, 1.0, -2.0]) / onp.sqrt(6.0)
  return jnp.asarray(d1), jnp.asarray(d2)


def plane_extent(extent):
  if onp.ndim(extent) == 0:
    return (-extent, extent, -extent, extent)
  return tuple(extent)


def slice_grid(psi, base, d1, d2, extent, n=201):
  r"""Evaluate psi on the plane ``base + u d1 + v d2`` of configuration space.

  ``base, d1, d2`` have shape ``(n_particles, dim)`` and ``extent = (u_min, u_max, v_min, v_max)``
  (a scalar ``a`` means ``(-a, a, -a, a)``). Returns ``(u, v, sign, logabs)``: 1-D ``u, v`` and
  ``(n, n)`` grids indexed ``[v, u]``, ready for ``imshow`` and ``contour``.
  """
  u0, u1, v0, v1 = plane_extent(extent)
  u, v = jnp.linspace(u0, u1, n), jnp.linspace(v0, v1, n)
  R = base + u[None, :, None, None] * d1 + v[:, None, None, None] * d2
  s, l = evaluate(psi, R)
  return onp.asarray(u), onp.asarray(v), onp.asarray(s), onp.asarray(l)


def particle_volume(psi, R, i, extent, n=48):
  r"""psi on an ``n^3`` grid of positions of particle ``i`` in 3D, the others frozen at ``R``.

  ``extent = (lo, hi)`` in every direction (a scalar ``a`` means ``(-a, a)``). Returns
  ``(x, sign, logabs)``: the 1-D grid coordinates and ``(n, n, n)`` arrays indexed ``[x, y, z]``.
  """
  lo, hi = (-extent, extent) if onp.ndim(extent) == 0 else extent
  x = jnp.linspace(lo, hi, n)
  pos = jnp.stack(jnp.meshgrid(x, x, x, indexing="ij"), axis=-1)
  R = jnp.asarray(R, dtype=float)
  Rs = jnp.broadcast_to(R, (n, n, n) + R.shape).at[..., i, :].set(pos)
  s, l = evaluate(psi, Rs)
  return onp.asarray(x), onp.asarray(s), onp.asarray(l)


def count_pockets(sign):
  r"""Number of connected positive and negative regions of a sign grid, ``(n_pos, n_neg)``.

  Grid points next to the node are dropped before labeling, so regions that meet only where nodal
  lines cross, or through a neck narrower than about two grid spacings, count separately. Pockets
  of a slice are not nodal cells: regions separated in the slice may connect through the other
  dimensions of configuration space.
  """
  from scipy import ndimage
  return tuple(ndimage.label(ndimage.binary_erosion(mask, border_value=1))[1]
               for mask in (sign > 0, sign < 0))
