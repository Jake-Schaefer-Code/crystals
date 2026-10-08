# src/invariants.py
r"""Invariant theory of the symmetric group acting on the line: Vandermonde, chamber, features.

In one dimension the whole of fermionic antisymmetry is one polynomial factor. S_n acts on R^n by
reflections in the hyperplanes ``x_i = x_j``, the Vandermonde ``V(x) = prod_{i<j} (x_j - x_i)`` is
the product of the linear forms cutting those mirrors, and an antisymmetric polynomial vanishes on
every mirror and so is divisible by all of them::

  psi(x) = V(x) S(x),   S symmetric

By the fundamental theorem of symmetric polynomials the invariant ring is free,
``C[x_1..x_n]^{S_n} = C[e_1..e_n]``, so ``psi = V * f(e_1, ..., e_n)`` with ``f`` an arbitrary
function of n variables. There is no determinant and no constraint on ``f``: this parametrizes the
antisymmetric sector exactly and completely.

The statement is special to one dimension. For the diagonal action of S_n on (R^d)^n with d > 1 the
sign-isotypic component is not free of rank one over the invariants: ``x_j - x_i`` is a scalar that
carries a sign, while ``r_j - r_i`` is a vector and ``|r_j - r_i|`` is symmetric, so no single
universal antisymmetric factor exists.

Everything here takes a 1-D array ``x`` of shape ``(n,)``. Use ``src.invariants.positions(R)`` to
pull the coordinate out of a ``(n, 1)`` configuration in the ``physics.nodal`` convention.

Conditioning
------------
``elementary_symmetric`` is the exact free-generating basis and is the wrong thing to feed a
network: ``e_k`` spans roughly n! in dynamic range and overflows around n = 15. Use it for
small-n algebraic checks. For features use ``symmetric_features``, whose every component is
smooth and bounded as two coordinates coalesce -- a requirement, not a preference, because
anything log-singular at coalescence lets a variational optimizer manufacture its own Vandermonde
power and violate the cusp condition.
"""
from __future__ import annotations

from typing import Sequence

import numpy as onp
import jax
import jax.numpy as jnp
from jax import lax


# --------------------------------------------------------------------------- #
# Chamber
# --------------------------------------------------------------------------- #

def positions(R):
  r"""The 1-D coordinate of a configuration ``R`` of shape ``(n, 1)`` or ``(n,)``."""
  R = jnp.asarray(R)
  return R[..., 0] if R.ndim >= 2 else R


def in_chamber(x):
  r"""True when ``x_1 < x_2 < ... < x_n``, i.e. x lies in the open Weyl chamber."""
  x = jnp.asarray(x)
  return jnp.all(x[..., 1:] > x[..., :-1], axis=-1)


def inversions(x):
  r"""Number of pairs ``i < j`` with ``x_i > x_j``.

  The length function of the sorting permutation in the Coxeter presentation of S_n by adjacent
  transpositions, counted in O(n^2) so it stays jittable.
  """
  x = jnp.asarray(x)
  gt = x[..., :, None] > x[..., None, :]
  iu = jnp.triu_indices(x.shape[-1], k=1)
  return jnp.sum(gt[..., iu[0], iu[1]], axis=-1)


def sort_sign(x):
  r"""Sign of the permutation that sorts ``x`` ascending, as ``+1`` or ``-1``."""
  return 1 - 2 * (inversions(x) % 2)


def to_chamber(x):
  r"""Return ``(x_sorted, sign)``: the chamber representative of x and the sign picked up.

  Every S_n orbit of distinct coordinates meets the open chamber exactly once, so sorting is the
  canonical projection onto the fundamental domain. An antisymmetric function is then recovered
  from its chamber restriction by ``psi(x) = sign * psi_chamber(x_sorted)``.
  """
  x = jnp.asarray(x)
  return jnp.sort(x, axis=-1), sort_sign(x)


# --------------------------------------------------------------------------- #
# Vandermonde
# --------------------------------------------------------------------------- #

def _vandermonde(x):
  x = jnp.asarray(x)
  d = x[..., None, :] - x[..., :, None]
  iu = jnp.triu_indices(x.shape[-1], k=1)
  return d[..., iu[0], iu[1]]

def log_vandermonde(x):
  r"""``sum_{i<j} log(x_j - x_i)`` for x sorted ascending.

  Summed in the log domain rather than formed as a product and then logged, which overflows for
  moderate n. Inside the chamber every gap is positive, so no absolute value and no sign.
  """
  return jnp.sum(jnp.log(_vandermonde(x)), axis=-1)


def log_abs_vandermonde(x):
  r"""``sum_{i<j} log|x_j - x_i|`` for arbitrary (unsorted) x."""
  return jnp.sum(jnp.log(jnp.abs(_vandermonde(x))), axis=-1)


def grad_log_vandermonde(x):
  r"""``d/dx_k log V = sum_{j != k} 1 / (x_k - x_j)``, in closed form.

  Better behaved than differentiating through the log near coalescence, and useful as an oracle
  against the autodiff path in tests.
  """
  x = jnp.asarray(x)
  n = x.shape[-1]
  d = x[..., :, None] - x[..., None, :]
  inv = jnp.where(jnp.eye(n, dtype=bool), 0.0, 1.0 / jnp.where(jnp.eye(n, dtype=bool), 1.0, d))
  return jnp.sum(inv, axis=-1)


def vandermonde(x):
  r"""The signed Vandermonde ``prod_{i<j} (x_j - x_i)``, evaluated directly.

  Only for small n and for tests; ``log_vandermonde`` is the numerically sane route.
  """
  return jnp.prod(_vandermonde(x), axis=-1)


# --------------------------------------------------------------------------- #
# Generators of the invariant ring
# --------------------------------------------------------------------------- #

def elementary_symmetric(x):
  r"""``[e_0, e_1, ..., e_n]``, the coefficients of ``prod_i (t + x_i) = sum_k e_k t^{n-k}``.

  Built by the O(n^2) convolution recurrence ``e_k <- e_k + x_i e_{k-1}``, never as an explicit
  sum over subsets. These freely generate the invariant ring and span about n! in magnitude; see
  the module docstring before feeding them to anything.
  """
  x = jnp.asarray(x)
  n = x.shape[-1]

  def step(e, xi):
    shifted = jnp.concatenate([jnp.zeros_like(e[:1]), e[:-1]])
    return e + xi * shifted, None

  e0 = jnp.zeros(n + 1, dtype=x.dtype).at[0].set(1.0)
  e, _ = lax.scan(step, e0, x)
  return e


def power_sums(x, kmax):
  r"""``[p_1, ..., p_kmax]`` with ``p_k = sum_i x_i^k``."""
  x = jnp.asarray(x)
  ks = jnp.arange(1, kmax + 1, dtype=x.dtype)
  return jnp.sum(x[..., None, :] ** ks[:, None], axis=-1)


def elementary_from_power_sums(p, n: int):
  r"""Elementary symmetric polynomials from power sums by Newton's identities.

  ``k e_k = sum_{i=1}^{k} (-1)^{i-1} e_{k-i} p_i``. Valid in characteristic zero only. Present as
  an independent route to ``elementary_symmetric`` for cross-checking, not for production use.
  """
  e = [jnp.ones((), dtype=p.dtype)]
  for k in range(1, n + 1):
    acc = jnp.zeros((), dtype=p.dtype)
    for i in range(1, k + 1):
      acc = acc + ((-1) ** (i - 1)) * e[k - i] * p[i - 1]
    e.append(acc / k)
  return jnp.stack(e)


# --------------------------------------------------------------------------- #
# Hilbert series of the sign-isotypic module
# --------------------------------------------------------------------------- #

def antisymmetric_dimensions(n: int, dmax: int) -> list[int]:
  r"""Dimensions of the degree-d antisymmetric polynomials in n variables, for d = 0..dmax.

  The sign-isotypic component of the polynomial ring is free of rank one over the invariants,
  generated by the Vandermonde, so its Hilbert series is::

    sum_d dim_d q^d  =  q^{n(n-1)/2} / prod_{k=1}^{n} (1 - q^k)

  The numerator is the degree of the Vandermonde and the denominator is the free invariant ring.
  ``src.symmetry.young.molien_sign`` computes the same integers from the character table of S_n;
  agreement between the two is a joint test of this module and that one.
  """
  coeffs = [0] * (dmax + 1)
  offset = n * (n - 1) // 2
  if offset <= dmax:
    coeffs[offset] = 1
  for k in range(1, n + 1):
    for d in range(k, dmax + 1):
      coeffs[d] += coeffs[d - k]
  return coeffs


# --------------------------------------------------------------------------- #
# Feature maps
# --------------------------------------------------------------------------- #

def scaled_power_sums(x, kmax: int = 4, scale: float = 2.0):
  r"""``p_k / (n scale^k)`` for k = 1..kmax: power sums rescaled to O(1).

  Not ``sign(p) |p|^{1/k}``. The k-th root looks like the natural way to undo degree-k growth, but
  its derivative diverges as ``p_k -> 0`` and the centre of mass ``p_1`` crosses zero constantly,
  which injects a spurious singularity into the local energy. Linear rescaling is smooth
  everywhere and works as well.
  """
  x = jnp.asarray(x)
  n = x.shape[-1]
  ks = jnp.arange(1, kmax + 1, dtype=x.dtype)
  return power_sums(x, kmax) / (n * scale ** ks)


def gap_features(x):
  r"""Four smooth bounded summaries of the gap structure, for x sorted ascending.

  Gaps are the distances to the reflection walls and so are the natural chamber coordinates, but
  every feature built from them must stay smooth as a gap closes. ``log(gap)`` and ``min(gap)``
  are the obvious choices and both are wrong: a network given ``log(gap)`` can manufacture an
  effective Vandermonde power and break the cusp condition, and ``min`` is not differentiable
  where two gaps cross. All short-range behaviour belongs to the Vandermonde exponent alone.
  """
  x = jnp.asarray(x)
  gaps = x[..., 1:] - x[..., :-1]
  return jnp.stack([
    jnp.mean(gaps, axis=-1),
    jnp.mean(jnp.exp(-gaps), axis=-1),
    -jnp.log(jnp.sum(jnp.exp(-gaps), axis=-1)),
    x[..., -1] - x[..., 0],
  ], axis=-1)


def symmetric_features(x, kmax: int = 4, scale: float = 2.0):
  r"""Concatenated scaled power sums and gap summaries: the default network input.

  Shape ``(kmax + 4,)``. Every component is S_n-invariant by construction, so a network applied to
  these is automatically symmetric and no permutation symmetry has to be imposed on it.
  """
  return jnp.concatenate([scaled_power_sums(x, kmax, scale), gap_features(x)], axis=-1)


# --------------------------------------------------------------------------- #
# Schur functions
# --------------------------------------------------------------------------- #

def schur(partition: Sequence[int], x):
  r"""``s_lambda(x) = det(x_i^{lambda_j + n - j}) / V(x)`` by the bialternant formula.

  The Vandermonde factorization made explicit for free fermions: the numerator is a Slater
  determinant of monomial orbitals, the denominator is the Vandermonde, and the quotient is the
  symmetric factor. For a free-fermion state this gives the exact symmetric function a variational
  ansatz should learn, pointwise and in closed form, indexed by a partition.

  ``partition`` is weakly decreasing of length n.
  """
  x = jnp.asarray(x)
  n = x.shape[-1]
  lam = jnp.asarray(partition, dtype=x.dtype)
  if lam.shape[-1] != n:
    raise ValueError(f"partition has length {lam.shape[-1]}, expected {n}")
  exps = lam[::-1] + jnp.arange(n, dtype=x.dtype)
  num = jnp.linalg.det(x[..., :, None] ** exps)
  den = jnp.linalg.det(x[..., :, None] ** jnp.arange(n, dtype=x.dtype))
  return num / den
