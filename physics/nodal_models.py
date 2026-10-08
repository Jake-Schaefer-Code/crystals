# physics/nodal_models.py
r"""Real antisymmetric wavefunctions with known nodal structure, for ``physics.nodal``.

Every model returns ``psi(R) -> (sign, log|psi|)`` for a single configuration ``R`` of shape
``(n_particles, dim)``; particle ``i`` has spin ``spins[i]`` (0 = up, 1 = down).

Ground truth worth testing against:

- ``harmonic_orbitals(1, n)``: psi is the Vandermonde product prod_{i<j} (x_j - x_i) times a
  Gaussian, so the node is exactly the coincidence planes x_i = x_j and there are n! cells.
- ``harmonic_orbitals(2, 3)``, three same-spin particles: psi is proportional to the signed area of
  the triangle r_0 r_1 r_2, so the node is "the three particles are collinear" (2 cells).
- ``harmonic_orbitals(3, 4)``, four same-spin particles: the signed volume of the tetrahedron.
- ``beryllium(0)``: node |r_0| = |r_1| or |r_2| = |r_3| (4 cells); any ``c_2p != 0`` mixes in
  1s^2 2p^2 and the cells fuse to 2 [Bressanini, Ceperley & Reynolds, "What do we know about wave
  function nodes?", Recent Advances in Quantum Monte Carlo Methods II (2002)].
"""
from __future__ import annotations

import itertools as it

import numpy as onp
import jax.numpy as jnp


def slater_determinant(orbitals, spins):
  r"""``psi(R) = det[phi_j(r_i)]_{i in up} * det[phi_j(r_i)]_{i in down}``.

  ``orbitals`` maps positions ``(m, dim)`` to values ``(m, n_orb)``; each spin channel occupies
  its lowest ``n_sigma`` orbitals.
  """
  spins = onp.asarray(spins)
  up, dn = onp.flatnonzero(spins == 0), onp.flatnonzero(spins == 1)

  def psi(R):
    s_up, l_up = jnp.linalg.slogdet(orbitals(R[up])[:, :len(up)])
    s_dn, l_dn = jnp.linalg.slogdet(orbitals(R[dn])[:, :len(dn)])
    return s_up * s_dn, l_up + l_dn

  return psi


def hermite_functions(x, n_max):
  r"""Normalized Hermite functions ``psi_0 .. psi_{n_max}`` at ``x``, stacked on a new last axis."""
  out = [jnp.pi ** -0.25 * jnp.exp(-0.5 * x ** 2)]
  if n_max > 0:
    out.append(jnp.sqrt(2.0) * x * out[0])
  for n in range(1, n_max):
    out.append(jnp.sqrt(2.0 / (n + 1)) * x * out[n] - jnp.sqrt(n / (n + 1)) * out[n - 1])
  return jnp.stack(out, axis=-1)


def harmonic_orbitals(dim, n_orb, omega=1.0):
  r"""Lowest ``n_orb`` Cartesian eigenfunctions of the isotropic oscillator in ``dim`` dimensions.

  Ordered by shell ``n_1 + ... + n_dim``. Closed shells (non-degenerate ground states) hold
  1, 3, 6, 10 orbitals for ``dim = 2`` and 1, 4, 10, 20 for ``dim = 3``.
  """
  quanta = sorted(it.product(range(n_orb), repeat=dim), key=lambda q: (sum(q), [-k for k in q]))
  quanta = onp.array(quanta[:n_orb]).reshape(n_orb, dim)
  n_max = int(quanta.max(initial=0))
  scale = onp.sqrt(omega)

  def orbitals(r):
    h = hermite_functions(scale * r, n_max)                         # (m, dim, n_max + 1)
    return omega ** (dim / 4) * jnp.prod(h[:, onp.arange(dim), quanta], axis=-1)

  return orbitals


def plane_wave_orbitals(dim, n_orb, L):
  r"""Lowest ``n_orb`` real plane waves ``1, cos(k.r), sin(k.r)``, ``k = 2 pi n / L``, in ``[0, L)^dim``.

  Ordered by ``|n|``. Closed shells hold 1, 5, 9, 13, 21 orbitals for ``dim = 2`` and
  1, 7, 19, 27, 33 for ``dim = 3``.
  """
  n_max = int(onp.ceil(n_orb ** (1 / dim))) + 1
  # one representative of each +-n pair: n = 0 or first nonzero component positive
  ns = [n for n in it.product(range(-n_max, n_max + 1), repeat=dim)
        if not any(n) or next(c for c in n if c) > 0]
  funcs = []
  for n in sorted(ns, key=lambda n: (sum(c * c for c in n), n)):
    funcs += [(n, False)] if not any(n) else [(n, False), (n, True)]
  funcs = funcs[:n_orb]
  K = 2 * onp.pi / L * onp.array([n for n, _ in funcs], dtype=float)  # (n_orb, dim)
  use_sin = onp.array([s for _, s in funcs])

  def orbitals(r):
    phase = r @ K.T
    return jnp.where(use_sin, jnp.sin(phase), jnp.cos(phase))

  return orbitals


def beryllium(c_2p=0.0, zeta_1s=3.7, zeta_2=1.0):
  r"""Four-electron Be atom, nucleus at the origin; electrons 0, 1 are up and 2, 3 are down.

  ``psi = D(1s,2s) D(1s,2s) + c_2p * sum_a D(1s,2p_a) D(1s,2p_a)`` where ``D`` is the 2x2 same-spin
  determinant of normalized Slater-type orbitals 1s ~ exp(-zeta_1s r), 2s ~ r exp(-zeta_2 r) and
  2p_a ~ r_a exp(-zeta_2 r). ``c_2p = 0`` is the Hartree-Fock-like single determinant;
  ``c_2p`` around -0.2 mimics the 2s^2 -> 2p^2 near-degeneracy mixing of the real Be ground state.
  """
  n2 = (2 * zeta_2) ** 2.5 / onp.sqrt(96 * onp.pi)

  def orbitals(r):                                                    # (m, 3) -> (m, 5)
    rn = jnp.linalg.norm(r, axis=-1)
    s1 = onp.sqrt(zeta_1s ** 3 / onp.pi) * jnp.exp(-zeta_1s * rn)
    radial = n2 * jnp.exp(-zeta_2 * rn)
    return jnp.column_stack([s1, radial * rn, onp.sqrt(3.0) * radial[:, None] * r])

  def psi(R):
    up, dn = orbitals(R[:2]), orbitals(R[2:])
    val = jnp.linalg.det(up[:, [0, 1]]) * jnp.linalg.det(dn[:, [0, 1]])
    for a in (2, 3, 4):
      val = val + c_2p * jnp.linalg.det(up[:, [0, a]]) * jnp.linalg.det(dn[:, [0, a]])
    return jnp.sign(val), jnp.log(jnp.abs(val))

  return psi
