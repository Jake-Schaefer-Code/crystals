# physics/models1d.py
r"""Exactly solvable one-dimensional fermion models, for benchmarking a trial wavefunction.

Calogero-Sutherland is the one to run first, because its exact ground state lies *inside* the
ansatz class of ``physics.ansatz1d``: it is a power of the Vandermonde times a Gaussian, which is
exactly Vandermonde times symmetric. Failure to recover it is therefore a bug in the
implementation rather than a limitation of the ansatz, which makes it a far sharper test than a
system whose answer the ansatz could only approximate.

  H = -1/2 sum_i d^2/dx_i^2 + 1/2 w^2 sum_i x_i^2 + sum_{i<j} lam (lam - 1) / (x_i - x_j)^2

  psi_0(x) = prod_{i<j} |x_i - x_j|^lam exp(-w/2 sum_i x_i^2)
  E_0      = w ( n/2 + lam n(n-1)/2 )

At ``lam = 1`` the interaction vanishes and this is free fermions in a harmonic trap,
``E_0 = w n^2 / 2``, giving a second independent check from the same code path.

Configurations follow the ``physics.nodal`` convention, shape ``(n, 1)``.
"""
from __future__ import annotations

import jax.numpy as jnp

import src.invariants as inv


# --------------------------------------------------------------------------- #
# Calogero-Sutherland
# --------------------------------------------------------------------------- #

def calogero_potential(omega=1.0, lam=1.0):
  r"""Return ``V(R)`` for the harmonic trap plus the inverse-square pair interaction."""
  g = lam * (lam - 1.0)

  def V(R):
    x = inv.positions(R)
    trap = 0.5 * omega ** 2 * jnp.sum(x ** 2)
    d = x[None, :] - x[:, None]
    iu = jnp.triu_indices(x.shape[0], k=1)
    return trap + g * jnp.sum(1.0 / d[iu] ** 2)

  return V


def calogero_energy(n, omega=1.0, lam=1.0):
  r"""Exact ground-state energy ``w (n/2 + lam n(n-1)/2)``."""
  return omega * (n / 2.0 + lam * n * (n - 1) / 2.0)


def calogero_log_psi(omega=1.0, lam=1.0):
  r"""The exact ground state as ``log_psi(params, R)``, ignoring ``params``.

  Shaped like a trial function so it can be dropped into ``physics.vmc`` unchanged, which is how
  the zero-variance property gets tested: feeding the exact state through the same local-energy
  code must return the analytic energy with no scatter.
  """
  def f(params, R):
    x = inv.positions(R)
    return lam * inv.log_abs_vandermonde(x) - 0.5 * omega * jnp.sum(x ** 2)

  return f


def calogero_wavefunction(omega=1.0, lam=1.0):
  r"""The exact ground state in the ``physics.nodal`` protocol, ``psi(R) -> (sign, log|psi|)``."""
  def psi(R):
    x = inv.positions(R)
    x_sorted, sign = inv.to_chamber(x)
    logabs = lam * inv.log_vandermonde(x_sorted) - 0.5 * omega * jnp.sum(x_sorted ** 2)
    return jnp.asarray(sign, dtype=x.dtype), logabs

  return psi


# --------------------------------------------------------------------------- #
# Free fermions
# --------------------------------------------------------------------------- #

def free_fermion_energy(n, omega=1.0):
  r"""``sum_{k=0}^{n-1} w (k + 1/2) = w n^2 / 2`` for n spinless fermions in a harmonic trap."""
  return omega * n ** 2 / 2.0


def harmonic_potential(omega=1.0):
  r"""Return ``V(R) = (1/2) w^2 sum_i x_i^2``."""
  def V(R):
    return 0.5 * omega ** 2 * jnp.sum(inv.positions(R) ** 2)

  return V
