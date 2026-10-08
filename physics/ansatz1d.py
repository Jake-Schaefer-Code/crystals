# physics/ansatz1d.py
r"""Determinant-free antisymmetric ansatz for one-dimensional fermions.

In one dimension every antisymmetric function factors as Vandermonde times symmetric
(``src.invariants``), so no determinant is needed and no permutation sign has to be tracked::

  log psi(x) = alpha sum_{i<j} log(x_j - x_i)   # Vandermonde: fixes the node exactly
             - (1/2) w sum_i x_i^2              # envelope: makes it normalizable
             + sum_{i<j} u(x_j - x_i)           # learned two-body Jastrow
             + f(symmetric features of x)       # learned many-body term

evaluated in the open Weyl chamber ``x_1 < ... < x_n`` where psi is strictly positive, and
continued to all of configuration space by ``psi(x) = sgn(sort) psi_chamber(sort(x))``. The result
is exactly antisymmetric, so ``physics.nodal.antisymmetry_error`` returns zero on it to machine
precision rather than to a tolerance.

Wavefunctions here follow the ``physics.nodal`` protocol: ``psi(R) -> (sign, log|psi|)`` on a
single configuration ``R`` of shape ``(n, 1)``. Everything in ``nodal`` -- slices, node distances,
``cell_census``, the fixed-node walk -- therefore applies unchanged.

Four constraints, all the same condition
----------------------------------------
The local energy must stay integrable where two particles coalesce. Three of the four ways to
violate it look like optimization choices and are not:

1. ``alpha`` is fixed by the cusp condition, not learned. For a pair interaction ``g / r^2`` the
   two-body problem forces ``psi ~ r^alpha`` with ``alpha(alpha - 1) = g``; away from that value
   the local energy keeps a residual ``(g - alpha(alpha - 1)) / r^2``, which is *positive* below
   the cusp. The variational principle itself survives this: with the walkers equilibrated,
   ``<E_L> >= E_0`` for every ``alpha > 1/2``, with equality only at the cusp. What fails is the
   estimator. The whole penalty for missing the cusp sits in a rare positive tail -- at
   ``alpha = 1`` with ``g = 2``, nine percent of walkers report above ``2 E_0`` -- and the
   variance of ``E_L`` is infinite for ``alpha < 3/2``. The MAD clipping that the gradient needs
   in order to tolerate that tail is exactly what deletes the penalty, so the clipped objective
   is minimized below the cusp at a value below ``E_0``. Clipping and learning alpha are
   incompatible. Use ``cusp_alpha`` and freeze it; see ``physics.vmc.adam_update`` for numbers.
2. Nothing else may be log-singular at coalescence. Feeding ``log(gap)`` to the Jastrow, or
   ``log(gap)`` or ``min(gap)`` to the feature map, lets the network manufacture its own effective
   Vandermonde power through a side door. ``src.invariants.symmetric_features`` is built to be
   smooth and bounded for this reason.
3. The envelope is not optional: ``alpha log V`` alone is not normalizable, so without it the
   walkers diffuse outward and the trap term drives the energy up without bound.
4. Enable float64. Second derivatives of log psi in single precision leave about 1e-3 of noise in
   the local energy, which hides the zero-variance signal entirely.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

import src.invariants as inv


# --------------------------------------------------------------------------- #
# Cusp condition
# --------------------------------------------------------------------------- #

def cusp_alpha(g):
  r"""Solve ``alpha (alpha - 1) = g`` for the physical root, ``g`` the coefficient of 1/r^2.

  Returns 1 for a non-singular interaction, recovering the free-fermion Vandermonde power, and
  ``lambda`` for a Calogero-Sutherland coupling ``g = lambda (lambda - 1)``.
  """
  return 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * jnp.asarray(g)))


# --------------------------------------------------------------------------- #
# A minimal explicit-parameter MLP
# --------------------------------------------------------------------------- #

def _init_mlp(key, sizes, scale=1.0):
  out = []
  for din, dout in zip(sizes[:-1], sizes[1:]):
    key, sub = jax.random.split(key)
    w = jax.random.normal(sub, (din, dout)) * (scale / jnp.sqrt(din))
    out.append((w, jnp.zeros(dout)))
  return out


def _apply_mlp(mlp, h):
  for w, b in mlp[:-1]:
    h = jnp.tanh(h @ w + b)
  w, b = mlp[-1]
  return h @ w + b


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #

KMAX = 4
N_GAP_FEATURES = 4


def init_params(key, n_particles, alpha=1.0, envelope=1.0, phi_width=16, phi_out=8,
                rho_width=32, jastrow_width=16, dtype=None):
  r"""Initial parameters. ``alpha`` should come from ``cusp_alpha`` and then stay frozen.

  Every leaf is an array, so an optimizer can ``tree_map`` over the whole dict; the feature count
  is a module constant rather than a parameter for the same reason.
  """
  if dtype is None:
    dtype = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32
  k_phi, k_rho, k_jas = jax.random.split(key, 3)
  n_feat = phi_out + KMAX + N_GAP_FEATURES
  return {
    "log_alpha": jnp.log(jnp.asarray(alpha, dtype=dtype)),
    "log_envelope": jnp.log(jnp.asarray(envelope, dtype=dtype)),
    "phi": _init_mlp(k_phi, [1, phi_width, phi_out], scale=1.0),
    "rho": _init_mlp(k_rho, [n_feat, rho_width, rho_width, 1], scale=0.1),
    "jastrow": _init_mlp(k_jas, [2, jastrow_width, 1], scale=0.01),
  }


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #

def _deepsets(phi, x):
  r"""``mean_i phi(x_i)``: the universal, well-conditioned symmetric embedding."""
  h = jax.vmap(lambda xi: _apply_mlp(phi, jnp.atleast_1d(xi)))(x)
  return jnp.mean(h, axis=0)


def _two_body(params, x):
  r"""Smooth two-body Jastrow, deliberately unable to produce a log divergence.

  Inputs are ``(gap, exp(-gap))``, both bounded and smooth as the gap closes, so all short-range
  behaviour is carried by ``alpha`` alone.
  """
  d = x[None, :] - x[:, None]
  iu = jnp.triu_indices(x.shape[0], k=1)
  gaps = d[iu]
  stacked = jnp.stack([gaps, jnp.exp(-gaps)], axis=-1)
  return jnp.sum(jax.vmap(lambda g: _apply_mlp(params["jastrow"], g))(stacked))


def log_psi_chamber(params, x):
  r"""``log psi`` for a sorted 1-D coordinate vector ``x`` of shape ``(n,)``.

  Strictly positive inside the chamber, so this is a genuine logarithm and no sign is involved.
  """
  alpha = jnp.exp(params["log_alpha"])
  feats = jnp.concatenate([
    _deepsets(params["phi"], x),
    inv.scaled_power_sums(x, KMAX),
    inv.gap_features(x),
  ])
  return (
    alpha * inv.log_vandermonde(x)
    - 0.5 * jnp.exp(params["log_envelope"]) * jnp.sum(x ** 2)
    + _two_body(params, x)
    + _apply_mlp(params["rho"], feats).squeeze()
  )


def log_psi(params, R):
  r"""``log|psi|`` on a configuration ``R`` of shape ``(n, 1)``, sorted internally.

  The sampler keeps walkers inside the chamber, so the sort is a no-op there; it is kept so that
  the function is well defined off-chamber and matches ``wavefunction`` exactly.
  """
  x = jnp.sort(inv.positions(R))
  return log_psi_chamber(params, x)


def wavefunction(params):
  r"""Return ``psi(R) -> (sign, log|psi|)`` in the ``physics.nodal`` convention.

  The chamber restriction is continued by ``psi(x) = sgn(sort) psi_chamber(sort(x))``, which is
  antisymmetric by construction: relabeling multiplies the sorting sign by the sign of the
  relabeling and leaves the sorted coordinates alone. Hashable by identity, so the ``lru_cache``
  in ``physics.nodal`` works on it.
  """
  def psi(R):
    x = inv.positions(R)
    x_sorted, sign = inv.to_chamber(x)
    return jnp.asarray(sign, dtype=x.dtype), log_psi_chamber(params, x_sorted)

  return psi


# --------------------------------------------------------------------------- #
# Walkers
# --------------------------------------------------------------------------- #

def init_walkers(key, n_walkers, n_particles, spread=1.0, dim=1):
  r"""Gaussian walkers sorted into the chamber, shape ``(n_walkers, n_particles, dim)``."""
  R = jax.random.normal(key, (n_walkers, n_particles, dim)) * spread
  return jnp.sort(R, axis=1)
