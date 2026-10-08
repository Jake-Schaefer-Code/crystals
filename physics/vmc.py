# physics/vmc.py
r"""Variational Monte Carlo: sampling, local energy, and energy gradients.

Written against the ``physics.nodal`` configuration convention -- ``R`` of shape
``(n_particles, dim)``, batched as ``(n_walkers, n_particles, dim)`` -- and against a trial
function given as ``log_psi(params, R) -> log|psi|``. Nothing below is one-dimensional except the
optional ``chamber`` constraint, so a different ansatz in any dimension drops straight in.

Sampling in the chamber
-----------------------
For a 1-D ansatz built on the Vandermonde (``physics.ansatz1d``) the wavefunction is strictly
positive inside the Weyl chamber ``x_1 < ... < x_n``, so restricting the walk there removes the
sign from the computation entirely. Gaussian proposals with out-of-chamber rejection keep detailed
balance without a correction, because the proposal and the chamber indicator are both symmetric in
``(R, R')``. Sorting the proposal instead is also correct -- the sorted configuration is the
unique chamber representative of its orbit -- and accepts more often; rejection is the default
because it is unambiguous.

This is a different tool from ``physics.nodal.metropolis``, which samples a fixed wavefunction
with single-particle moves to explore a nodal cell. This one drives optimization.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp


# --------------------------------------------------------------------------- #
# Constraints
# --------------------------------------------------------------------------- #

def chamber(R):
  r"""True when the 1-D coordinates of ``R`` are strictly increasing."""
  x = R[..., 0]
  return jnp.all(x[..., 1:] > x[..., :-1], axis=-1)


def unconstrained(R):
  r"""Accept every configuration; the default for dimensions above one."""
  return jnp.ones(R.shape[:-2], dtype=bool)


# --------------------------------------------------------------------------- #
# Metropolis
# --------------------------------------------------------------------------- #

@functools.lru_cache(maxsize=32)
def _jit_metropolis(log_psi, n_sweeps, constraint):
  r"""Compile a sweeper for one ``(log_psi, n_sweeps, constraint)`` triple.

  Cached on the callables the way ``physics.nodal`` caches its evaluators, which keeps the static
  arguments out of ``jax.jit`` and so out of the positional-versus-keyword trap.
  """

  def run(params, R, key, step):
    def sweep(carry, sub):
      R, acc = carry
      k_prop, k_acc = jax.random.split(sub)
      prop = R + step * jax.random.normal(k_prop, R.shape)

      lp_old = jax.vmap(log_psi, in_axes=(None, 0))(params, R)
      # Evaluate the proposal on its chamber representative so log_psi stays finite even where
      # the proposal is about to be rejected; inside the chamber the sort is the identity.
      safe = jnp.sort(prop, axis=1)
      lp_new = jax.vmap(log_psi, in_axes=(None, 0))(params, safe)

      ok = constraint(prop)
      u = jnp.log(jax.random.uniform(k_acc, lp_old.shape))
      accept = ok & (u < 2.0 * (lp_new - lp_old))

      R = jnp.where(accept[:, None, None], prop, R)
      return (R, acc + jnp.mean(accept)), None

    keys = jax.random.split(key, n_sweeps)
    (R, acc), _ = jax.lax.scan(sweep, (R, 0.0), keys)
    return R, acc / n_sweeps

  return jax.jit(run)


def metropolis(log_psi, params, R, key, n_sweeps=1, step=0.1, constraint=chamber):
  r"""All-particle Gaussian Metropolis on ``|psi|^2``, returning ``(R, acceptance)``.

  ``R`` has shape ``(n_walkers, n_particles, dim)``. Proposals leaving the region allowed by
  ``constraint`` are rejected; with the default ``chamber`` constraint the walk never leaves the
  fundamental domain and psi keeps one sign. Pass stable ``log_psi`` and ``constraint`` objects
  across calls, since each distinct pair compiles once.
  """
  return _jit_metropolis(log_psi, n_sweeps, constraint)(params, R, key, step)


# --------------------------------------------------------------------------- #
# Local energy
# --------------------------------------------------------------------------- #

def laplacian(log_psi, params, R):
  r"""``sum_k d^2/dR_k^2 log|psi|``, by one forward-over-reverse pass per coordinate."""
  shape = R.shape
  flat = R.reshape(-1)
  grad_flat = jax.grad(lambda v: log_psi(params, v.reshape(shape)))
  eye = jnp.eye(flat.size, dtype=R.dtype)

  def diag(e):
    _, hv = jax.jvp(grad_flat, (flat,), (e,))
    return jnp.dot(hv, e)

  return jnp.sum(jax.vmap(diag)(eye))


def local_energy(log_psi, params, R, potential):
  r"""``E_L = -1/2 (lap log|psi| + |grad log|psi||^2) + V(R)``, in Hartree atomic units."""
  g = jax.grad(lambda v: log_psi(params, v))(R)
  lap = laplacian(log_psi, params, R)
  return -0.5 * (lap + jnp.sum(g ** 2)) + potential(R)


def batched_local_energy(log_psi, params, R, potential):
  r"""``local_energy`` over a batch of walkers of shape ``(n_walkers, n_particles, dim)``."""
  return jax.vmap(local_energy, in_axes=(None, None, 0, None))(log_psi, params, R, potential)


# --------------------------------------------------------------------------- #
# Energy gradient
# --------------------------------------------------------------------------- #

def energy_and_grad(log_psi, params, R, potential, clip_mad=5.0):
  r"""Return ``(mean energy, variance, parameter gradient)``.

  The VMC gradient is ``grad E = 2 <(E_L - <E_L>) grad_theta log|psi|>``. The centered local
  energy is clipped at ``clip_mad`` median absolute deviations before entering the gradient:
  local-energy distributions have heavy tails from near-coalescence configurations, and without
  clipping a handful of walkers dominate and the optimization walks away from the minimum. The
  reported energy and variance are unclipped.
  """
  e_loc = batched_local_energy(log_psi, params, R, potential)
  e_mean = jnp.mean(e_loc)
  e_var = jnp.var(e_loc)

  med = jnp.median(e_loc)
  mad = jnp.mean(jnp.abs(e_loc - med))
  clipped = jnp.clip(e_loc, med - clip_mad * mad, med + clip_mad * mad)
  centered = jax.lax.stop_gradient(clipped - jnp.mean(clipped))

  def surrogate(p):
    lp = jax.vmap(log_psi, in_axes=(None, 0))(p, R)
    return 2.0 * jnp.mean(centered * lp)

  return e_mean, e_var, jax.grad(surrogate)(params)


# --------------------------------------------------------------------------- #
# Adam
# --------------------------------------------------------------------------- #

def adam_init(params):
  r"""Zeroed first and second moment trees."""
  zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
  return {"m": zeros, "v": zeros, "t": 0}


def adam_update(state, params, grads, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8, frozen=()):
  r"""One Adam step, optionally zeroing the gradient of named top-level groups.

  ``frozen`` exists for the cusp exponent, and the reason is the clipping above rather than
  anything wrong with the variational principle.

  Near a two-particle coalescence a trial function behaving as ``r^alpha`` against a ``g / r^2``
  interaction leaves ``E_L`` with a residual ``(g - alpha(alpha - 1)) / r^2``. Under the sampling
  weight ``r^{2 alpha}``, the mean of ``E_L`` converges for ``alpha > 1/2`` and its variance only
  for ``alpha > 3/2``. So in the window ``1/2 < alpha < 3/2`` the true expectation is finite and
  obeys ``<E_L> >= E_0``, but it has infinite variance: no central limit theorem, no meaningful
  error bar, and a sample mean dominated by rare excursions toward coalescence.

  Clipping is the standard cure for that tail, and here it is also the poison. The entire excess
  over ``E_0`` lives in the tail, so removing the tail removes the penalty. Measured on
  Calogero-Sutherland at ``n = 3``, ``lambda = 2`` (``E_0 = 7.5``, cusp at ``alpha = 2``), with
  equilibrated walkers:

      alpha           1.60     1.80     1.90     2.00     2.10     2.30
      true mean     7.7104   7.5476   7.5109   7.5000   7.5094   7.5750
      5-MAD mean    7.6235   7.5146   7.4972   7.5000   7.5197   7.5976
      median        7.3728   7.4103   7.4498   7.5000   7.5599   7.7031

  The true mean is minimized at the cusp and equals ``E_0`` there. The clipped mean is minimized
  at ``alpha = 1.90`` and the median at ``alpha = 1.60``, both *below* ``E_0``. A gradient built
  on either one is therefore pulled off the cusp, correctly optimizing the wrong objective.

  With alpha pinned, nothing else in ``physics.ansatz1d`` can produce a ``1 / r^2`` term, the
  tail disappears, and clipping stops mattering: the same run reaches ``7.50147`` with
  ``clip_mad = 5`` and ``7.50107`` with clipping effectively off.
  """
  if frozen:
    grads = dict(grads)
    for key in frozen:
      grads[key] = jax.tree_util.tree_map(jnp.zeros_like, grads[key])

  t = state["t"] + 1
  m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, state["m"], grads)
  v = jax.tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, state["v"], grads)
  m_hat = jax.tree_util.tree_map(lambda a: a / (1 - b1 ** t), m)
  v_hat = jax.tree_util.tree_map(lambda a: a / (1 - b2 ** t), v)
  params = jax.tree_util.tree_map(
    lambda p, a, b: p - lr * a / (jnp.sqrt(b) + eps), params, m_hat, v_hat
  )
  return {"m": m, "v": v, "t": t}, params


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

@functools.lru_cache(maxsize=32)
def _jit_energy_and_grad(log_psi, potential, clip_mad):
  r"""Compile ``energy_and_grad`` for one ``(log_psi, potential, clip_mad)`` triple.

  Without this the optimization loop re-traces a vmapped Laplacian every iteration, which is the
  difference between seconds and minutes for a few hundred steps.
  """
  return jax.jit(lambda params, R: energy_and_grad(log_psi, params, R, potential, clip_mad))


def optimize(log_psi, params, R, key, potential, n_steps=1000, n_sweeps=4, step=0.25,
             lr=5e-3, frozen=(), constraint=chamber, clip_mad=5.0, callback=None):
  r"""Run VMC optimization, returning ``(params, R, history)``.

  ``history`` is a list of ``(step, energy, variance, acceptance)``, and ``callback`` if given is
  called with that tuple each iteration. ``log_psi``, ``potential`` and ``constraint`` are cached
  on, so keep them stable across calls.
  """
  energy_fn = _jit_energy_and_grad(log_psi, potential, clip_mad)
  opt = adam_init(params)
  history = []
  for i in range(n_steps + 1):
    key, sub = jax.random.split(key)
    R, acc = metropolis(log_psi, params, R, sub, n_sweeps, step, constraint)
    energy, variance, grads = energy_fn(params, R)
    opt, params = adam_update(opt, params, grads, lr=lr, frozen=frozen)
    row = (i, float(energy), float(variance), float(acc))
    history.append(row)
    if callback is not None:
      callback(row)
  return params, R, history
