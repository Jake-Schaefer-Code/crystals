# physics/markov.py
r"""Entropy and relative entropy on distributions, and the mismatch-cost bounds for a repeated map.

Everything takes a probability vector ``p`` (or batch along an axis) and, where there is dynamics, a
column-stochastic matrix ``G`` acting as ``p -> G @ p``. Entropies use ``0 log 0 = 0``.

- ``H``, ``KL``, ``JS``: entropy, relative entropy, Jensen-Shannon-type spread.
- ``matrix_powers``, ``get_traj``: ``[I, G, ..., G^(N-1)]`` and the orbit ``p, Gp, ..., G^N p``.
- ``get_q_star``, ``bound``, ``bound2``: the time-averaged prior ``q*`` and the generalized
  Landauer bound it gives.
- ``mmc``, ``pmmc``, ``pmmc2``, ``pop``: mismatch cost of a fixed prior, and of the optimal periodic
  prior (``pop`` returns the averaged distribution that is that prior).

- ``periodic_mmc_curves``: the orbit, ``pmmc`` for every horizon, and the stationary state in one call.

Moved from ``notebooks/stoch_thermo/utils.py``; ``pmmc`` now follows the dtype of ``G`` and
``periodic_mmc_curves`` is the helper that ``DFA.ipynb`` and ``prior.ipynb`` each carried as ``get_stuff``.
"""
from __future__ import annotations

import jax
from jax import Array, lax
import jax.numpy as jnp
from jax.scipy.special import xlogy
from functools import partial
import numpy as onp

# ---- entropy and KL divergence use convention that 0 log 0 = 0
# TODO make custom vjp since both branches can participate in autodiff tracing, which is unpleasant
# (see jax xlogy vjp)
def H(p, axis=None):
  r""" Entropy """
  return - jnp.sum(jnp.where(p != 0., xlogy(p, p), jnp.zeros_like(p)), axis=axis)
S = H
Hv = lambda p: H(p, axis=1)
r""" Vectorized entropy """

def KL(p, q, axis=None):
  r""" Relative entropy """
  return jnp.sum(jnp.where(p != 0., xlogy(p, p) - xlogy(p,q), jnp.zeros_like(p)), axis=axis)

KLv = lambda p, q: KL(p, q, axis=1)
r""" Vectorized relative entropy """

def KL2(p, q, axis=None):
  r""" Relative entropy V2 TODO """
  p_ok = p != 0.
  return jnp.sum(jnp.where(p_ok, lax.mul(p, lax.log(p)) - lax.mul(p, lax.log(q)), jnp.zeros_like(p)), axis=axis)


def JS(distributions, base=jnp.e):
  P = jnp.asarray(distributions, dtype=jnp.float32)
  P /= P.sum(axis=1, keepdims=True)
  N = P.shape[0]
  return N * H(P.mean(axis=0)) - jnp.sum(jax.vmap(H)(P))

def matrix_powers(G: Array, N: int):
  def step(Gk: Array, _):
    return Gk @ G, Gk

  _, Gs = lax.scan(
    step,
    init=jnp.eye(G.shape[0], dtype=G.dtype),
    xs=None,
    length=N,
  )
  return Gs  # [I, G, G^2, ..., G^(N-1)]

# ---- optimal prior
def get_q_star(p: Array, G: Array, n_steps: int):
  Gs = matrix_powers(G, n_steps)
  traj = jax.vmap(lambda G_i: G_i @ p)(Gs)
  return jnp.mean(traj, axis=0)

def bound(p: Array, G: Array, n_steps: int):
  Gs = matrix_powers(G, n_steps)
  Gn = G @ Gs[-1]
  traj = jax.vmap(lambda G_i: G_i @ p)(Gs)
  q_star = jnp.mean(traj, axis=0)
  gen_landauer =  S(p) - S(Gn @ p)
  return -gen_landauer + n_steps * (S(q_star) - S(G @ q_star)), gen_landauer


def bound2(p: Array, G: Array, n_steps: int, *, scan_G: bool=False):
  def _step(carry, _):
    p_i, agg = carry
    return (G @ p_i, agg + p_i), None

  def _step_G(carry, _):
    Gi, agg = carry
    return (G @ Gi, agg + Gi @ p), None

  d = p.shape[0]
  if scan_G:
    init = (jnp.eye(d, d), jnp.zeros((d,)))
    step = _step_G
  else:
    init = (p, jnp.zeros((d,)))
    step = _step

  out, _ = lax.scan(step, init=init, xs=None, length=n_steps)
  # p_n = G^np
  if scan_G:
    Gn, traj_f = out
    q_star = traj_f / n_steps
    gen_landauer =  S(p) - S(Gn @ p)
  else:
    p_n, traj_f = out
    q_star = traj_f / n_steps
    gen_landauer =  S(p) - S(p_n)
  return -gen_landauer + n_steps * (S(q_star) - S(G @ q_star)), gen_landauer

  

def mmc(
  q: Array, 
  G: Array,
  p0: Array,
  n_steps: int,
):
  Gq = G @ q
  # Gs = matrix_powers(G, n_steps)
  def mmc1(p_i, _):
    p_next = G @ p_i
    cost_i = (
      KL(p_i, q) - KL(p_next, Gq)
    )
    return p_next, cost_i
  
  _, costs = lax.scan(mmc1, init=p0, xs=None, length = n_steps)
  return jnp.sum(costs)


# unjitted, and constructing this function in the T loop, it is much slower than just a 
# for loop with a numpy array. However, jitted beforehand, it is much faster -> 2-3 orders of magnitude faster
@partial(jax.jit, static_argnames=("N_max",))
def get_traj(G, p0, N_max):
  # p <- G @ p
  def traj(p, x):
    return G @ p, p # one mat-vec per step, not mat-mat
  pN, pts = lax.scan(traj, init=p0, length=N_max)
  return pN, jnp.concatenate((pts, pN[None]))

def pmmc(p0, G, N: int):
  # Carry dtype must match G @ p, so follow G rather than forcing float32 (which broke under x64).
  p0 = jnp.asarray(p0)
  p0 = p0.astype(jnp.result_type(p0, G, jnp.float32))

  if N == 0:
    return jnp.array(0.0)

  p = p0.copy()

  def body(i, carry):
    p_avg, p = carry
    p_avg = p_avg + p
    return (p_avg, G @ p)

  p_avg, p = lax.fori_loop(0, N, body, (jnp.zeros_like(p0), p))
  p_avg = p_avg / N

  return H(p) - H(p0) + N * (H(p_avg) - H(G @ p_avg))


def pop(p0, G, N):
  p_avg = jnp.zeros_like(p0, dtype=float)
  if N==0:
    return jnp.zeros_like(p0)
  # sparse matrices font work with mat power
  p = p0.copy()
  for _ in range(N):
    p_avg += p
    p = G @ p

  return p_avg / N


def pmmc2(p0, G, N):
  p_avg = pop(p0, G, N)
  return (
    H(jnp.linalg.matrix_power(G, N) @ p0) - H(p0) + N * (H(p_avg) - H(G @ p_avg))
  )


def periodic_mmc_curves(G, p0, N_max: int):
  r"""Orbit, periodic mismatch cost for ``N = 1..N_max``, stationary state and its distance to ``p0``.

  Returns ``(pts, pmmcs, pi, kl)``: ``pts`` is the orbit ``p_0..p_{N_max}`` (``N_max + 1`` rows),
  ``pmmcs[n-1] = H(p_n) - H(p_0) + n (H(avg_{<n}) - H(G avg_{<n}))``, ``pi`` the eigenvector of ``G``
  nearest eigenvalue 1 (clipped to be nonnegative and normalized) and ``kl = KL(p0, pi)``.
  """
  Ns = jnp.arange(1, N_max + 1)
  pN, pts = get_traj(G, p0, N_max)
  p_avgs = jnp.cumsum(pts[:-1], axis=0) / Ns[:, None]
  Gp_avgs = p_avgs @ G.T
  landauer = - H(pts[1:], axis=1) + H(p0)
  pmmcs = - landauer + Ns * (H(p_avgs, axis=1) - H(Gp_avgs, axis=1))
  vals, vecs = onp.linalg.eig(G)
  idx = onp.argmin(onp.abs(vals - 1.0))
  pi = onp.real(vecs[:, idx])
  pi = onp.clip(pi / pi.sum(), 0, None)
  kl = KL(p0, pi)
  return pts, pmmcs, pi, kl
