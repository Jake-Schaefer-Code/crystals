# physics/speed_limits.py
r"""Wasserstein-1 distance on the spin hypercube and the activity-based speed limit on state change.

For a continuous-time Markov chain whose transitions each flip one spin, the Hamming distance makes
the state space the hypercube ``{0, 1}^N``. With ``W1`` the Wasserstein-1 distance between the
initial and current distributions, ``<A>_T`` the time-averaged activity (weighted mean of the
symmetrized flip rates) and ``v = W1 / T``, the speed limit is

  ``2 W1 arctanh(v / (2 <A>_T))``,

returned by ``get_speed_limit`` along a trajectory. ``W1`` has two routes: the exact linear program
over 1-Lipschitz potentials (``w1_dual_exact``, with ``hypercube_lipschitz_constraints`` supplying
the edge constraints) and a gradient ascent over Lipschitz-rescaled potentials (``w1_dual``).

Moved from ``notebooks/stoch_thermo/IsingSpins.ipynb``; ``get_speed_limit`` now reads the number of
states from the graph instead of a notebook global.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

GraphT = tuple[Array, Array, Array]


def lipschitz_constant(f, N):
  """
  Lipschitz constant of f on the N-dimensional hypercube
  with Hamming distance.
  """
  states = jnp.arange(1 << N)
  def body(i, L):
    # flip state i
    neighbors = states ^ (1 << i)
    return jnp.maximum(L, jnp.max(jnp.abs(f - f[neighbors])))
  return lax.fori_loop(0, N, body, jnp.array(0.0))


def make_lipschitz(f, N):
  # Fix arbitrary additive constant
  f = f - f[0]
  L = lipschitz_constant(f, N)
  return f / jnp.maximum(L, 1.0)


def w1_dual_objective(f_raw, p, q, N):
  f = make_lipschitz(f_raw, N)
  return jnp.dot(f, p - q)


def w1_dual(p, q, N, n_steps=1_000, lr=1e-2):
  r""" Lower bound on ``W1(p, q)`` by Adam ascent on the dual potential; needs ``optax``. """
  import optax

  f = jnp.zeros_like(p)
  optimizer = optax.adam(lr)
  opt_state = optimizer.init(f)

  @jax.jit
  def step(f, opt_state):
    # Optax minimizes, so negate the dual objective
    loss, grad = jax.value_and_grad(lambda f: -w1_dual_objective(f, p, q, N))(f)
    updates, opt_state = optimizer.update(grad, opt_state, f)
    f = optax.apply_updates(f, updates)
    return f, opt_state, loss

  for _ in range(n_steps):
    f, opt_state, loss = step(f, opt_state)

  f_star = make_lipschitz(f, N)
  W1 = jnp.dot(f_star, p - q)

  return W1, f_star


def hypercube_lipschitz_constraints(N):
  r""" Sparse ``A, b`` with ``A f <= b`` encoding ``|f(x) - f(x')| <= 1`` on every edge of the hypercube. """
  M = 1 << N
  states = np.arange(M)
  srcs = []
  dsts = []

  for i in range(N):
    # include every undirected edge once
    src = states[(states & (1 << i)) == 0]
    dst = src ^ (1 << i)
    srcs.append(src)
    dsts.append(dst)

  src = np.concatenate(srcs)
  dst = np.concatenate(dsts)

  E = len(src)

  # f_src - f_dst <= 1
  # f_dst - f_src <= 1

  Es = np.arange(E)
  r = np.concatenate([Es, Es, E + Es, E + Es])
  c = np.concatenate([src, dst, src, dst])
  data = np.concatenate([np.ones(E), -np.ones(E), -np.ones(E), np.ones(E)])
  A = coo_matrix((data, (r, c)), shape=(2 * E, M)).tocsr()
  # 1-lipschitz condition
  b = np.ones(2 * E)
  return A, b


def w1_dual_exact(p, q, A_lip, b_lip):
  r""" Exact ``W1(p, q)`` and an optimal potential, by linear programming; the potential is gauged to ``f[0] = 0``. """
  M = len(p)
  # scipy minimizes, whereas we want max <f,p-q>
  c = -(p - q)
  bounds = [(None, None)] * M
  # remove additive gauge freedom
  bounds[0] = (0.0, 0.0)
  result = linprog(c, A_ub=A_lip, b_ub=b_lip, bounds=bounds, method="highs")

  if not result.success:
    raise RuntimeError(result.message)

  return -result.fun, result.x


def get_speed_limit(graph: GraphT, p0: Array, pts: Array, rates_nu: Array, dt: float, N_max: int, N: int, w1_num_points: int = 200):
  r"""Speed limit along a trajectory ``pts`` (``p_0, p_1, ...`` spaced by ``dt``) of an ``N``-spin chain.

  ``rates_nu[nu, i, sigma]`` are the per-bath flip rates of ``graph``. ``W1`` is evaluated at up to
  ``w1_num_points`` log-spaced steps. Returns ``(Ts, speed_limit)`` at those times.
  """
  rows = graph[0]
  n_states = rows.shape[1]
  cols = jnp.tile(jnp.arange(n_states), N).reshape(*rows.shape)
  mask_u = rows < cols
  mask_l = rows > cols

  masked_rates = rates_nu[:, mask_u]
  # this is same as reverse_rates
  reverse_rates = rates_nu[:, mask_l]

  gather_indices_u = cols[mask_u].reshape(-1)[:, None]
  # want the probability at the destination of the forward edge, i.e. rows[mask_u], not cols[mask_l]
  gather_indices_l = cols[mask_l].reshape(-1)[:, None]

  dim_nums = lax.GatherDimensionNumbers(
    offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)
  )

  def mu_A(p):
    Jp = masked_rates * lax.gather(p, gather_indices_u, dim_nums, slice_sizes=(1,))[None, :]
    Jm = reverse_rates * lax.gather(p, gather_indices_l, dim_nums, slice_sizes=(1,))[None, :]
    return jnp.sum(0.5 * (Jp + Jm))

  mu_A_vals = jax.vmap(mu_A)(pts)

  if w1_num_points >= N_max:
    sample_indices = np.arange(N_max)
  else:
    sample_indices = np.unique(
      np.rint(np.geomspace(1, N_max, num=w1_num_points)).astype(int)
    ) - 1
  Ns = jnp.asarray(sample_indices + 1)
  Ts = Ns * dt
  A_lip, b_lip = hypercube_lipschitz_constraints(N)
  w1 = jnp.array([w1_dual_exact(p0, pts[i + 1], A_lip, b_lip)[0] for i in sample_indices])

  dx_reduced = jnp.asarray(1.0)
  # dx is constant along integration axis
  # average activity over [0, T_n]
  mu_A_integrals = jnp.cumsum(mu_A_vals, axis=-1)[1:] - 0.5 * (mu_A_vals[0] + mu_A_vals[1:])
  mu_A_avgs = dx_reduced * mu_A_integrals[sample_indices] / Ns
  v1s = w1 / Ts
  ratios = jnp.clip(v1s / (2 * mu_A_avgs), 0, 1 - 1e-9)
  speed_limit_jax = 2 * w1 * jnp.arctanh(ratios)

  return Ts, speed_limit_jax
