# tests/test_speed_limits.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as onp
import pytest

import physics.glauber as gl
import physics.markov as mk
import physics.speed_limits as sl


def _delta(M, i):
  return onp.eye(M)[i]


def _popcount(N):
  return onp.array([bin(s).count("1") for s in range(1 << N)], dtype=float)


# --------------------------------------------------------------------------- #
# Wasserstein-1 on the hypercube
# --------------------------------------------------------------------------- #

def test_lipschitz_constant_of_hamming_weight_is_one():
  N = 4
  assert float(sl.lipschitz_constant(jnp.asarray(_popcount(N)), N)) == pytest.approx(1.0)
  assert float(sl.lipschitz_constant(jnp.asarray(3.0 * _popcount(N)), N)) == pytest.approx(3.0)


def test_make_lipschitz_gauges_to_zero_and_never_exceeds_one():
  N = 3
  f = jnp.asarray(onp.random.default_rng(0).normal(size=1 << N)) * 5.0
  g = sl.make_lipschitz(f, N)
  assert float(g[0]) == 0.0
  assert float(sl.lipschitz_constant(g, N)) <= 1.0 + 1e-12


def test_the_constraint_matrix_has_two_rows_per_edge_direction():
  N = 3
  A, b = sl.hypercube_lipschitz_constraints(N)
  n_edges = N * (1 << (N - 1))
  assert A.shape == (2 * n_edges, 1 << N) and b.shape == (2 * n_edges,)
  f = _popcount(N)
  assert onp.all(A @ f <= b + 1e-12)                                  # Hamming weight is 1-Lipschitz
  assert onp.any(A @ (2.0 * f) > b + 1e-12)                           # twice it is not


@pytest.mark.parametrize("N", [2, 3, 4])
def test_w1_between_antipodes_neighbors_and_uniform(N):
  M = 1 << N
  A, b = sl.hypercube_lipschitz_constraints(N)
  zero, ones, one_flip = _delta(M, 0), _delta(M, M - 1), _delta(M, 1)
  assert sl.w1_dual_exact(zero, ones, A, b)[0] == pytest.approx(N)            # Hamming distance
  assert sl.w1_dual_exact(zero, one_flip, A, b)[0] == pytest.approx(1.0)
  uniform = onp.ones(M) / M
  assert sl.w1_dual_exact(zero, uniform, A, b)[0] == pytest.approx(N / 2)     # mean distance to 0
  assert sl.w1_dual_exact(uniform, uniform, A, b)[0] == pytest.approx(0.0, abs=1e-9)


def test_w1_matches_optimal_transport_with_the_hamming_cost():
  ot = pytest.importorskip("ot")
  N = 3
  M = 1 << N
  rng = onp.random.default_rng(1)
  p, q = rng.dirichlet(onp.ones(M)), rng.dirichlet(onp.ones(M))
  states = onp.arange(M)
  D = onp.array([[bin(a ^ b).count("1") for b in states] for a in states], dtype=float)
  A, b = sl.hypercube_lipschitz_constraints(N)
  assert sl.w1_dual_exact(p, q, A, b)[0] == pytest.approx(float(ot.emd2(p, q, D)), abs=1e-9)


def test_the_gradient_dual_is_a_lower_bound_on_the_exact_value():
  pytest.importorskip("optax")
  N = 3
  M = 1 << N
  rng = onp.random.default_rng(2)
  p, q = rng.dirichlet(onp.ones(M)), rng.dirichlet(onp.ones(M))
  A, b = sl.hypercube_lipschitz_constraints(N)
  exact = sl.w1_dual_exact(p, q, A, b)[0]
  approx, f_star = sl.w1_dual(jnp.asarray(p), jnp.asarray(q), N, n_steps=1500, lr=2e-2)
  assert float(sl.lipschitz_constant(f_star, N)) <= 1.0 + 1e-9
  assert float(approx) <= exact + 1e-9
  assert float(approx) > 0.9 * exact


# --------------------------------------------------------------------------- #
# The speed limit is a bound on dissipation
# --------------------------------------------------------------------------- #

def test_the_speed_limit_never_exceeds_the_entropy_production():
  N, dt, N_max = 3, 0.1, 40
  J, h = 1.0, 0.1
  betas, gammas = jnp.array([2.0, 0.7]), jnp.array([1.0, 1.0])
  graph = gl.make_mean_field_graph(N)
  rows, rates, rates_nu, _, g, lam, w, d = gl.setup_uniformization(graph, J, h, betas, gammas, dt)
  p0 = jnp.zeros(1 << N).at[(1 << N) - 1].set(1.0)
  _, pts, heat_steps, _ = gl.get_uniformized_traj(p0, rows, rates, g, lam, w, d, N_max)

  Ts, limit = sl.get_speed_limit(graph, p0, pts, rates_nu, dt, N_max, N, w1_num_points=12)

  idx = onp.rint(onp.asarray(Ts) / dt).astype(int)                          # cycle count of each sample
  entropy_change = mk.Hv(pts[1:]) - mk.H(p0)
  ep = entropy_change - jnp.cumsum(heat_steps @ betas)                      # Sigma_T = Delta S - sum_nu beta_nu Q_nu
  assert onp.all(onp.asarray(limit) > 0.0)
  assert onp.all(onp.asarray(limit) <= onp.asarray(ep)[idx - 1] + 1e-9)
  assert onp.all(onp.diff(onp.asarray(Ts)) > 0)
