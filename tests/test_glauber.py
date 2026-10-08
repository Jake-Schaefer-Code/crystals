# tests/test_glauber.py
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as onp
import pytest
from jax.scipy.linalg import expm

import physics.glauber as gl
import physics.markov as mk

J, H_FIELD = 1.3, 0.2
BETAS = jnp.array([0.7, 1.1])
GAMMAS = jnp.array([1.0, 0.5])


def _popcount(n_spins):
  return onp.array([bin(s).count("1") for s in range(1 << n_spins)])


def _magnetization(n_spins):
  return 2 * _popcount(n_spins) - n_spins


def _gibbs(energy, beta):
  w = onp.exp(-beta * (energy - energy.min()))
  return jnp.asarray(w / w.sum())


def _mean_field_energy(n_spins, J, h):
  M = _magnetization(n_spins)
  return -J * M**2 / (2 * n_spins) - h * M


def _lumping(n_spins):
  r"""``L[k, sigma] = 1`` iff sigma has k up spins; sums a full distribution into sectors."""
  k = _popcount(n_spins)
  return jnp.asarray((k[None, :] == onp.arange(n_spins + 1)[:, None]).astype(float))


@pytest.fixture(scope="module")
def mf():
  N = 3
  graph = gl.make_mean_field_graph(N)
  K, g = gl.build_system_from_graph(graph, J, H_FIELD, BETAS, GAMMAS)
  return N, graph, K, g


# --------------------------------------------------------------------------- #
# Graphs
# --------------------------------------------------------------------------- #

def test_state_space_graph_flips_are_involutions_on_pm_one_spins():
  rows, spins = gl.make_state_space_graph(4)
  rows = onp.asarray(rows).astype(int)
  assert onp.all(onp.abs(spins) == 1.0)
  idx = onp.arange(16)
  for i in range(4):
    assert onp.array_equal(rows[i][rows[i]], idx)          # flipping twice is the identity
    assert onp.all(spins[i][rows[i]] == -spins[i])          # and flips exactly spin i
    assert onp.array_equal(onp.delete(spins, i, 0), onp.delete(spins[:, rows[i]], i, 0))


def test_open_boundary_neighbors_of_a_square_lattice():
  nbrs = gl.compute_site_neighbors(3)
  assert sorted(len(n) for n in nbrs) == [2, 2, 2, 2, 3, 3, 3, 3, 4]
  assert len(nbrs[4]) == 4                                   # center site
  for i, ns in enumerate(nbrs):
    for j in ns:
      assert i in nbrs[j]                                    # adjacency is symmetric


# --------------------------------------------------------------------------- #
# Generators, equilibrium, detailed balance
# --------------------------------------------------------------------------- #

def test_the_generator_is_a_valid_column_generator(mf):
  _, _, K, _ = mf
  K = onp.asarray(K)
  assert onp.allclose(K.sum(axis=0), 0.0, atol=1e-12)
  assert onp.all(K - onp.diag(onp.diag(K)) >= 0.0)


def test_gibbs_is_stationary_for_the_mean_field_graph(mf):
  N, _, K, _ = mf
  # a single bath at inverse temperature beta: rates exp(-beta dE / 2) satisfy detailed balance
  graph = gl.make_mean_field_graph(N)
  K1, _ = gl.build_system_from_graph(graph, J, H_FIELD, jnp.array([0.9]), jnp.array([1.0]))
  pi = _gibbs(_mean_field_energy(N, J, H_FIELD), 0.9)
  assert onp.allclose(K1 @ pi, 0.0, atol=1e-6)


def test_detailed_balance_edge_by_edge_on_a_square_lattice():
  graph = gl.make_interaction_graph((2, 2))
  rows = onp.asarray(graph[0]).astype(int)
  beta = 0.8
  rates = onp.asarray(gl.get_rates(graph, J, 0.0, jnp.array([beta]), jnp.array([1.0])).sum(0))
  _, spins = gl.make_state_space_graph(4)
  nbrs = gl.compute_site_neighbors(2)
  energy = -J * sum(spins[i] * spins[j] for i in range(4) for j in nbrs[i] if j > i)
  pi = onp.asarray(_gibbs(energy, beta))
  for i in range(4):
    assert onp.allclose(rates[i] * pi, (rates[i] * pi)[rows[i]], rtol=1e-5)


def test_matrix_free_application_matches_the_dense_generator(mf):
  _, graph, K, _ = mf
  rng = onp.random.default_rng(0)
  p = jnp.asarray(rng.random(8))
  rates = gl.get_rates(graph, J, H_FIELD, BETAS, GAMMAS).sum(0)
  assert onp.allclose(gl.apply_K_from_rates(p, graph[0], rates), K @ p, atol=1e-12)
  assert onp.allclose(gl.apply_K(p, graph, J, H_FIELD, BETAS, GAMMAS), K @ p, atol=1e-12)


def test_uniformized_step_is_identity_plus_generator_over_lambda(mf):
  _, graph, K, _ = mf
  rng = onp.random.default_rng(1)
  p = jnp.asarray(rng.random(8))
  rates = gl.get_rates(graph, J, H_FIELD, BETAS, GAMMAS).sum(0)
  lam = 1.01 * float(rates.sum(0).max())
  P = onp.eye(8) + onp.asarray(K) / lam
  assert onp.allclose(gl.apply_from_rates(p, graph[0], rates, lam), P @ p, atol=1e-12)
  assert onp.allclose(gl.apply_from_rates(p, graph[0], rates, lam, transpose=True), P.T @ p, atol=1e-12)


def test_glauber_transition_is_one_uniformized_step_at_rate_n(mf):
  N, graph, _, _ = mf
  beta = 0.9
  K1, _ = gl.build_system_from_graph(graph, J, H_FIELD, jnp.array([beta]), jnp.array([1.0]))
  rng = onp.random.default_rng(2)
  p = jnp.asarray(rng.random(8))
  got = gl.apply_glauber_transition(p, graph, beta, H_FIELD, J)
  assert onp.allclose(got, p + (K1 @ p) / N, atol=1e-6)


# --------------------------------------------------------------------------- #
# Propagation
# --------------------------------------------------------------------------- #

def test_propagate_ctmc_matches_the_matrix_exponential(mf):
  _, graph, K, _ = mf
  p0 = jnp.zeros(8).at[0].set(1.0)
  pt, aux = gl.propagate_ctmc(p0, 0.4, graph, J, H_FIELD, BETAS, GAMMAS, n_terms=80)
  assert float(aux["mass"]) < 1e-12
  assert onp.allclose(pt, expm(K * 0.4) @ p0, atol=1e-10)


def test_uniformized_trajectory_matches_expm_and_integrates_heat(mf):
  _, graph, K, _ = mf
  dt, n = 0.1, 6
  rows, rates, _, _, g, lam, w, d = gl.setup_uniformization(graph, J, H_FIELD, BETAS, GAMMAS, dt)
  p0 = jnp.zeros(8).at[7].set(1.0)
  pN, pts, heat_steps, _ = gl.get_uniformized_traj(p0, rows, rates, g, lam, w, d, n)
  G, heat_op = gl.reduced_step_operators(K, g, dt)  # exp([[K,0],[g,0]] dt), reused for any (K, g)
  for k in range(n + 1):
    assert onp.allclose(pts[k], onp.linalg.matrix_power(onp.asarray(G), k) @ p0, atol=1e-9)
  for k in range(n):
    assert onp.allclose(heat_steps[k], heat_op @ pts[k], atol=1e-9)
  assert onp.allclose(pN, pts[-1])


def test_the_two_uniformized_kernels_agree(mf):
  _, graph, _, _ = mf
  rows, rates, _, _, g, lam, w, d = gl.setup_uniformization(graph, J, H_FIELD, BETAS, GAMMAS, 0.1)
  p0 = jnp.zeros(8).at[2].set(1.0)
  a = gl.get_uniformized_traj(p0, rows, rates, g, lam, w, d, 5)
  b = gl.get_uniformized_traj2(p0, rows, rates, g, lam, w, d, 5)
  for x, y in zip(a, b):
    assert onp.allclose(x, y, atol=1e-10)


def test_stationary_distribution_of_a_single_bath_is_gibbs(mf):
  N, graph, _, _ = mf
  beta = 0.9
  rates = gl.get_rates(graph, J, H_FIELD, jnp.array([beta]), jnp.array([1.0])).sum(0)
  lam = 1.01 * float(rates.sum(0).max())
  pi, err, _ = gl.find_stationary(graph[0], rates, lam, tol=1e-13)
  assert float(err) < 1e-12
  assert onp.allclose(pi, _gibbs(_mean_field_energy(N, J, H_FIELD), beta), atol=1e-6)


# --------------------------------------------------------------------------- #
# Exact mean-field reduction
# --------------------------------------------------------------------------- #

def test_the_reduced_system_is_the_exact_lumping_of_the_full_one(mf):
  N, graph, K, g = mf
  Kr, heat_rate, escape, log_deg = gl.build_mean_field_reduced_system(N, J, H_FIELD, BETAS, GAMMAS)
  L = _lumping(N)
  assert onp.allclose(L @ K, Kr @ L, atol=1e-10)             # strong lumpability
  assert onp.allclose(g, heat_rate @ L, atol=1e-10)           # heat depends on the sector only
  assert onp.allclose(-onp.diag(Kr), escape)
  assert onp.allclose(jnp.exp(log_deg), [comb(N, k) for k in range(N + 1)])


def test_micro_entropy_restores_the_binomial_degeneracy(mf):
  N, *_ = mf
  rng = onp.random.default_rng(3)
  P = rng.random(N + 1)
  P = jnp.asarray(P / P.sum())
  _, _, _, log_deg = gl.build_mean_field_reduced_system(N, J, H_FIELD, BETAS, GAMMAS)
  full = P[_popcount(N)] / jnp.asarray([comb(N, k) for k in _popcount(N)])  # uniform inside a sector
  assert float(gl.mean_field_micro_entropy(P, log_deg)) == pytest.approx(float(mk.H(full)), abs=1e-12)


def test_reduced_curves_reproduce_the_full_state_mismatch_cost(mf):
  N, _, K, _ = mf
  dt, n_max = 0.15, 4
  P0 = jnp.zeros(N + 1).at[N].set(1.0)                         # all up, a single microstate
  p0 = jnp.zeros(1 << N).at[(1 << N) - 1].set(1.0)
  pmmc, _ = gl.mean_field_reduced_curves(P0, N, J, H_FIELD, BETAS, GAMMAS, dt, n_max)
  G = expm(K * dt)
  for n in range(1, n_max + 1):
    # the full graph stores its energy changes in float32, the reduced system builds them in float64
    assert float(pmmc[n - 1]) == pytest.approx(float(mk.pmmc2(p0, G, n)), abs=1e-6)


def test_entropy_production_is_nonnegative_and_vanishes_in_equilibrium(mf):
  N, *_ = mf
  P0 = jnp.zeros(N + 1).at[N].set(1.0)
  _, ep = gl.mean_field_reduced_curves(P0, N, J, H_FIELD, BETAS, GAMMAS, 0.15, 8)
  assert onp.all(onp.asarray(ep) >= -1e-10)                  # second law, two baths
  beta = 0.9
  b1, g1 = jnp.array([beta]), jnp.array([1.0])
  _, _, _, log_deg = gl.build_mean_field_reduced_system(N, J, H_FIELD, b1, g1)
  gibbs = _gibbs(_mean_field_energy(N, J, H_FIELD), beta)
  P_eq = jnp.asarray(_lumping(N) @ gibbs)
  pm, ep_eq = gl.mean_field_reduced_curves(P_eq, N, J, H_FIELD, b1, g1, 0.15, 5)
  assert onp.allclose(ep_eq, 0.0, atol=1e-8) and onp.allclose(pm, 0.0, atol=1e-8)


# --------------------------------------------------------------------------- #
# Optimal prior
# --------------------------------------------------------------------------- #

def _ep_data(mf, betas, gammas, dt):
  _, graph, _, _ = mf
  K, g = gl.build_system_from_graph(graph, J, H_FIELD, betas, gammas)
  _, A = gl.precompute_ep(K, g, dt)                           # A[nu] = integral exp(K^T t) g_nu dt
  return K, g, expm(K * dt), betas @ A


def test_one_bath_optimal_prior_is_gibbs_with_zero_cost(mf):
  N, *_ = mf
  b1, g1 = jnp.array([0.9]), jnp.array([1.0])
  _, _, G, c = _ep_data(mf, b1, g1, 0.2)
  q, cost, info = gl.find_optimal_prior(G, c)
  assert info["converged"]
  assert onp.allclose(q, _gibbs(_mean_field_energy(N, J, H_FIELD), 0.9), atol=1e-6)
  assert cost == pytest.approx(0.0, abs=1e-8)


def test_two_bath_optimal_prior_beats_random_priors_and_is_not_free(mf):
  _, _, G, c = _ep_data(mf, BETAS, GAMMAS, 0.2)
  q, cost, info = gl.find_optimal_prior(G, c)
  assert info["converged"] and cost > 1e-6                     # driven: dissipation is unavoidable
  rng = onp.random.default_rng(4)
  for _ in range(30):
    p = jnp.asarray(rng.dirichlet(onp.ones(8)))
    assert float(gl.EP_cost(p, G, c)) >= cost - 1e-9


def test_the_matrix_free_prior_matches_the_dense_one(mf):
  _, graph, K, g = mf
  dt = 0.2
  _, _, G, c_dense = _ep_data(mf, BETAS, GAMMAS, dt)
  rows, rates, _, _, gg, lam, w, d = gl.setup_uniformization(graph, J, H_FIELD, BETAS, GAMMAS, dt)
  c = sum(BETAS[nu] * gl.integrated_GT(gg[nu], rows, rates, lam, d) for nu in range(2))
  assert onp.allclose(c, c_dense, atol=1e-8)
  q_dense, cost_dense, _ = gl.find_optimal_prior(G, c_dense)
  q, cost, _ = gl.find_optimal_prior_rows_rates(rows, rates, c, lam, w)
  assert onp.allclose(q, q_dense, atol=1e-6)
  assert float(cost) == pytest.approx(cost_dense, abs=1e-8)


# --------------------------------------------------------------------------- #
# Sherrington-Kirkpatrick dynamics
# --------------------------------------------------------------------------- #

def _sk(N, Jij, beta, gamma=1.0):
  Theta = onp.zeros(N)
  rows, spins, fields = gl.make_asymmetric_sk_graph(N, Jij, Theta)
  rates = gl.get_sk_rates(spins, fields, beta, gamma)
  K = gl.dense_generator_from_rates(rows, rates)
  return rows, onp.asarray(spins, dtype=float), rates, K


def test_symmetric_couplings_give_gibbs_and_zero_entropy_production():
  N, beta = 4, 0.7
  rng = onp.random.default_rng(5)
  A = rng.normal(size=(N, N))
  Jsym = (A + A.T) / 2
  onp.fill_diagonal(Jsym, 0.0)
  rows, spins, rates, K = _sk(N, Jsym, beta)
  energy = -0.5 * onp.einsum("is,ij,js->s", spins, Jsym, spins)
  pi = _gibbs(energy, beta)
  assert onp.allclose(K @ pi, 0.0, atol=1e-6)
  assert onp.allclose(gl.stationary_distribution(onp.asarray(K)), pi, atol=1e-6)
  assert float(gl.sk_ep_rate(pi, rows, rates)) == pytest.approx(0.0, abs=1e-6)


def test_asymmetric_couplings_dissipate_in_the_steady_state():
  N, beta = 4, 0.7
  rng = onp.random.default_rng(6)
  Jasym = rng.normal(size=(N, N))
  onp.fill_diagonal(Jasym, 0.0)
  rows, _, rates, K = _sk(N, Jasym, beta)
  pi = gl.stationary_distribution(onp.asarray(K))
  assert float(gl.sk_ep_rate(jnp.asarray(pi), rows, rates)) > 1e-4


def test_sk_observables_run_and_report_zero_dissipation_for_symmetric_couplings():
  r"""Regression: ``observables_for_sk`` passed ``(spins, Jij, Theta)`` to ``sk_fields(Jij, spins, Theta)``."""
  N, beta = 4, 0.7
  rng = onp.random.default_rng(7)
  A = rng.normal(size=(N, N))
  Jsym = (A + A.T) / 2
  onp.fill_diagonal(Jsym, 0.0)
  rows, spins, _ = gl.make_asymmetric_sk_graph(N, Jsym, onp.zeros(N))
  abs_m, ep_per_spin = gl.observables_for_sk(
    beta, jnp.asarray(Jsym), onp.asarray(spins, dtype=float), rows, N, jnp.zeros(N), 1.0
  )
  assert 0.0 <= abs_m <= 1.0
  assert float(ep_per_spin) == pytest.approx(0.0, abs=1e-6)
