# tests/test_clock_information.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as onp
import pytest

import physics.clock_information as ci
from physics.markov import KL


def _generator(rng, n):
  K = rng.random((n, n)) + 0.1
  onp.fill_diagonal(K, 0.0)
  return jnp.asarray(K - onp.diag(K.sum(axis=0)))  # columns sum to 0: dp/dt = K @ p


def _dist(rng, n):
  p = rng.random(n) + 0.05
  return jnp.asarray(p / p.sum())


def _trapezoid(values, ts):
  return float(jnp.sum(0.5 * (values[1:] + values[:-1]) * jnp.diff(ts)))


# --------------------------------------------------------------------------- #
# Windows and clock information
# --------------------------------------------------------------------------- #

def test_compensation_identity_holds_for_every_reference():
  rng = onp.random.default_rng(0)
  P = jnp.stack([_dist(rng, 5) for _ in range(7)])
  w = jnp.asarray(rng.random(7))
  w = w / w.sum()
  info, q = ci.clock_information(P, w), ci.time_average(P, w)
  for _ in range(5):
    r = _dist(rng, 5)
    assert onp.isclose(jnp.sum(w * KL(P, r, axis=1)), info + KL(q, r))   # sum_i w_i D(p_i||r) = I + D(qbar||r)


def test_clock_information_vanishes_exactly_for_a_constant_trajectory():
  rng = onp.random.default_rng(1)
  p = _dist(rng, 4)
  w = ci.uniform_weights(9)
  assert float(ci.clock_information(jnp.tile(p, (9, 1)), w)) == pytest.approx(0.0, abs=1e-14)
  P = jnp.stack([_dist(rng, 4) for _ in range(9)])
  assert float(ci.clock_information(P, w)) > 0.0
  assert onp.isclose(float(jnp.sum(w * jnp.sum(P * ci.log_ratio(P, w), axis=1))), float(ci.clock_information(P, w)))


# --------------------------------------------------------------------------- #
# The minimal cost: boundary formula, optimality, slope of the clock information
# --------------------------------------------------------------------------- #

def test_boundary_formula_equals_the_integrated_contraction_rate():
  rng = onp.random.default_rng(2)
  K, p0, T = _generator(rng, 4), _dist(rng, 4), 2.0
  ts = jnp.linspace(0.0, T, 2001)
  P = ci.ctmc_trajectory(K, p0, ts)
  q = ci.time_average(P, ci.uniform_weights(len(ts)))
  rates = jax.vmap(lambda p: ci.contraction_rate(K, p, q))(P)
  assert _trapezoid(rates, ts) == pytest.approx(float(ci.uniform_window_mmc(P)), rel=1e-5)


def test_time_average_is_the_optimal_prior_with_the_exact_gap():
  rng = onp.random.default_rng(3)
  K, p0, T = _generator(rng, 4), _dist(rng, 4), 1.5
  ts = jnp.linspace(0.0, T, 2001)
  P = ci.ctmc_trajectory(K, p0, ts)
  q = ci.time_average(P, ci.uniform_weights(len(ts)))
  cost = lambda r: _trapezoid(jax.vmap(lambda p: ci.contraction_rate(K, p, r))(P), ts)
  for _ in range(4):
    r = _dist(rng, 4)
    gap = cost(r) - cost(q)
    assert gap > 0.0
    assert gap == pytest.approx(T * float(ci.contraction_rate(K, q, r)), rel=1e-4)   # MC(r) = MC(qbar) + T D_K(qbar||r)


def test_slope_of_the_clock_information_is_minus_the_minimal_cost_rate():
  rng = onp.random.default_rng(4)
  K, p0, T = _generator(rng, 3), _dist(rng, 3), 2.0
  ts = jnp.linspace(0.0, T, 1601)
  P = ci.ctmc_trajectory(K, p0, ts)
  h = 1e-5
  info = ci.clock_information_decay(K, P, ci.uniform_weights(len(ts)), jnp.array([0.0, h]))
  assert -float(info[1] - info[0]) / h == pytest.approx(float(ci.uniform_window_mmc(P)) / T, rel=1e-3)


def test_long_windows_approach_the_budget_from_below():
  K = ci.ring_generator(5, 1.0, 0.5)
  p0 = jnp.zeros(5).at[0].set(1.0)
  scan = ci.ctmc_window_scan(K, p0, (5.0, 20.0, 80.0), n_t=1601)
  assert scan.budget == pytest.approx(onp.log(5))
  assert onp.all(onp.diff(scan.mmc) > 0) and onp.all(scan.mmc < scan.budget)
  assert scan.mmc[-1] > 0.9 * scan.budget


# --------------------------------------------------------------------------- #
# Spectral filters, near equilibrium
# --------------------------------------------------------------------------- #

def test_mode_filters_have_the_stated_limits_and_derivative():
  x = onp.array([1e-4, 2e-3, 0.5, 1.0, 5.0, 50.0])
  g = onp.asarray(ci.mode_filter_uniform(x))
  assert onp.allclose(g[:2], x[:2] ** 3 / 12, rtol=1e-2)
  assert g[-1] == pytest.approx(0.5 - 1 / 50, rel=1e-6)
  xx = onp.linspace(0.05, 10.0, 400)
  dg = onp.gradient(onp.asarray(ci.mode_filter_uniform(xx)), xx, edge_order=2)
  assert onp.allclose(dg, (onp.exp(-xx) - (1 - onp.exp(-xx)) / xx) ** 2, atol=1e-4)   # g' is a perfect square
  h = onp.asarray(ci.mode_filter_exponential(onp.array([0.0, 1.0, 1e6])))
  assert h[0] == 0.0 and h[1] == pytest.approx(1 / 12) and h[2] == pytest.approx(0.5, rel=1e-5)


def test_near_equilibrium_ou_cost_follows_the_mode_filter():
  xs = jnp.linspace(-7.0, 7.0, 701)
  Ts = onp.array([1.0, 5.0])
  scan = ci.ou_window_scan(Ts, xs=xs, a=1.0, m0=0.3, v0=1.0, n_t=801)
  assert onp.allclose(scan.mmc / scan.budget, 2.0 * onp.asarray(ci.mode_filter_uniform(Ts)), rtol=1e-2)


def test_autodiff_time_score_matches_the_closed_form():
  t, x = 0.4, 1.3
  m, v = 2.0 * onp.exp(-t), 1.0 - 0.75 * onp.exp(-2 * t)
  m_dot, v_dot = -m, 1.5 * onp.exp(-2 * t)
  closed = (x - m) * m_dot / v + (x - m) ** 2 * v_dot / (2 * v ** 2) - v_dot / (2 * v)
  auto = jax.grad(ci.ou_log_density, argnums=0)(t, x, a=1.0, m0=2.0, v0=0.25)
  assert float(auto) == pytest.approx(closed, rel=1e-12)


def test_autodiff_rates_agree_with_the_grid_versions_away_from_the_time_edges():
  ou = dict(a=1.0, m0=2.0, v0=0.25)
  ts, xs = jnp.linspace(0.0, 3.0, 301), jnp.linspace(-7.0, 7.0, 701)
  w = ci.uniform_weights(len(ts))
  P = ci.ou_trajectory(ts, xs, **ou)
  J_grid, J_exact = ci.time_fisher_information(P, ts), ci.ou_time_fisher_exact(ts, xs, **ou)
  rel = onp.abs(onp.asarray(J_grid - J_exact)) / onp.asarray(J_exact)
  assert rel[1:-1].max() < 1e-3 and rel[0] > 1e-2                     # jnp.gradient is first order at the ends
  D_grid = ci.ou_contraction_rate(P, ci.time_average(P, w), xs, a=1.0)
  D_exact = ci.ou_contraction_rate_exact(ts, xs, w, **ou)
  assert onp.allclose(D_grid, D_exact, rtol=1e-3)
  assert float(jnp.sum(w * D_exact)) == pytest.approx(float(ci.uniform_window_mmc(P)) / 3.0, rel=1e-3)


# --------------------------------------------------------------------------- #
# Clock blur
# --------------------------------------------------------------------------- #

def test_heat_smoothing_keeps_the_time_average_and_obeys_de_bruijn():
  T, n, s = 3.0, 400, 0.01
  xs = jnp.linspace(-7.0, 7.0, 701)
  ts = ci.midpoint_times(T, n)
  P = ci.ou_trajectory(ts, xs, a=1.0, m0=2.0, v0=0.25)
  w = jnp.full(n, 1.0 / n)
  Ps = ci.heat_smooth(P, s, T)
  assert float(jnp.max(jnp.abs(Ps.mean(axis=0) - P.mean(axis=0)))) < 1e-12
  info = lambda scale: ci.clock_information(jnp.clip(ci.heat_smooth(P, scale, T), 1e-300, None), w)
  dI = (float(info(1.01 * s)) - float(info(s))) / (0.01 * s)
  assert dI == pytest.approx(-float(jnp.mean(ci.time_fisher_information(Ps, ts))), rel=1e-2)


def test_unblurred_cost_rate_matches_the_boundary_formula():
  xs = jnp.linspace(-7.0, 7.0, 701)
  blur = ci.ou_blur_scan([0.0], T=3.0, n_t=400, xs=xs, display_sigmas=(0.0,))
  window = ci.ou_space_time(T=3.0, n_t=801, xs=xs)
  assert blur.mmc[0] == pytest.approx(window.mmc / 3.0, rel=2e-3)


# --------------------------------------------------------------------------- #
# Rings
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("n, k_plus, k_minus", [(10, 1.0, 0.01), (6, 2.0, 0.5), (12, 1.0, 0.2)])
def test_uniform_ring_sits_on_the_coherence_bound(n, k_plus, k_minus):
  lam = onp.linalg.eigvals(onp.asarray(ci.ring_generator(n, k_plus, k_minus)))
  lam = lam[onp.argsort(-lam.real)][1]
  assert abs(lam.imag / lam.real) == pytest.approx(ci.coherence_bound(n, k_plus, k_minus), rel=1e-8)
  assert onp.allclose(onp.asarray(ci.ring_generator(n, k_plus, k_minus)).sum(axis=0), 0.0)
  assert onp.allclose(ci.stationary_distribution(ci.ring_generator(n, k_plus, k_minus)), 1.0 / n)


def test_ring_comparison_rings_only_without_detailed_balance():
  r = ci.ring_comparison(10, 1.0, 0.01, t_max=20.0, n_t=201, Ts=(2.0, 10.0))
  driven, reversible = r.start_prob
  assert onp.any(onp.diff(driven) > 1e-4) and onp.all(onp.diff(reversible) <= 1e-12)
  assert onp.allclose(r.mode[1].imag, 0.0, atol=1e-12)
  assert r.scans[0].mmc[-1] > r.scans[1].mmc[-1]   # the circulating packet carries more clock information
