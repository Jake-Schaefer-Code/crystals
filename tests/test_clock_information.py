# tests/test_clock_information.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as onp
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq

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


def test_time_harmonics_rebuild_the_trajectory_and_follow_the_heat_flow():
  T, n, s = 3.0, 120, 0.2
  ts = ci.midpoint_times(T, n)
  P = ci.ou_trajectory(ts, jnp.linspace(-7.0, 7.0, 141), a=1.0, m0=2.0, v0=0.25)
  modes = jnp.arange(1, n)
  A = jnp.stack([ci.time_harmonic(P, k) for k in modes])                  # (n - 1, n_x)
  cos = jnp.cos(jnp.pi * modes[None, :] * ts[:, None] / T)                # (n, n - 1)
  q = P.mean(axis=0)
  assert float(jnp.max(jnp.abs(q + cos @ A - P))) < 1e-13                 # the cosine modes are complete
  decay = jnp.exp(-(modes * jnp.pi / T) ** 2 * s)                         # the explicit sum against the DCT filter
  assert float(jnp.max(jnp.abs(q + cos @ (decay[:, None] * A) - ci.heat_smooth(P, s, T)))) < 1e-13


def test_heavy_blur_nodal_set_is_the_predicted_cross_and_hyperbola():
  T, xs = 3.0, jnp.linspace(-7.0, 7.0, 1401)
  blur = ci.ou_blur_scan([0.0], T=T, n_t=400, xs=xs, display_sigmas=(2.5, 3.0))
  cross = blur.cross

  def harmonic(k, x):                                                     # (2/T) int p_t(x) cos(k pi t / T) dt, closed-form OU
    def integrand(t):
      m, v = 2.0 * onp.exp(-t), 1.0 - 0.75 * onp.exp(-2 * t)
      return onp.exp(-(x - m) ** 2 / (2 * v)) / onp.sqrt(2 * onp.pi * v) * onp.cos(k * onp.pi * t / T)
    return 2.0 / T * quad(integrand, 0.0, T, epsabs=1e-13, epsrel=1e-11)[0]

  x_star = brentq(lambda x: harmonic(1, x), 0.0, 1.5, xtol=1e-12)
  h = 1e-5
  slope = (harmonic(1, x_star + h) - harmonic(1, x_star - h)) / (2 * h)
  c = -harmonic(2, x_star) * T / (onp.pi * slope)
  assert cross.x_star.shape == (1,) and cross.x_star[0] == pytest.approx(x_star, abs=1e-4)
  assert cross.c[0] == pytest.approx(c, rel=1e-3)

  xn, rows = onp.asarray(xs), range(len(blur.ts) // 2 - 3, len(blur.ts) // 2 + 3)
  for sigma, Ps in zip(blur.display_sigmas, blur.display_P):
    field = Ps - Ps.mean(axis=0)                                          # zero set of p^sigma_t - qbar
    for i in rows:
      k = onp.flatnonzero(onp.sign(field[i, :-1]) * onp.sign(field[i, 1:]) < 0)
      roots = xn[k] - field[i, k] * (xn[k + 1] - xn[k]) / (field[i, k + 1] - field[i, k])
      x_nodal = roots[onp.argmin(abs(roots - cross.x_star[0]))]
      assert (blur.ts[i] - T / 2) * (x_nodal - cross.x_star[0]) == pytest.approx(cross.product(sigma)[0], rel=2e-2)


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


# --------------------------------------------------------------------------- #
# The dynamical Hamiltonian
# --------------------------------------------------------------------------- #

def _reversible_generator(rng, n):
  pi = onp.asarray(_dist(rng, n))
  S = rng.random((n, n)) + 0.1
  S = S + S.T
  onp.fill_diagonal(S, 0.0)
  K = S / pi[None, :]                                                      # K[y, x] pi[x] = S[y, x] is symmetric
  K = K - onp.diag(K.sum(axis=0))
  return jnp.asarray(3.0 * K / onp.max(onp.abs(onp.linalg.eigvals(K)))), pi   # relaxation rates up to 3: towers interleave


@pytest.fixture(scope="module")
def rings():
  return ci.ring_hamiltonian(10, 1.0, 0.01, T=8.0, n_t=161)


def test_dynamical_hamiltonian_is_the_least_squares_form():
  rng = onp.random.default_rng(5)
  n, n_t, T = 4, 7, 1.5
  K = onp.asarray(_generator(rng, n))
  pi = ci.stationary_distribution(K)
  H, scale = ci.dynamical_hamiltonian(K, T, n_t)
  phi = rng.standard_normal(n_t * n)
  p = phi.reshape(n_t, n) * onp.asarray(scale)
  h = T / (n_t - 1)
  residuals = [(p[i + 1] - p[i]) / h - K @ (p[i] + p[i + 1]) / 2 for i in range(n_t - 1)]
  w = onp.full(n_t, h)
  w[[0, -1]] = h / 2
  assert phi @ onp.asarray(H) @ phi == pytest.approx(h * sum(r @ (r / pi) for r in residuals), rel=1e-12)
  assert phi @ phi == pytest.approx(sum(wi * (pi_t @ (pi_t / pi)) for wi, pi_t in zip(w, p)), rel=1e-12)


def test_zero_level_of_the_dynamical_hamiltonian_is_the_trajectories():
  rng = onp.random.default_rng(6)
  n, n_t, T = 4, 161, 2.0
  K = _generator(rng, n)
  H, scale = map(onp.asarray, ci.dynamical_hamiltonian(K, T, n_t))
  w, V = onp.linalg.eigh(H)
  assert onp.all(onp.abs(w[:n]) < 1e-10) and w[n] > 1e-2                  # an n-fold zero level, then a gap
  h = T / (n_t - 1)
  step = onp.linalg.solve(onp.eye(n) - h * onp.asarray(K) / 2, onp.eye(n) + h * onp.asarray(K) / 2)
  p = [onp.asarray(_dist(rng, n))]
  for _ in range(n_t - 1):
    p.append(step @ p[-1])                                                # the discrete master equation
  assert onp.max(onp.abs(H @ (onp.stack(p) / scale).ravel())) < 1e-10
  exact = (onp.asarray(ci.ctmc_trajectory(K, _dist(rng, n), jnp.linspace(0.0, T, n_t))) / scale).ravel()
  assert exact @ H @ exact / (exact @ exact) < 1e-5 * w[n]                # an exact trajectory, up to O(h^2)
  assert onp.sum((V[:, :n].T @ exact) ** 2) / (exact @ exact) > 1 - 1e-5


def test_reversible_generator_separates_into_the_normal_levels():
  rng = onp.random.default_rng(7)
  n, n_t, T = 5, 161, 3.0
  K, pi = _reversible_generator(rng, n)
  H, scale = map(onp.asarray, ci.dynamical_hamiltonian(K, T, n_t, pi))
  w, V = onp.linalg.eigh(H)
  assert onp.allclose(w[:15], ci.normal_levels(K, T, 15), rtol=1e-3, atol=1e-9)   # 0 (n-fold), then (k pi / T)^2 + lambda_j^2
  for i in range(n, 2 * n):                                                       # the k = 1 tower is a product of t and x
    sv = onp.linalg.svd((V[:, i:i + 1] * scale.reshape(-1, 1)).reshape(n_t, n), compute_uv=False)
    assert sv[1] / sv[0] < 1e-8


def test_uniform_driven_ring_has_the_reversible_levels_and_traveling_nodes(rings):
  rev, drv = rings.systems[:2]
  h = 8.0 / 160
  assert onp.allclose(rev.energies, drv.energies, rtol=3e-3, atol=1e-9)           # the drive is a gauge phase on a normal K
  assert onp.allclose(drv.energies, drv.normal, rtol=3e-3, atol=1e-9)             # and only Re mu enters the formula

  def level_gap(n_t):                                                             # driven against reversible, O(h^2)
    E = [onp.linalg.eigvalsh(onp.asarray(ci.dynamical_hamiltonian(ci.ring_generator(10, kp, km), 8.0, n_t)[0]))[:30]
         for kp, km in ((1.0, 0.01), (0.505, 0.505))]
    return onp.max(onp.abs(E[0] - E[1]))
  assert 3.0 < level_gap(81) / level_gap(161) < 5.0

  A = onp.fft.fft(drv.state, axis=1)[:, 1]                                        # wavenumber 1 of the first state
  strong = (onp.abs(A[1:]) > 0.3 * onp.abs(A).max()) & (onp.abs(A[:-1]) > 0.3 * onp.abs(A).max())
  omega = onp.median(onp.angle(A[1:] * onp.conj(A[:-1]))[strong]) / h
  assert omega == pytest.approx(-(1.0 - 0.01) * onp.sin(2 * onp.pi / 10), rel=1e-2)   # Im of the slowest eigenvalue of K
  assert rev.rank_residual < 1e-9 and drv.rank_residual > 0.3                     # product against a rotating wave
  assert rev.gap < 1e-9 and drv.gap < 1e-6                                        # nodal lines cross in both
  assert all(s.trajectory < 1e-4 * s.energies[10] for s in rings.systems)         # the trajectory is a zero mode

  g = rev.state[:, int(onp.argmax(onp.abs(rev.state).max(axis=0)))]               # time profile at the peak site
  i = int(onp.flatnonzero(g[:-1] * g[1:] < 0)[0])
  t_star = rings.ts[i] - g[i] * h / (g[i + 1] - g[i])
  gamma = 2 * 0.505 * (1 - onp.cos(2 * onp.pi / 10))
  assert t_star == pytest.approx(8.0 / onp.pi * onp.arctan(onp.pi / (gamma * 8.0)), rel=5e-3)   # free ends move the clock node off T/2


def test_driven_ring_in_a_potential_leaves_the_formula_and_avoids_the_crossing(rings):
  rev, drv = rings.systems[2:]
  off = lambda s: onp.max(onp.abs(s.energies[10:] - s.normal[10:]) / s.normal[10:])
  assert off(rev) < 3e-3 and rev.rank_residual < 1e-9 and rev.gap < 1e-9          # reversible: separable, crossing exact
  assert off(drv) > 0.05 and drv.rank_residual > 0.3 and drv.gap > 1e-3           # driven, not normal: levels move, crossing avoided


def test_crossing_gap_of_a_field_with_a_known_saddle():
  ts, eps = onp.linspace(0.0, 4.0, 81), 0.05
  x = onp.arange(10)
  product = (ts[:, None] - 2.013) * onp.cos(2 * onp.pi * (x[None, :] - 0.3) / 10)   # nodal lines cross, off the grid
  assert ci.crossing_gap(product, ts) < 1e-6
  opened = product + eps                                                          # saddles of value eps at (2.013, 2.8), (2.013, 7.8)
  assert ci.crossing_gap(opened, ts) == pytest.approx(eps / onp.abs(ci.ring_interpolate(opened)).max(), rel=1e-4)
  bowl = (ts[:, None] - 2.0) ** 2 + 1 - onp.cos(2 * onp.pi * x[None, :] / 10)     # a minimum of value 0 is not a saddle
  assert ci.crossing_gap(bowl, ts) == pytest.approx(2.0 / 6.0, rel=1e-4)          # the saddle (2, 5) has value 2, the maximum is 6
  assert onp.isnan(ci.crossing_gap(ts[:, None] + onp.cos(2 * onp.pi * x[None, :] / 10), ts))   # no critical point at all


def test_potential_ring_has_boltzmann_detailed_balance_when_reversible():
  n, depth = 8, 1.7
  K = onp.asarray(ci.potential_ring_generator(n, 0.7, 0.7, depth))
  pi = onp.exp(-depth * onp.cos(2 * onp.pi * onp.arange(n) / n))
  pi /= pi.sum()
  assert onp.allclose(K.sum(axis=0), 0.0) and onp.allclose(K @ pi, 0.0)
  assert onp.allclose(K * pi[None, :], (K * pi[None, :]).T)                       # K[y, x] pi[x] symmetric
  assert onp.allclose(ci.potential_ring_generator(n, 1.0, 0.01, 0.0), ci.ring_generator(n, 1.0, 0.01))
