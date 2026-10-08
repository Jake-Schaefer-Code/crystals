# tests/test_markov.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as onp
import pytest

import physics.markov as mk


def _stochastic(rng, n):
  G = rng.random((n, n)) + 0.05
  return jnp.asarray(G / G.sum(axis=0, keepdims=True))  # columns sum to 1: p -> G @ p


def _dist(rng, n):
  p = rng.random(n) + 0.05
  return jnp.asarray(p / p.sum())


# --------------------------------------------------------------------------- #
# Entropy and relative entropy
# --------------------------------------------------------------------------- #

def test_entropy_of_uniform_and_point_masses():
  assert onp.isclose(mk.H(jnp.ones(8) / 8), onp.log(8))
  assert float(mk.H(jnp.array([0.0, 1.0, 0.0]))) == 0.0


def test_zero_log_zero_is_zero_not_nan():
  p = jnp.array([0.5, 0.5, 0.0])
  assert onp.isfinite(mk.H(p)) and onp.isfinite(mk.KL(p, jnp.array([0.25, 0.5, 0.25])))


def test_relative_entropy_is_nonnegative_and_zero_only_on_the_diagonal():
  rng = onp.random.default_rng(0)
  p, q = _dist(rng, 6), _dist(rng, 6)
  assert float(mk.KL(p, p)) == pytest.approx(0.0, abs=1e-14)
  assert float(mk.KL(p, q)) > 0.0


def test_relative_entropy_to_uniform_is_log_n_minus_entropy():
  rng = onp.random.default_rng(1)
  p = _dist(rng, 7)
  assert float(mk.KL(p, jnp.ones(7) / 7)) == pytest.approx(onp.log(7) - float(mk.H(p)), abs=1e-12)


def test_vectorized_forms_act_along_axis_one():
  rng = onp.random.default_rng(2)
  P = jnp.stack([_dist(rng, 5) for _ in range(3)])
  Q = jnp.stack([_dist(rng, 5) for _ in range(3)])
  assert onp.allclose(mk.Hv(P), [mk.H(p) for p in P])
  assert onp.allclose(mk.KLv(P, Q), [mk.KL(p, q) for p, q in zip(P, Q)])


def test_the_two_kl_implementations_agree():
  rng = onp.random.default_rng(3)
  p, q = _dist(rng, 6), _dist(rng, 6)
  assert float(mk.KL2(p, q)) == pytest.approx(float(mk.KL(p, q)), abs=1e-12)


def test_js_is_n_times_the_jensen_shannon_divergence():
  # JS returns N * (H(mean) - mean(H)); float32 internally.
  rng = onp.random.default_rng(4)
  P = onp.stack([onp.asarray(_dist(rng, 5)) for _ in range(3)])
  want = 3 * (float(mk.H(jnp.asarray(P.mean(0)))) - float(onp.mean([mk.H(jnp.asarray(p)) for p in P])))
  assert float(mk.JS(P)) == pytest.approx(want, abs=1e-5)
  assert float(mk.JS(onp.stack([P[0], P[0]]))) == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Orbits
# --------------------------------------------------------------------------- #

def test_matrix_powers_stack_identity_through_g_to_the_n_minus_one():
  rng = onp.random.default_rng(5)
  G = _stochastic(rng, 4)
  Gs = mk.matrix_powers(G, 5)
  assert Gs.shape == (5, 4, 4)
  for k in range(5):
    assert onp.allclose(Gs[k], onp.linalg.matrix_power(onp.asarray(G), k))


def test_get_traj_returns_the_orbit_and_its_endpoint():
  rng = onp.random.default_rng(6)
  G, p0 = _stochastic(rng, 4), _dist(rng, 4)
  pN, pts = mk.get_traj(G, p0, 6)
  assert pts.shape == (7, 4)
  for k in range(7):
    assert onp.allclose(pts[k], onp.linalg.matrix_power(onp.asarray(G), k) @ onp.asarray(p0))
  assert onp.allclose(pN, pts[-1])


# --------------------------------------------------------------------------- #
# Mismatch cost
# --------------------------------------------------------------------------- #

def test_mmc_is_nonnegative_by_data_processing():
  rng = onp.random.default_rng(7)
  G, p0, q = _stochastic(rng, 5), _dist(rng, 5), _dist(rng, 5)
  assert float(mk.mmc(q, G, p0, 6)) >= -1e-12


def test_mmc_at_a_stationary_prior_telescopes():
  rng = onp.random.default_rng(8)
  G, p0 = _stochastic(rng, 4), _dist(rng, 4)
  vals, vecs = onp.linalg.eig(onp.asarray(G))
  pi = jnp.asarray(onp.real(vecs[:, onp.argmax(onp.real(vals))]))
  pi = pi / pi.sum()
  n = 5
  pN = onp.linalg.matrix_power(onp.asarray(G), n) @ onp.asarray(p0)
  assert float(mk.mmc(pi, G, p0, n)) == pytest.approx(float(mk.KL(p0, pi) - mk.KL(jnp.asarray(pN), pi)), abs=1e-10)


def test_the_time_averaged_orbit_is_the_optimal_periodic_prior():
  r"""``pmmc`` is the mismatch cost of ``q = mean(p_0..p_{N-1})``, and no other prior does better."""
  rng = onp.random.default_rng(9)
  G, p0, N = _stochastic(rng, 5), _dist(rng, 5), 6
  q_star = mk.pop(p0, G, N)
  best = float(mk.mmc(q_star, G, p0, N))
  assert best == pytest.approx(float(mk.pmmc(p0, G, N)), abs=1e-10)
  for _ in range(20):
    assert float(mk.mmc(_dist(rng, 5), G, p0, N)) >= best - 1e-12


def test_every_route_to_the_optimal_cost_agrees():
  rng = onp.random.default_rng(10)
  G, p0, N = _stochastic(rng, 5), _dist(rng, 5), 4
  want = float(mk.pmmc2(p0, G, N))
  assert float(mk.pmmc(p0, G, N)) == pytest.approx(want, abs=1e-10)
  assert float(mk.bound(p0, G, N)[0]) == pytest.approx(want, abs=1e-10)
  for scan_G in (False, True):
    assert float(mk.bound2(p0, G, N, scan_G=scan_G)[0]) == pytest.approx(want, abs=1e-10)
  assert onp.allclose(mk.get_q_star(p0, G, N), mk.pop(p0, G, N))


def test_pmmc_runs_in_the_ambient_precision_of_g():
  r"""Regression: pmmc forced float32 for the carry and failed against a float64 ``G`` under x64."""
  rng = onp.random.default_rng(11)
  G, p0 = _stochastic(rng, 4), _dist(rng, 4)
  assert G.dtype == jnp.float64
  out = mk.pmmc(p0, G, 3)
  assert out.dtype == jnp.float64
  assert float(mk.pmmc(p0, G.astype(jnp.float32), 3)) == pytest.approx(float(out), abs=1e-5)


def test_generalized_landauer_gap_is_the_entropy_decrease():
  rng = onp.random.default_rng(12)
  G, p0, N = _stochastic(rng, 4), _dist(rng, 4), 3
  _, gen_landauer = mk.bound(p0, G, N)
  pN = onp.linalg.matrix_power(onp.asarray(G), N) @ onp.asarray(p0)
  assert float(gen_landauer) == pytest.approx(float(mk.H(p0) - mk.H(jnp.asarray(pN))), abs=1e-10)


def test_periodic_mmc_curves_match_the_closed_forms_and_the_stationary_state():
  rng = onp.random.default_rng(13)
  G, p0 = _stochastic(rng, 5), _dist(rng, 5)
  pts, pmmcs, pi, kl = mk.periodic_mmc_curves(G, p0, 8)
  assert pts.shape == (9, 5) and pmmcs.shape == (8,)
  for n in range(1, 9):
    assert float(pmmcs[n - 1]) == pytest.approx(float(mk.pmmc2(p0, G, n)), abs=1e-10)
  assert onp.allclose(G @ pi, pi, atol=1e-10) and float(pi.sum()) == pytest.approx(1.0)
  assert float(kl) == pytest.approx(float(mk.KL(p0, pi)))
  assert float(kl) >= 0.0 and onp.all(onp.diff(onp.asarray(pmmcs)) > -1e-12)    # cost accumulates
