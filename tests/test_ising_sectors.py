# tests/test_ising_sectors.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from math import comb

import jax

jax.config.update("jax_enable_x64", True)

import numpy as onp
import pytest
from scipy.linalg import expm
from scipy.special import logsumexp

import physics.markov as markov
import physics.ising_sectors as sec


@pytest.fixture(scope="module")
def one_bath():
  model = sec.build_sector_model(7, 0.6, 0.0, [1.1], [1.0])
  return model, sec.make_step_map(model, 0.2)


# --------------------------------------------------------------------------- #
# The reduced model
# --------------------------------------------------------------------------- #

def test_reduction_agrees_with_the_full_microstate_dynamics():
  errors = sec.validate_sector_reduction()
  assert max(errors.values()) < 2e-5


def test_the_generator_is_a_column_generator_with_binomial_degeneracy(one_bath):
  model, _ = one_bath
  assert onp.allclose(model.generator.sum(axis=0), 0.0, atol=1e-12)
  assert onp.allclose(onp.exp(model.log_degeneracy), [comb(7, k) for k in range(8)])
  assert logsumexp(model.log_degeneracy) == pytest.approx(7 * onp.log(2.0))


def test_one_bath_stationary_state_is_gibbs_and_dissipation_free(one_bath):
  model, _ = one_bath
  pi = sec.stationary_distribution(model.generator, continuous=True)
  assert onp.allclose(pi, sec.gibbs_sector_distribution(model, 1.1), atol=1e-12)
  assert float(model.heat_rate[0] @ pi) == pytest.approx(0.0, abs=1e-12)


def test_the_step_map_is_stochastic_and_is_the_exponential_of_the_generator(one_bath):
  model, step = one_bath
  assert onp.allclose(step.G.sum(axis=0), 1.0, atol=1e-12) and step.G.min() > -1e-12
  assert onp.allclose(step.G, expm(model.generator * 0.2), atol=1e-12)


def test_microstate_entropy_adds_the_degeneracy_of_each_sector(one_bath):
  model, _ = one_bath
  rng = onp.random.default_rng(0)
  P = rng.random(8)
  P /= P.sum()
  full = onp.array([P[k] / comb(7, k) for k in range(8) for _ in range(comb(7, k))])
  assert float(sec.microstate_entropy(P, model.log_degeneracy)) == pytest.approx(float(markov.H(full)))


def test_invalid_inputs_are_rejected(one_bath):
  model, step = one_bath
  with pytest.raises(ValueError):
    sec.build_sector_model(5, 1.0, 0.0, [1.0, 2.0], [1.0])
  with pytest.raises(ValueError):
    sec.build_sector_model(5, 1.0, 0.0, [1.0], [0.0])
  with pytest.raises(ValueError):
    sec.make_step_map(model, -1.0)
  with pytest.raises(ValueError):
    sec.repeat_map(step, onp.ones(8), 3, model.log_degeneracy, [1.1])      # not normalized
  with pytest.raises(ValueError):
    sec.repeat_map(step, sec.all_up(7), 0, model.log_degeneracy, [1.1])


# --------------------------------------------------------------------------- #
# Repeated maps
# --------------------------------------------------------------------------- #

def test_repeat_map_pop_matches_the_markov_bound_without_degeneracy(one_bath):
  _, step = one_bath
  p0 = sec.all_up(7)
  m = sec.repeat_map(step, p0, 6, onp.zeros(8), [1.1])
  assert m.pmmc[-1] == pytest.approx(float(markov.bound2(p0, step.G, 6)[0]), abs=1e-10)


def test_one_bath_entropy_production_is_the_drop_in_distance_to_gibbs(one_bath):
  model, step = one_bath
  m = sec.repeat_map(step, sec.all_up(7), 25, model.log_degeneracy, [1.1])
  assert onp.allclose(m.total_ep, m.stationary_kl_drop, atol=1e-9)
  assert onp.max(onp.abs(m.ep_remainder)) < 1e-9
  assert onp.all(m.pmmc <= m.total_ep + 1e-9) and onp.all(m.pmmc >= -1e-9)    # POP is a lower bound


# --------------------------------------------------------------------------- #
# Analyses
# --------------------------------------------------------------------------- #

def test_hidden_housekeeping_is_nonnegative_and_settles_to_the_steady_rate():
  r = sec.hidden_housekeeping(n_spins=7, n_cycles=150, contrasts=[0.0, 0.1, 0.2, 0.4, 1.0])
  assert onp.min(r.hidden_ep) > -1e-9
  assert r.steady_ep_rate > 0.0 and r.stationary_error < 1e-11
  # relaxation transients die out, after which the hidden channel dissipates at the steady rate:
  # hidden_ep = rate * t + const, with the offset converging from below
  offset = r.hidden_ep - r.steady_ep_rate * r.times
  assert onp.all(onp.diff(offset) >= -1e-12)
  assert abs(offset[-1] - offset[-2]) < 1e-10
  late = slice(-50, None)
  slope = onp.polyfit(r.times[late], r.hidden_ep[late], 1)[0]
  assert slope == pytest.approx(r.steady_ep_rate, rel=1e-8)
  assert r.ep_rates[0] == pytest.approx(0.0, abs=1e-12)                       # no contrast, no current
  assert r.quadratic_coefficient > 0.0
  assert r.crossover_time == pytest.approx(r.information_budget / r.steady_ep_rate)
  assert set(r.summary()) >= {"steady_ep_rate", "crossover_time", "quadratic_coefficient"}


def test_symmetry_bit_costs_exactly_ln2_and_is_bounded_by_it():
  run = sec.symmetry_bit_relaxation(9, n_cycles=600)
  assert run.initial_kl == pytest.approx(onp.log(2.0), abs=1e-12)
  assert onp.all(run.metrics.pmmc <= onp.log(2.0) + 1e-9)
  assert run.positive_mass[0] == pytest.approx(1.0) and run.positive_mass[-1] < 0.6
  assert 0.0 < run.t50 < run.t90
  with pytest.raises(AssertionError):
    sec.symmetry_bit_relaxation(8)                                          # even n has no zero-magnetization-free split


def test_critical_scan_orders_and_slows_with_coupling():
  r"""Finite-size Curie-Weiss: relaxation lengthens through ``J_c`` and keeps growing as the
  magnetization sectors decouple (tunnelling between ``+m`` and ``-m`` is exponentially slow)."""
  scan = sec.critical_scan(
    n_spins=21, J_values=onp.linspace(0.25, 0.75, 9), max_time=20.0, horizons=[0.5, 2.0, 5.0, 20.0]
  )
  assert scan.critical_J == pytest.approx(0.5)
  assert onp.all(onp.diff(scan.relaxation_times) > 0.0)
  assert scan.relaxation_times[-1] > 10 * scan.relaxation_times[0]
  assert scan.abs_magnetization[-1] > 2 * scan.abs_magnetization[0]
  assert onp.all(scan.fractions > -1e-9) and onp.all(scan.fractions < 1.0 + 1e-6)
  assert onp.all(onp.diff(scan.fractions, axis=0) >= -1e-9)                  # saturation grows with the horizon


def test_periodic_drive_obeys_the_first_law_and_the_steady_identity():
  scan = sec.periodic_drive_scan(n_spins=11, periods=onp.logspace(-1, 1, 5), n_observation_cycles=20)
  assert scan.first_law_errors.max() < 1e-9 and scan.steady_identity_errors.max() < 1e-9
  assert onp.all(scan.steady_work > 0.0)                                      # hysteresis costs work
  assert onp.all(scan.final_fraction <= 1.0 + 1e-6)
  assert len(scan.loops) == 3
  for field, magnetization in scan.loops.values():
    assert field.shape == magnetization.shape and set(onp.unique(field)) == {-0.25, 0.25}
