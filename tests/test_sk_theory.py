# tests/test_sk_theory.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as onp
import pytest

import physics.sk_theory as sk


def test_gauss_hermite_nodes_reproduce_standard_normal_moments():
  assert sk.gaussian_expectation(onp.ones_like) == pytest.approx(1.0)
  assert sk.gaussian_expectation(lambda z: z) == pytest.approx(0.0, abs=1e-12)
  assert sk.gaussian_expectation(lambda z: z**2) == pytest.approx(1.0)
  assert sk.gaussian_expectation(lambda z: z**4) == pytest.approx(3.0)


def test_couplings_have_no_self_interaction_and_the_right_moments():
  rng = onp.random.default_rng(0)
  Z = rng.normal(size=(400, 400))
  Jij = onp.asarray(sk.make_Jij(Z, 0.6, 0.9))
  assert onp.all(onp.diag(Jij) == 0.0)
  off = Jij[~onp.eye(400, dtype=bool)]
  assert off.mean() == pytest.approx(0.6 / 400, abs=5e-3)
  assert off.std() == pytest.approx(0.9 / onp.sqrt(400), rel=0.02)


def test_the_traced_and_host_constructions_agree():
  Z = onp.random.default_rng(1).normal(size=(8, 8))
  assert onp.allclose(sk.Jij_from_disorder(Z, 1.0, 0.5), sk.make_Jij(Z, 1.0, 0.5))


@pytest.mark.parametrize("beta", [1.5, 3.0])
def test_without_disorder_m_is_the_curie_weiss_solution(beta):
  m = sk.solve_m(beta, 0.0)
  assert m == pytest.approx(onp.tanh(beta * m), abs=1e-10)
  assert m > 0.5


def test_the_ordered_solution_satisfies_its_self_consistency_equation():
  beta, DeltaJ = 2.0, 0.3
  m = sk.solve_m(beta, DeltaJ)
  assert m == pytest.approx(sk.gaussian_expectation(lambda z: onp.tanh(beta * (m + DeltaJ * z))), abs=1e-10)


@pytest.mark.parametrize("beta", [1.5, 2.0, 3.0])
def test_the_ordered_solution_disappears_at_the_critical_disorder(beta):
  critical = sk.critical_DeltaJ(beta)
  assert 0.0 < critical < 2.0
  assert sk.solve_m(beta, 0.9 * critical) > 0.1
  assert sk.solve_m(beta, 1.1 * critical) == pytest.approx(0.0, abs=1e-8)


def test_there_is_no_ordered_phase_above_the_ordering_temperature():
  assert onp.isnan(sk.critical_DeltaJ(0.9))
  assert onp.isnan(sk.critical_DeltaJ(1.0))
