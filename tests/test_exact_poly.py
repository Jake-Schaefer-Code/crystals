# tests/test_exact_poly.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import itertools as it
from fractions import Fraction

import numpy as onp
import pytest

import src.exact_poly as bridge
import src.invariants as inv

def _evaluate(poly, x):
  total = Fraction(0) if all(isinstance(v, (int, Fraction)) for v in x) else 0.0
  for exps, coeff in poly.items():
    term = coeff
    for xi, e in zip(x, exps):
      term = term * xi ** e
    total = total + term
  return total


# --------------------------------------------------------------------------- #
# Exact polynomial plumbing
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("n", [2, 3, 4])
def test_the_expanded_vandermonde_agrees_with_the_numeric_one(n):
  poly = bridge.vandermonde_poly(n)
  x = [Fraction(k * k + 1, 3) for k in range(1, n + 1)]
  expected = 1
  for i, j in it.combinations(range(n), 2):
    expected *= x[j] - x[i]
  assert _evaluate(poly, x) == expected


@pytest.mark.parametrize("n", [2, 3, 4])
def test_the_expanded_vandermonde_matches_src_invariants(n):
  poly = bridge.vandermonde_poly(n)
  x = [0.3 * k - 0.5 for k in range(n)]
  assert onp.isclose(float(_evaluate(poly, x)), float(inv.vandermonde(onp.array(x))))


def test_the_vandermonde_has_the_expected_degree():
  for n in (2, 3, 4):
    degrees = {sum(e) for e in bridge.vandermonde_poly(n)}
    assert degrees == {n * (n - 1) // 2}


# --------------------------------------------------------------------------- #
# Divisibility: the factorization theorem, machine-checked
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("n", [2, 3])
def test_the_vandermonde_divides_itself(n):
  assert bridge.divides_vandermonde(bridge.vandermonde_poly(n), n)


@pytest.mark.parametrize("n", [2, 3])
def test_the_vandermonde_divides_an_antisymmetric_polynomial(n):
  r"""V times any symmetric polynomial is antisymmetric, and must be divisible by V."""
  sym = {tuple([1] * n): Fraction(1), (0,) * n: Fraction(3)}  # e_n + 3
  f = bridge.multiply(bridge.vandermonde_poly(n), sym)
  assert bridge.divides_vandermonde(f, n)


@pytest.mark.parametrize("n", [2, 3])
def test_the_vandermonde_does_not_divide_a_symmetric_polynomial(n):
  sym = {tuple([1] * n): Fraction(1), (0,) * n: Fraction(3)}
  assert not bridge.divides_vandermonde(sym, n)


# --------------------------------------------------------------------------- #
# Saturation: spurious nodes
# --------------------------------------------------------------------------- #

def test_the_vandermonde_itself_has_no_extra_nodes():
  r"""Its node IS the coincidence set, so saturating it away leaves the unit ideal."""
  assert bridge.has_no_extra_nodes(bridge.vandermonde_poly(2), 2)


def test_a_vandermonde_times_a_factor_has_that_factor_as_an_extra_node():
  r"""V * (x_0 + x_1) vanishes on a plane that is not a coincidence; saturation must find it."""
  n = 2
  extra = {(1, 0): Fraction(1), (0, 1): Fraction(1)}
  f = bridge.multiply(bridge.vandermonde_poly(n), extra)
  assert not bridge.has_no_extra_nodes(f, n)
  sat = bridge.extra_node_ideal(f, n)
  # The saturation is the principal ideal of the extra factor, up to a unit.
  x = [Fraction(2), Fraction(-2)]  # a point on x_0 + x_1 = 0, off the diagonal
  assert all(_evaluate(g, x) == 0 for g in sat)


def test_a_squared_vandermonde_still_has_no_extra_nodes():
  r"""Saturation removes the coincidence component to all orders, not just the first."""
  n = 2
  v = bridge.vandermonde_poly(n)
  assert bridge.has_no_extra_nodes(bridge.multiply(v, v), n)


# --------------------------------------------------------------------------- #
# Plucker
# --------------------------------------------------------------------------- #

def test_the_gr_2_4_plucker_ideal_is_one_quadric():
  polys = bridge.plucker_ideal(4, 2)
  assert len(polys) == 1
  gb = bridge.groebner_basis(polys, 6)
  assert len(gb) == 1
  assert {sum(e) for e in gb[0]} == {2}


def test_the_klein_quadric_from_the_bridge_matches_src_grassmann():
  import jax.numpy as jnp

  import src.grassmann as gr

  polys = bridge.plucker_ideal(4, 2)
  assert len(polys) >= 1
  p = [Fraction(k + 1, k + 2) for k in range(6)]
  pj = jnp.array([float(v) for v in p])
  target = float(gr.klein_quadric(pj))
  values = {abs(float(_evaluate(poly, p))) for poly in polys}
  assert any(onp.isclose(v, abs(target)) for v in values)


@pytest.mark.parametrize("n_orb, n_occ, count", [(4, 2, 1), (5, 2, 5), (6, 2, 15), (5, 3, 5)])
def test_the_number_of_plucker_relations_is_the_classical_binomial(n_orb, n_occ, count):
  r"""For Gr(2,n) the minimal relations are indexed by 4-subsets, so there are binom(n, 4)."""
  polys = bridge.plucker_ideal(n_orb, n_occ)
  assert len(polys) == count
  assert all({sum(e) for e in poly} == {2} for poly in polys)


def test_the_gr_2_5_groebner_basis_is_homogeneous_and_at_least_minimal():
  r"""The Grassmannian is projective, so its ideal is homogeneous and every GB element is too.

  A reduced Groebner basis is generally larger and of higher degree than a minimal generating
  set: here the five quadrics pick up one cubic S-polynomial remainder.
  """
  polys = bridge.plucker_ideal(5, 2)
  gb = bridge.groebner_basis(polys, 10)
  assert len(gb) >= len(polys)
  assert all(len({sum(e) for e in g}) == 1 for g in gb)
  assert sum(1 for g in gb if {sum(e) for e in g} == {2}) == len(polys)
