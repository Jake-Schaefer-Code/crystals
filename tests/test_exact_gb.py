# tests/test_exact_gb.py
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import src.exact_gb as gb


def P(terms, n):
  return gb.poly_from_terms(terms, n)


def test_a_float_engine_receipt_the_circle_and_the_hyperbola_never_give_the_unit_ideal():
  r"""<x^2 + y^2 - 1, theta x y + y^2 - 1/4> is proper for every rational theta (upstream's receipt)."""
  for theta in [Fraction(k, 29) for k in range(-14, 15)]:
    f = P({(2, 0): 1, (0, 2): 1, (0, 0): -1}, 2)
    g = P({(1, 1): theta, (0, 2): 1, (0, 0): Fraction(-1, 4)}, 2)
    basis = gb.buchberger([f, g], gb.GRLEX)
    assert not gb.is_one_in_ideal(basis, 2)


def test_the_unit_ideal_is_detected():
  basis = gb.buchberger([P({(1,): 1}, 1), P({(0,): 1, (1,): 1}, 1)], gb.GRLEX)
  assert gb.is_one_in_ideal(basis, 1)


def test_a_principal_ideal_with_a_constant_term_is_not_the_unit_ideal():
  basis = gb.buchberger([P({(2,): 1, (0,): -1}, 1)], gb.GRLEX)
  assert not gb.is_one_in_ideal(basis, 1)


@pytest.mark.parametrize("order", [gb.LEX, gb.GRLEX])
def test_the_reduced_basis_is_monic_and_inter_reduced(order):
  f = P({(2, 0): 1, (0, 1): -1}, 2)  # x^2 - y
  g = P({(1, 1): 1, (0, 0): -1}, 2)  # x y - 1
  basis = gb.buchberger([f, g], order)
  for h in basis:
    assert h[gb.lm(h, order)] == 1
  leads = [gb.lm(h, order) for h in basis]
  for i, h in enumerate(basis):
    for j, l in enumerate(leads):
      if i != j:
        assert not any(gb._divides(l, e) for e in h)


def test_membership_requires_and_uses_a_groebner_basis():
  f = P({(2, 0): 1, (0, 1): -1}, 2)
  g = P({(1, 1): 1, (0, 0): -1}, 2)
  basis = gb.buchberger([f, g], gb.GRLEX)
  # y*f + x*g*... : x^3 - 1 = x*(x^2 - y) + (x y - 1)
  member = P({(3, 0): 1, (0, 0): -1}, 2)
  assert gb.ideal_membership(basis, member, gb.GRLEX)
  assert not gb.ideal_membership(basis, P({(1, 0): 1}, 2), gb.GRLEX)


def test_standard_monomials_of_a_zero_dimensional_ideal():
  basis = gb.buchberger([P({(2, 0): 1, (0, 0): -1}, 2), P({(0, 2): 1, (0, 0): -1}, 2)], gb.GRLEX)
  assert sorted(gb.standard_monomials(basis, 2)) == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_a_positive_dimensional_ideal_has_no_finite_quotient_basis():
  basis = gb.buchberger([P({(1, 1): 1}, 2)], gb.GRLEX)
  with pytest.raises(ValueError):
    gb.standard_monomials(basis, 2)


def test_radical_membership_sees_through_a_power():
  gens = [P({(2,): 1}, 1)]  # <x^2>
  assert gb.radical_membership(gens, P({(1,): 1}, 1))
  assert not gb.radical_membership(gens, P({(0,): 1}, 1))


def test_saturation_removes_the_saturating_factor():
  # <x*y> : x^inf = <y>
  sat = gb.saturate([P({(1, 1): 1}, 2)], P({(1, 0): 1}, 2))
  assert sat == [{(0, 1): Fraction(1)}]


def test_saturating_by_a_unit_changes_nothing_and_by_a_nilpotent_gives_the_unit_ideal():
  sat = gb.saturate([P({(2,): 1}, 1)], P({(1,): 1}, 1))
  assert sat == [{(0,): Fraction(1)}]
