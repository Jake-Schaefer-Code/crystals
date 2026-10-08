# tests/test_spectral_curves.py
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as onp
import pytest
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physics import spectral_curves as sc


# 1. semicircle ---------------------------------------------------------------

def test_semicircle_branch_is_the_stieltjes_transform():
  rng = onp.random.default_rng(0)
  z = rng.uniform(-4, 4, 200) + 1j * rng.uniform(-3, 3, 200)
  s1, s2 = sc.semicircle_s(z, 1), sc.semicircle_s(z, 2)
  assert onp.allclose(s1**2 + z * s1 + 1, 0) and onp.allclose(s1 * s2, 1)
  assert onp.all(onp.abs(s1) < 1)
  assert onp.allclose(sc.joukowski(s1), z)
  big = 1e4 * onp.exp(1j * onp.linspace(0.1, 3.0, 7))
  assert onp.allclose(sc.semicircle_s(big) * big, -1, atol=1e-6)
  x = onp.linspace(-1.99, 1.99, 101)
  jump = sc.semicircle_s(x + 1e-14j) - sc.semicircle_s(x - 1e-14j)
  assert onp.allclose(jump, 2j * onp.pi * sc.semicircle_density(x))


def test_critical_points_interlace_and_are_zeros_of_p_prime():
  eigs = onp.array([-1.7, -1.05, -0.35, 0.3, 1.0, 1.75])
  cp = sc.critical_points(eigs)
  assert onp.all((eigs[:-1] < cp) & (cp < eigs[1:]))
  assert onp.allclose(onp.polyval(onp.polyder(onp.poly(eigs)), cp), 0, atol=1e-10)
  z = onp.array([0.3 + 0.4j, -1 + 2j])
  expected = -onp.polyval(onp.polyder(onp.poly(eigs)), z) / onp.polyval(onp.poly(eigs), z) / len(eigs)
  assert onp.allclose(sc.stieltjes(eigs, z), expected)


# 2. edges --------------------------------------------------------------------

def test_tracy_widom_moments():
  s = onp.linspace(-7, 5, 161)
  for beta, mean, var in ((2, -1.7710868, 0.8131948), (1, -1.2065336, 1.6077810)):
    F, f = sc.tracy_widom(s, beta, m=40)
    mu = onp.trapezoid(s * f, s)
    assert abs(F[-1] - 1) < 1e-4 and abs(mu - mean) < 2e-3
    assert abs(onp.trapezoid((s - mu) ** 2 * f, s) - var) < 5e-3


def test_beta_hermite_spectrum_fills_minus_two_two():
  top = sc.beta_hermite_top(300, 2, 200, k=300, rng=0)
  assert abs(onp.mean(top**2) - 1) < 0.02 and abs(onp.max(top) - 2) < 0.1


# 3. matrix models --------------------------------------------------------------

@pytest.mark.parametrize("a", [1.0, -1.0, -1.9])
def test_symmetric_one_cut_closed_form(a):
  curve = sc.spectral_curve(sc.QuarticModel(1.0, a))
  b = onp.sqrt(2 * (-a + onp.sqrt(a * a + 12)) / 3)
  assert curve.genus == 0 and onp.allclose(curve.edges, [-b, b])
  assert abs(curve.filling_fractions().sum() - 1) < 1e-8
  x = 50.0
  assert abs(curve.y(x) - (curve.model.Vp(x) - 2 / x)) < 1e-3


def test_symmetric_two_cut_closed_form():
  curve = sc.spectral_curve(sc.QuarticModel(1.0, -3.0))
  assert curve.genus == 1
  assert onp.allclose(curve.edges, [-onp.sqrt(5), -1, 1, onp.sqrt(5)])
  assert onp.allclose(curve.filling_fractions(), 0.5)


def test_tilted_two_cut_periods():
  curve = sc.spectral_curve(sc.QuarticModel(1.0, -3.0, 0.25))
  eps = curve.filling_fractions()
  a_periods = onp.array([curve.a_period(i) for i in range(2)])
  assert abs(eps.sum() - 1) < 1e-8 and eps[0] > 0.55
  assert onp.allclose(a_periods, eps, atol=1e-8)
  assert abs(curve.b_period()) < 1e-10
  v = curve.effective_potential(onp.array([0.5 * sum(cut) for cut in curve.cuts]))
  assert onp.allclose(v, curve.fermi_level, atol=1e-6)


def test_log_gas_gaussian_moments():
  lam, acc = sc.sample_log_gas(lambda x: x**2 / 2, 40, n_chains=16, n_sweeps=160, n_burn=60, rng=0)
  assert 0.1 < acc < 0.9
  assert abs(onp.mean(lam**2) - 1) < 0.05


def test_abel_map_rectangle():
  curve = sc.spectral_curve(sc.QuarticModel(1.0, -3.0, 0.25))
  X, H, u, K_A, K_B = sc.abel_map(curve)
  e = curve.edges
  col = lambda x: onp.argmin(onp.abs(X - x))
  assert abs(u[0, col(0.5 * (e[2] + e[3]))].real + K_B) < 1e-3
  assert abs(u[0, col(0.5 * (e[1] + e[2]))].imag - K_A) < 1e-3


# 4. free probability -------------------------------------------------------------

def test_free_sums_by_resultant():
  G, Z = sc.G, sc.Z
  arcsine = sc.free_sum(sc.bernoulli_poly(1), sc.bernoulli_poly(1))
  assert sp.simplify(arcsine / (G**2 * (Z**2 - 4) - 1)).is_constant()
  double = sc.free_sum(sc.semicircle_poly(1), sc.semicircle_poly(1))
  assert sp.simplify(double / sc.semicircle_poly(2)).is_constant()
  x = onp.linspace(-1.9, 1.9, 39)
  assert onp.allclose(sc.density_from_curve(arcsine, x), 1 / (onp.pi * onp.sqrt(4 - x * x)), rtol=1e-5)


# 5. maps and topological recursion ---------------------------------------------------

@pytest.mark.parametrize("k", range(1, 6))
def test_gluings_by_genus_match_harer_zagier(k):
  genera = [sc.gluing_genus(sig) for sig in sc.pairings(k)]
  assert len(genera) == math.prod(range(1, 2 * k, 2))
  for g in range(k // 2 + 1):
    assert genera.count(g) == sc.harer_zagier(g, k)
  assert sc.harer_zagier(0, k) == math.comb(2 * k, k) // (k + 1)


def test_airy_curve_gives_witten_kontsevich_numbers():
  airy = sc.airy_curve()
  assert sc.intersection_numbers(airy, 0, 3) == {(0, 0, 0): 1}
  assert sc.intersection_numbers(airy, 1, 1) == {(1,): sp.Rational(1, 24)}
  assert sc.intersection_numbers(airy, 1, 2) == {(1, 1): sp.Rational(1, 24), (2, 0): sp.Rational(1, 24)}
  assert sc.intersection_numbers(airy, 2, 1) == {(4,): sp.Rational(1, 1152)}


def test_semicircle_curve_gives_genus_one_gluings():
  counts = sc.gluing_counts_from_curve(sc.semicircle_curve(), 1, 6)
  assert counts == [sc.harer_zagier(1, k) for k in range(7)]


# 6. finite free probability -----------------------------------------------------------

def test_finite_free_sum():
  p = sc.elementary_symmetric([Fraction(1), Fraction(-2), Fraction(3)])
  shift = sc.elementary_symmetric([Fraction(5)] * 3)   # (x - 5)^3 shifts every root by 5
  roots = onp.sort(sc.polynomial_roots(sc.finite_free_sum(p, shift)).real)
  assert onp.allclose(roots, [3, 6, 8])
  e = sc.elementary_symmetric([1] * 8 + [-1] * 8)
  r = sc.polynomial_roots(sc.finite_free_sum(e, e), dps=80)
  assert onp.max(onp.abs(r.imag)) < 1e-30 and onp.max(onp.abs(r.real)) < 2
