# tests/test_invariants.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import itertools as it

import numpy as onp
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

import src.invariants as inv
import src.symmetry.permutations as perms

KEY = jax.random.PRNGKey(0)
X3 = jnp.array([-1.3, 0.2, 1.7])
X5 = jnp.array([-2.0, -0.7, 0.1, 0.9, 2.4])


def test_vandermonde_is_the_product_of_gaps():
  x = X3
  expected = (x[1] - x[0]) * (x[2] - x[0]) * (x[2] - x[1])
  assert onp.isclose(float(inv.vandermonde(x)), float(expected))


def test_log_vandermonde_matches_the_log_of_the_product():
  for x in (X3, X5):
    assert onp.isclose(float(inv.log_vandermonde(x)), float(jnp.log(inv.vandermonde(x))))


def test_vandermonde_is_antisymmetric_under_transpositions():
  x = onp.asarray(X5)
  base = float(inv.vandermonde(x))
  for i, j in it.combinations(range(len(x)), 2):
    y = x.copy()
    y[i], y[j] = y[j], y[i]
    assert onp.isclose(float(inv.vandermonde(y)), -base)


def test_closed_form_vandermonde_gradient_matches_autodiff():
  g_auto = jax.grad(lambda x: inv.log_vandermonde(x))(X5)
  g_exact = inv.grad_log_vandermonde(X5)
  assert onp.allclose(onp.asarray(g_auto), onp.asarray(g_exact))


def test_elementary_symmetric_are_the_coefficients_of_the_product():
  x = onp.asarray(X5)
  # onp.poly gives coefficients of prod(t - x_i); ours are of prod(t + x_i).
  expected = onp.poly(-x)
  assert onp.allclose(onp.asarray(inv.elementary_symmetric(X5)), expected)


def test_newton_identities_recover_the_elementary_symmetric():
  n = X5.shape[0]
  p = inv.power_sums(X5, n)
  assert onp.allclose(
    onp.asarray(inv.elementary_from_power_sums(p, n)),
    onp.asarray(inv.elementary_symmetric(X5)),
  )


def test_sorting_sign_matches_the_parity_of_the_sorting_permutation():
  rng = onp.random.default_rng(0)
  for _ in range(20):
    x = rng.normal(size=6)
    order = tuple(int(k) for k in onp.argsort(x))
    assert int(inv.sort_sign(x)) == perms.sgn(order)


def test_chamber_projection_leaves_sorted_input_alone():
  x_sorted, sign = inv.to_chamber(X5)
  assert onp.allclose(onp.asarray(x_sorted), onp.asarray(X5))
  assert int(sign) == 1
  assert bool(inv.in_chamber(X5))


def test_symmetric_features_are_permutation_invariant():
  x = onp.asarray(X5)
  base = onp.asarray(inv.symmetric_features(jnp.sort(jnp.asarray(x))))
  rng = onp.random.default_rng(1)
  for _ in range(5):
    y = jnp.sort(jnp.asarray(rng.permutation(x)))
    assert onp.allclose(onp.asarray(inv.symmetric_features(y)), base)


def test_symmetric_features_stay_finite_as_particles_coalesce():
  # The whole point of avoiding log(gap) and min(gap): nothing may blow up here.
  for eps in (1e-2, 1e-4, 1e-8):
    x = jnp.array([0.0, eps, 1.0])
    feats = onp.asarray(inv.symmetric_features(x))
    assert onp.all(onp.isfinite(feats))


@pytest.mark.parametrize("n", [2, 3, 4])
def test_antisymmetric_dimensions_begin_at_the_vandermonde_degree(n):
  dims = inv.antisymmetric_dimensions(n, 12)
  offset = n * (n - 1) // 2
  assert dims[:offset] == [0] * offset
  assert dims[offset] == 1


def test_schur_of_a_single_box_is_the_first_power_sum():
  x = jnp.array([0.4, 1.1])
  assert onp.isclose(float(inv.schur((1, 0), x)), float(jnp.sum(x)))


def test_schur_functions_are_symmetric():
  x = jnp.array([0.4, 1.1, 2.3])
  base = float(inv.schur((2, 1, 0), x))
  for p in it.permutations(range(3)):
    y = x[jnp.array(p)]
    assert onp.isclose(float(inv.schur((2, 1, 0), y)), base)


def test_vandermonde_factorization_is_exact_for_a_slater_determinant():
  # det(x_i^j) for monomial orbitals IS the Vandermonde: the lam = 0 Schur function is 1.
  x = X5
  assert onp.isclose(float(inv.schur((0,) * 5, x)), 1.0)
