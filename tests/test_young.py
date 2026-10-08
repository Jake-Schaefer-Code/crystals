# tests/test_young.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math

import numpy as onp
import pytest

import src.invariants as inv
import src.symmetry.permutations as perms
import src.symmetry.young as young
from src.symmetry.group_algebra import GroupAlgebra


def _group_algebra_product(a, b):
  n = len(next(iter(a)))
  algebra = GroupAlgebra.symmetric(n)
  return (algebra.element(a) * algebra.element(b)).terms


# --------------------------------------------------------------------------- #
# Partitions and tableaux
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("n, count", [(1, 1), (2, 2), (3, 3), (4, 5), (5, 7), (6, 11), (7, 15)])
def test_partition_counts_match_the_partition_function(n, count):
  assert len(list(young.partitions(n))) == count


@pytest.mark.parametrize("lam", [(3, 1), (2, 2), (4, 2, 1), (2, 1, 1, 1), (3, 3)])
def test_conjugation_is_an_involution(lam):
  assert young.conjugate(young.conjugate(lam)) == tuple(lam)


@pytest.mark.parametrize("lam", [(2, 1), (3, 1), (2, 2), (3, 2, 1), (2, 1, 1)])
def test_hook_length_formula_counts_standard_tableaux(lam):
  assert young.dimension(lam) == len(list(young.standard_tableaux(lam)))


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_squared_irrep_dimensions_sum_to_the_group_order(n):
  total = sum(young.dimension(lam) ** 2 for lam in young.partitions(n))
  assert total == math.factorial(n)


def test_the_sign_and_trivial_irreps_are_one_dimensional():
  for n in range(1, 7):
    assert young.dimension((1,) * n) == 1
    assert young.dimension((n,)) == 1


# --------------------------------------------------------------------------- #
# Characters
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_the_column_irrep_character_is_the_permutation_sign(n):
  group = perms.PermutationGroup.symmetric(n)
  for g in group.elements:
    rho = tuple(perms.cycle_type(g))
    assert young.character((1,) * n, rho) == perms.sgn(g)


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_the_row_irrep_character_is_identically_one(n):
  for rho in young.partitions(n):
    assert young.character((n,), rho) == 1


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_character_degrees_match_the_hook_length_formula(n):
  identity = (1,) * n
  for lam in young.partitions(n):
    assert young.character(lam, identity) == young.dimension(lam)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_the_character_table_is_orthonormal(n):
  shapes, table = young.character_table(n)
  sizes = onp.array([young.class_size(rho) for rho in shapes], dtype=onp.int64)
  gram = (table * sizes) @ table.T / math.factorial(n)
  assert onp.allclose(gram, onp.eye(len(shapes)))


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_class_sizes_agree_with_the_permutations_module(n):
  for rho in young.partitions(n):
    assert young.class_size(rho) == perms.class_size_symmetric(rho)


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_class_sizes_sum_to_the_group_order(n):
  assert sum(young.class_size(rho) for rho in young.partitions(n)) == math.factorial(n)


def test_character_function_interoperates_with_CharG():
  n = 4
  chi_sign = young.character_function(n, (1,) * n)
  chi_triv = young.character_function(n, (n,))
  assert onp.isclose(complex(chi_sign | chi_sign).real, 1.0)
  assert onp.isclose(complex(chi_triv | chi_triv).real, 1.0)
  assert onp.isclose(complex(chi_sign | chi_triv).real, 0.0)


# --------------------------------------------------------------------------- #
# Molien: the joint test against src.invariants
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_molien_sign_series_matches_the_invariant_theory_closed_form(n):
  r"""Two independent routes to the dimension of the degree-d antisymmetric polynomials.

  ``young.molien_sign`` sums the sign character over conjugacy classes; the closed form in
  ``invariants`` uses freeness of the invariant ring. They share no code.
  """
  dmax = 14
  assert young.molien_sign(n, dmax) == inv.antisymmetric_dimensions(n, dmax)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_molien_of_the_trivial_character_counts_symmetric_polynomials(n):
  r"""The invariant ring is free on e_1..e_n, so its Hilbert series is 1 / prod_k (1 - q^k)."""
  dmax = 12
  got = young.molien(n, lambda rho: 1, dmax)
  expected = [0] * (dmax + 1)
  expected[0] = 1
  for k in range(1, n + 1):
    for d in range(k, dmax + 1):
      expected[d] += expected[d - k]
  assert got == expected


@pytest.mark.parametrize("n", [2, 3, 4])
def test_molien_isotypic_dimensions_sum_to_the_full_polynomial_ring(n):
  r"""Summing dim(lam) * (isotypic multiplicity) over irreps must give binom(d + n - 1, n - 1)."""
  dmax = 8
  total = [0] * (dmax + 1)
  for lam in young.partitions(n):
    series = young.molien(n, lambda rho, lam=lam: young.character(lam, rho), dmax)
    for d, c in enumerate(series):
      total[d] += young.dimension(lam) * c
  expected = [math.comb(d + n - 1, n - 1) for d in range(dmax + 1)]
  assert total == expected


# --------------------------------------------------------------------------- #
# Young symmetrizers
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("n", [2, 3, 4])
def test_the_single_column_symmetrizer_is_the_antisymmetrizer(n):
  tab = young.first_tableau((1,) * n)
  c = young.young_symmetrizer(tab, n)
  group = perms.PermutationGroup.symmetric(n)
  assert len(c) == math.factorial(n)
  for g in group.elements:
    assert c[g] == perms.sgn(g)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_the_single_row_symmetrizer_is_the_symmetrizer(n):
  tab = young.first_tableau((n,))
  c = young.young_symmetrizer(tab, n)
  assert len(c) == math.factorial(n)
  assert set(c.values()) == {1}


@pytest.mark.parametrize("lam", [(2, 1), (3, 1), (2, 2), (2, 1, 1)])
def test_the_young_symmetrizer_is_idempotent_up_to_its_known_scalar(lam):
  r"""``c^2 = (n! / dim lam) c``, the standard normalization of the Young symmetrizer."""
  n = sum(lam)
  tab = young.first_tableau(lam)
  c = young.young_symmetrizer(tab, n)
  c2 = _group_algebra_product(c, c)
  scale = math.factorial(n) // young.dimension(lam)
  assert set(c2) == set(c)
  for g, coeff in c.items():
    assert c2[g] == scale * coeff


@pytest.mark.parametrize("lam", [(2, 1), (2, 2), (3, 1)])
def test_the_normalized_symmetrizer_is_a_genuine_idempotent(lam):
  n = sum(lam)
  tab = young.first_tableau(lam)
  e = young.normalized_symmetrizer(tab, lam, n)
  e2 = _group_algebra_product(e, e)
  for g, coeff in e.items():
    assert e2[g] == coeff


def test_the_antisymmetrizer_annihilates_a_symmetric_function():
  r"""Applying the (1^n) symmetrizer to anything symmetric gives zero."""
  n = 3
  tab = young.first_tableau((1,) * n)
  c = young.young_symmetrizer(tab, n)
  x = [0.3, 1.2, 2.5]
  assert onp.isclose(young.apply_symmetrizer(c, lambda v: sum(v), x), 0.0)
  assert onp.isclose(young.apply_symmetrizer(c, lambda v: v[0] * v[1] * v[2], x), 0.0)


def test_the_antisymmetrizer_of_a_monomial_is_the_vandermonde():
  r"""Antisymmetrizing ``x_0^0 x_1^1 x_2^2`` gives the Vandermonde determinant exactly."""
  n = 3
  tab = young.first_tableau((1,) * n)
  c = young.young_symmetrizer(tab, n)
  x = [0.3, 1.2, 2.5]
  got = young.apply_symmetrizer(c, lambda v: v[0] ** 0 * v[1] ** 1 * v[2] ** 2, x)
  assert onp.isclose(got, float(inv.vandermonde(onp.array(x))))
