# tests/test_group_algebra.py
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import random

import numpy as onp
import pytest

import src.symmetry.permutations as perms
import src.symmetry.young as young
from src.symmetry.group_algebra import GroupAlgebra


def _random_element(A, rng, k=4):
  basis = A.basis
  return A.element([(rng.choice(basis), Fraction(rng.randint(-3, 3), rng.randint(1, 3))) for _ in range(k)])


@pytest.fixture(scope="module")
def S4():
  return GroupAlgebra.of(perms.PermutationGroup.symmetric(4))


# --------------------------------------------------------------------------- #
# Ring axioms
# --------------------------------------------------------------------------- #

def test_the_unit_is_a_two_sided_identity(S4):
  rng = random.Random(0)
  x = _random_element(S4, rng)
  assert S4.unit() * x == x == x * S4.unit()


def test_multiplication_is_associative_and_distributes(S4):
  rng = random.Random(1)
  for _ in range(10):
    x, y, z = (_random_element(S4, rng) for _ in range(3))
    assert (x * y) * z == x * (y * z)
    assert x * (y + z) == x * y + x * z
    assert (x + y) * z == x * z + y * z


def test_multiplication_is_not_commutative_in_s3_but_is_in_a_cyclic_group():
  S3 = GroupAlgebra.symmetric(3)
  a = S3.basis_element((1, 0, 2))
  b = S3.basis_element((0, 2, 1))
  assert a * b != b * a
  C3 = GroupAlgebra.of(perms.PermutationGroup.generated([(1, 2, 0)]))
  g = C3.basis_element((1, 2, 0))
  assert g * (g + C3.unit()) == (g + C3.unit()) * g


def test_a_basis_element_multiplies_like_the_group(S4):
  g, h = (1, 2, 3, 0), (1, 0, 2, 3)
  assert S4.basis_element(g) * S4.basis_element(h) == S4.basis_element(perms.compose_perm(g, h))


def test_cancellation_prunes_zero_coefficients(S4):
  x = S4.basis_element((1, 0, 2, 3)) * 3
  assert not (x - x)
  assert (x - x) == S4.zero()
  assert len((x + x) - x) == 1


def test_scalar_multiplication_and_division_commute_with_the_product(S4):
  rng = random.Random(2)
  x, y = _random_element(S4, rng), _random_element(S4, rng)
  assert (x * Fraction(3, 2)) * y == (x * y) * Fraction(3, 2)
  assert 2 * x == x + x
  assert (x / 2) * 2 == x


def test_power_by_squaring_matches_repeated_multiplication(S4):
  rng = random.Random(3)
  x = _random_element(S4, rng)
  assert x ** 0 == S4.unit()
  assert x ** 5 == x * x * x * x * x
  with pytest.raises(ValueError):
    x ** -1


# --------------------------------------------------------------------------- #
# The involution
# --------------------------------------------------------------------------- #

def test_star_is_an_anti_automorphism_and_an_involution(S4):
  rng = random.Random(4)
  for _ in range(5):
    x, y = _random_element(S4, rng), _random_element(S4, rng)
    assert (x * y).star() == y.star() * x.star()
    assert x.star().star() == x


def test_star_conjugates_complex_coefficients():
  C3 = GroupAlgebra.of(perms.PermutationGroup.generated([(1, 2, 0)]))
  x = C3.element({(1, 2, 0): 1j})
  assert x.star() == C3.element({(2, 0, 1): -1j})


# --------------------------------------------------------------------------- #
# Membership
# --------------------------------------------------------------------------- #

def test_elements_outside_the_group_are_rejected():
  C3 = GroupAlgebra.of(perms.PermutationGroup.generated([(1, 2, 0)]))
  with pytest.raises(ValueError):
    C3.basis_element((1, 0, 2))
  with pytest.raises(ValueError):
    GroupAlgebra.symmetric(3).basis_element((0, 0, 1))
  with pytest.raises(ValueError):
    GroupAlgebra.symmetric(3).basis_element((0, 1))


def test_mixing_elements_of_different_algebras_raises():
  a = GroupAlgebra.symmetric(3).unit()
  b = GroupAlgebra.symmetric(4).unit()
  with pytest.raises(ValueError):
    a * b
  with pytest.raises(ValueError):
    a + b


def test_a_subgroup_algebra_is_closed_under_the_product():
  group = perms.PermutationGroup.generated([(1, 2, 0)])
  C3 = GroupAlgebra.of(group)
  x = C3.element({g: Fraction(1, 3) for g in group})
  assert x.is_idempotent()  # the trivial-rep projector (1/|G|) sum g


# --------------------------------------------------------------------------- #
# Regular representation
# --------------------------------------------------------------------------- #

def test_the_structure_tensor_is_associative_and_has_one_entry_per_pair():
  A = GroupAlgebra.of(perms.PermutationGroup.symmetric(3))
  T = A.structure_tensor()
  assert T.shape == (6, 6, 6)
  assert onp.all(T.sum(axis=2) == 1)
  left = onp.einsum("abe,ecf->abcf", T.astype(int), T.astype(int))
  right = onp.einsum("bce,aef->abcf", T.astype(int), T.astype(int))
  assert onp.array_equal(left, right)


def test_left_regular_is_a_homomorphism_and_exact_over_fractions(S4):
  rng = random.Random(5)
  x, y = _random_element(S4, rng), _random_element(S4, rng)
  Lx, Ly, Lxy = S4.left_regular(x), S4.left_regular(y), S4.left_regular(x * y)
  assert Lx.dtype == object
  assert onp.array_equal(Lx @ Ly, Lxy)
  assert onp.array_equal(S4.left_regular(S4.unit()), onp.eye(24, dtype=int))


def test_left_regular_of_a_group_element_is_a_permutation_matrix(S4):
  M = S4.left_regular(S4.basis_element((1, 2, 3, 0)))
  assert onp.array_equal(M.sum(axis=0), onp.ones(24)) and onp.array_equal(M.sum(axis=1), onp.ones(24))


def test_left_regular_matches_the_structure_tensor():
  A = GroupAlgebra.of(perms.PermutationGroup.symmetric(3))
  T = A.structure_tensor()
  for a, g in enumerate(A.basis):
    M = A.left_regular(A.basis_element(g))
    for b in range(6):
      assert onp.array_equal(M[:, b], T[a, b, :])


def test_trace_is_the_normalized_regular_character():
  A = GroupAlgebra.of(perms.PermutationGroup.symmetric(3))
  rng = random.Random(6)
  x = _random_element(A, rng)
  assert onp.trace(A.left_regular(x)) == A.order * x.trace()


# --------------------------------------------------------------------------- #
# Consumers of the product
# --------------------------------------------------------------------------- #

def test_the_antisymmetrizer_over_n_factorial_is_an_idempotent():
  A = GroupAlgebra.symmetric(4)
  tab = young.first_tableau((1, 1, 1, 1))
  c = A.element(young.young_symmetrizer(tab, 4)) / 24
  assert c.is_idempotent()
  assert c.star() == c


def test_the_normalized_young_symmetrizer_is_idempotent_but_not_hermitian_in_general():
  A = GroupAlgebra.symmetric(4)
  tab = young.first_tableau((2, 1, 1))
  e = A.element(young.normalized_symmetrizer(tab, (2, 1, 1), 4))
  assert e.is_idempotent()


def test_convolve_matches_the_direct_double_sum():
  G = perms.PermutationGroup.symmetric(3)
  rng = random.Random(7)
  P = {g: rng.random() for g in G}
  Q = {g: rng.random() for g in G}
  expected = {g: 0.0 for g in G}
  for g, pg in P.items():
    for h, qh in Q.items():
      expected[perms.compose_perm(g, h)] += pg * qh
  got = perms.convolve(G, P, Q)
  assert got.keys() == expected.keys()
  assert all(onp.isclose(got[g], expected[g]) for g in G)


def test_power_dist_keeps_total_mass_one():
  G = perms.PermutationGroup.symmetric(3)
  step = {g: 0.0 for g in G}
  step[(1, 0, 2)] = step[(0, 2, 1)] = 0.5
  assert onp.isclose(sum(perms.power_dist(G, step, 5).values()), 1.0)


# --------------------------------------------------------------------------- #
# Generic in the element type
# --------------------------------------------------------------------------- #

from src.symmetry.group_laws import AffineLaw, PermutationLaw
from src.symmetry.named_groups import cyclic_group, dihedral_group, tetrahedral_group
from src.symmetry.symmetry import AffineOperation
from src.symmetry.operations import rotation2d


def test_a_subscripted_algebra_builds_the_same_object():
  assert GroupAlgebra[perms.Permutation].symmetric(3) == GroupAlgebra.symmetric(3)
  assert GroupAlgebra[AffineOperation].of(dihedral_group(4)).order == 8


@pytest.fixture(scope="module")
def D4():
  return GroupAlgebra[AffineOperation].of(dihedral_group(4))


def test_the_affine_group_algebra_satisfies_the_ring_axioms(D4):
  rng = random.Random(10)
  for _ in range(5):
    x, y, z = (_random_element(D4, rng) for _ in range(3))
    assert (x * y) * z == x * (y * z)
    assert x * (y + z) == x * y + x * z
    assert D4.unit() * x == x == x * D4.unit()


def test_a_float_valued_group_closes_despite_roundoff(D4):
  r = D4.basis_element(rotation2d(onp.pi / 2))
  assert (r ** 4) == D4.unit()
  assert (r ** 2) != D4.unit()


def test_the_regular_representation_of_a_point_group_is_a_homomorphism(D4):
  rng = random.Random(11)
  x, y = _random_element(D4, rng), _random_element(D4, rng)
  assert onp.array_equal(D4.left_regular(x) @ D4.left_regular(y), D4.left_regular(x * y))
  T = D4.structure_tensor()
  assert T.shape == (8, 8, 8) and onp.all(T.sum(axis=2) == 1)


def test_the_defining_matrices_extend_to_an_algebra_homomorphism(D4):
  rng = random.Random(12)
  x, y = _random_element(D4, rng), _random_element(D4, rng)
  matrix = lambda g: g.matrix
  assert onp.allclose((x * y).represent(matrix), x.represent(matrix) @ y.represent(matrix))


@pytest.mark.parametrize("make", [lambda: dihedral_group(4), lambda: cyclic_group(5), tetrahedral_group])
def test_the_normalized_sum_represents_as_the_reynolds_projector(make):
  action = make()
  A = GroupAlgebra[AffineOperation].of(action)
  e = A.element([(g, Fraction(1, A.order)) for g in A.basis])
  assert e.is_idempotent(atol=1e-12)
  P = e.represent(lambda g: g.matrix)
  trivial = action.linear_projector([1.0] * action.order)
  assert onp.allclose(P, trivial)


def test_star_on_a_point_group_inverts_rotations(D4):
  r = rotation2d(onp.pi / 2)
  assert D4.basis_element(r).star() == D4.basis_element(r.inverse())


def test_a_set_that_is_not_a_group_is_rejected():
  with pytest.raises(ValueError, match="closed"):
    AffineLaw((rotation2d(0.0), rotation2d(onp.pi / 2)))


def test_equal_groups_built_separately_share_an_algebra():
  a, b = GroupAlgebra.of(dihedral_group(4)), GroupAlgebra.of(dihedral_group(4))
  assert a == b
  assert a.unit() + b.unit() == 2 * a.unit()


def test_algebras_of_different_element_types_do_not_mix(D4):
  with pytest.raises(ValueError):
    D4.unit() * GroupAlgebra.symmetric(2).unit()


def test_of_adapts_each_group_type_to_its_law():
  assert isinstance(GroupAlgebra.of(perms.PermutationGroup.symmetric(3)).group, PermutationLaw)
  assert isinstance(GroupAlgebra.of(dihedral_group(3)).group, AffineLaw)
  law = PermutationLaw.symmetric(3)
  assert GroupAlgebra.of(law).group is law
