# tests/test_grassmann.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as onp
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

import src.grassmann as gr

KEY = jax.random.PRNGKey(0)


def _random_frame(key, n_orb, n_occ):
  return jax.random.normal(key, (n_orb, n_occ))


# --------------------------------------------------------------------------- #
# The Plucker embedding
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("n_orb, n_occ", [(4, 2), (5, 2), (5, 3), (6, 3)])
def test_plucker_coordinate_count_is_the_binomial(n_orb, n_occ):
  import math
  p = gr.plucker_coordinates(_random_frame(KEY, n_orb, n_occ))
  assert p.shape == (math.comb(n_orb, n_occ),)


@pytest.mark.parametrize("n_orb, n_occ", [(4, 2), (5, 2), (5, 3), (6, 3), (6, 2)])
def test_plucker_relations_vanish_on_a_slater_determinant(n_orb, n_occ):
  r"""A single Slater determinant is by definition a point of the Grassmannian."""
  key = jax.random.PRNGKey(n_orb * 10 + n_occ)
  p = gr.plucker_coordinates(_random_frame(key, n_orb, n_occ))
  rel = gr.plucker_relations(p, n_orb, n_occ)
  assert onp.allclose(onp.asarray(rel), 0.0, atol=1e-10)


@pytest.mark.parametrize("n_orb, n_occ", [(5, 2), (5, 3), (6, 3)])
def test_a_generic_ci_vector_is_not_a_single_determinant(n_orb, n_occ):
  import math
  key = jax.random.PRNGKey(n_orb * 100 + n_occ)
  p = jax.random.normal(key, (math.comb(n_orb, n_occ),))
  assert float(gr.decomposability_residual(p, n_orb, n_occ)) > 1e-3


def test_the_plucker_point_depends_only_on_the_span():
  r"""A GL(n_occ) mixing of the occupied orbitals rescales p by a determinant, nothing more."""
  k1, k2 = jax.random.split(KEY)
  C = _random_frame(k1, 5, 3)
  M = jax.random.normal(k2, (3, 3))
  p0 = gr.plucker_coordinates(C)
  p1 = gr.plucker_coordinates(C @ M)
  assert onp.allclose(onp.asarray(p1), float(jnp.linalg.det(M)) * onp.asarray(p0))


def test_the_decomposability_residual_is_scale_invariant():
  key = jax.random.PRNGKey(7)
  p = jax.random.normal(key, (10,))
  base = float(gr.decomposability_residual(p, 5, 2))
  for s in (0.1, 3.0, 17.0):
    assert onp.isclose(float(gr.decomposability_residual(s * p, 5, 2)), base)


# --------------------------------------------------------------------------- #
# The Klein quadric and the Pfaffian
# --------------------------------------------------------------------------- #

def test_the_klein_quadric_is_the_four_by_four_pfaffian():
  r"""The decomposability test for two fermions and the pairing Pfaffian are one polynomial."""
  key = jax.random.PRNGKey(3)
  for _ in range(5):
    key, sub = jax.random.split(key)
    p = jax.random.normal(sub, (6,))
    C = gr.coefficient_matrix(p, 4)
    assert onp.isclose(float(gr.klein_quadric(p)), float(gr.pfaffian(C)))


def test_the_klein_quadric_is_the_only_plucker_relation_for_gr_2_4():
  key = jax.random.PRNGKey(4)
  p = jax.random.normal(key, (6,))
  rel = gr.plucker_relations(p, 4, 2)
  nonzero = onp.asarray(rel)[onp.abs(onp.asarray(rel)) > 1e-12]
  assert len(nonzero) > 0
  assert onp.allclose(onp.abs(nonzero), abs(float(gr.klein_quadric(p))))


def test_the_klein_quadric_vanishes_on_a_single_determinant():
  key = jax.random.PRNGKey(5)
  for _ in range(5):
    key, sub = jax.random.split(key)
    p = gr.plucker_coordinates(_random_frame(sub, 4, 2))
    assert onp.isclose(float(gr.klein_quadric(p)), 0.0, atol=1e-10)


@pytest.mark.parametrize("n", [2, 4, 6])
def test_the_pfaffian_squares_to_the_determinant(n):
  key = jax.random.PRNGKey(n)
  M = jax.random.normal(key, (n, n))
  A = M - M.T
  assert onp.isclose(float(gr.pfaffian(A)) ** 2, float(jnp.linalg.det(A)))


def test_the_pfaffian_rejects_odd_dimensions():
  with pytest.raises(ValueError):
    gr.pfaffian(jnp.zeros((3, 3)))


# --------------------------------------------------------------------------- #
# Slater rank
# --------------------------------------------------------------------------- #

def test_slater_rank_of_a_single_determinant_is_one():
  key = jax.random.PRNGKey(11)
  for n_orb in (4, 5, 6):
    key, sub = jax.random.split(key)
    p = gr.plucker_coordinates(_random_frame(sub, n_orb, 2))
    assert gr.slater_rank(p, n_orb) == 1


def test_slater_rank_of_a_generic_two_fermion_state_is_maximal():
  key = jax.random.PRNGKey(12)
  for n_orb in (4, 6):
    key, sub = jax.random.split(key)
    import math
    p = jax.random.normal(sub, (math.comb(n_orb, 2),))
    assert gr.slater_rank(p, n_orb) == n_orb // 2


def test_the_coefficient_matrix_is_antisymmetric():
  key = jax.random.PRNGKey(13)
  p = jax.random.normal(key, (10,))
  C = onp.asarray(gr.coefficient_matrix(p, 5))
  assert onp.allclose(C, -C.T)
