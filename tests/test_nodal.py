# tests/test_nodal.py
import itertools as it
import sys
from pathlib import Path

import numpy as onp
import pytest
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
jax.config.update("jax_enable_x64", True)

from physics import nodal, nodal_models as models
import src.symmetry.permutations as perms

KEY = jax.random.PRNGKey(0)
HO1 = models.slater_determinant(models.harmonic_orbitals(1, 3), [0] * 3)
HO2 = models.slater_determinant(models.harmonic_orbitals(2, 3), [0] * 3)


def _equilibrated(psi, n, dim, step, key, n_walkers=16):
  samples, _ = nodal.metropolis(psi, jax.random.normal(key, (n_walkers, n, dim)), key, 200, step)
  return samples[-1]


def _same_up_to_global_sign(s, reference):
  ratio = onp.asarray(s) * onp.sign(reference)
  return onp.all(ratio == ratio[0]) and ratio[0] != 0


def test_1d_harmonic_node_is_vandermonde():
  psi = models.slater_determinant(models.harmonic_orbitals(1, 4), [0] * 4)
  R = jax.random.normal(KEY, (200, 4, 1))
  x = onp.asarray(R[..., 0])
  vandermonde = onp.prod([x[:, j] - x[:, i] for i, j in it.combinations(range(4), 2)], axis=0)
  assert _same_up_to_global_sign(nodal.evaluate(psi, R)[0], vandermonde)


def test_2d_harmonic_node_is_collinearity():
  R = onp.asarray(jax.random.normal(KEY, (200, 3, 2)))
  e1, e2 = R[:, 1] - R[:, 0], R[:, 2] - R[:, 0]
  signed_area = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
  assert _same_up_to_global_sign(nodal.evaluate(HO2, R)[0], signed_area)


def test_beryllium_hf_node_is_radial():
  R = onp.asarray(jax.random.normal(KEY, (200, 4, 3)))
  r = onp.linalg.norm(R, axis=-1)
  label = (r[:, 1] - r[:, 0]) * (r[:, 3] - r[:, 2])
  assert _same_up_to_global_sign(nodal.evaluate(models.beryllium(0.0), R)[0], label)


@pytest.mark.parametrize("psi, spins, dim", [
  (models.slater_determinant(models.harmonic_orbitals(2, 3), [0, 0, 0, 1, 1, 1]), [0, 0, 0, 1, 1, 1], 2),
  (models.slater_determinant(models.plane_wave_orbitals(3, 7, 5.0), [0] * 7), [0] * 7, 3),
  (models.beryllium(-0.2), [0, 0, 1, 1], 3),
])
def test_models_are_antisymmetric(psi, spins, dim):
  n_sign_errors, log_error = nodal.antisymmetry_error(psi, jax.random.normal(KEY, (20, len(spins), dim)), spins)
  assert n_sign_errors == 0 and log_error < 1e-10


def test_node_distance_is_exact_for_a_linear_psi():
  psi = lambda R: (jnp.sign(R[1, 0] - R[0, 0]), jnp.log(jnp.abs(R[1, 0] - R[0, 0])))
  R = jax.random.normal(KEY, (50, 2, 1))
  exact = jnp.abs(R[:, 1, 0] - R[:, 0, 0]) / jnp.sqrt(2.0)
  assert jnp.allclose(nodal.node_distance(psi, R), exact)


def test_fixed_node_walk_stays_in_its_cell():
  # Be HF cells are labelled by sign(|r1| - |r0|) and sign(|r3| - |r2|). The (+,+) and (-,-) cells
  # share the sign of psi, so a walker jumping between them would go unnoticed by the sign alone.
  psi = models.beryllium(0.0)
  R0 = _equilibrated(psi, 4, 3, 0.25, KEY)[0]
  r0 = onp.linalg.norm(onp.asarray(R0), axis=-1)
  label0 = onp.sign([r0[1] - r0[0], r0[3] - r0[2]])

  def labels(samples):
    r = onp.linalg.norm(onp.asarray(samples), axis=-1)
    return onp.sign(onp.stack([r[..., 1] - r[..., 0], r[..., 3] - r[..., 2]], axis=-1))

  walkers = jnp.broadcast_to(R0, (32, 4, 3))
  fixed, acc = nodal.metropolis(psi, walkers, KEY, 400, 0.4, fixed_node=True)
  free, _ = nodal.metropolis(psi, walkers, KEY, 400, 0.4)
  assert acc > 0.3
  assert onp.all(labels(fixed) == label0)
  assert not onp.all(labels(free) == label0)


def test_jacobi_slice_of_three_fermions_has_six_wedges():
  d1, d2 = nodal.jacobi_plane(3)
  _, _, sign, _ = nodal.slice_grid(HO1, jnp.zeros((3, 1)), d1, d2, 3.0, n=200)
  assert nodal.count_pockets(sign) == (3, 3)


def test_stabilizer_order_matches_group_closure():
  spins = onp.array([0, 0, 0, 0, 1, 1, 1])
  cyc = lambda a, b, c: nodal._relabel(7, {a: b, b: c, c: a})
  double = nodal._relabel(7, {0: 1, 1: 0, 4: 5, 5: 4})
  for certified in ([cyc(0, 1, 2)],
                    [cyc(0, 1, 2), cyc(2, 3, 0)],
                    [cyc(0, 1, 2), cyc(4, 5, 6)],
                    [cyc(0, 1, 2), cyc(1, 2, 3), cyc(4, 5, 6), double]):
    shortcut = nodal._stabilizer_order(certified, spins, group_order=144, max_group_order=0)
    assert shortcut == perms.generated_group(certified).order


def _census(psi, spins, dim, step):
  k_equil, k_census = jax.random.split(KEY)
  R0 = _equilibrated(psi, len(spins), dim, step, k_equil)[0]
  return nodal.cell_census(psi, spins, R0, k_census, step=step)


def test_census_one_dimension_has_n_factorial_cells():
  census = _census(HO1, [0] * 3, 1, 0.5)
  assert census.certified == () and census.n_cells_upper == 6


def test_census_polarized_2d_trap_has_two_cells():
  assert _census(HO2, [0] * 3, 2, 0.5).n_cells_upper == 2


def test_census_beryllium_hartree_fock_vs_two_configuration():
  hf = _census(models.beryllium(0.0), [0, 0, 1, 1], 3, 0.25)
  assert hf.certified == () and hf.n_cells_upper == 4
  psi = models.beryllium(-0.2)
  mc = _census(psi, [0, 0, 1, 1], 3, 0.25)
  assert mc.n_cells_upper == 2 and mc.certified == ((1, 0, 3, 2),)
  for p, path in mc.witnesses.items():
    assert onp.allclose(path[-1], path[0][list(p)])
    sign, _ = nodal.evaluate(psi, path)
    assert onp.all(sign == sign[0])


def test_sign_agreement():
  R = _equilibrated(HO2, 3, 2, 0.5, KEY, n_walkers=256)
  flipped = lambda R: (-HO2(R)[0], HO2(R)[1])
  other = models.slater_determinant(models.plane_wave_orbitals(2, 3, 6.0), [0] * 3)
  assert nodal.sign_agreement(HO2, flipped, R) == 1.0
  assert nodal.sign_agreement(HO2, other, R) < 0.95
