# tests/test_ansatz1d.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as onp
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

import src.invariants as inv
from physics import ansatz1d, models1d, nodal
from physics import vmc

KEY = jax.random.PRNGKey(0)
OMEGA = 1.0
LAM = 2.0


def _params(n, key=KEY, alpha=None, envelope=1.0):
  if alpha is None:
    alpha = float(ansatz1d.cusp_alpha(LAM * (LAM - 1.0)))
  return ansatz1d.init_params(key, n, alpha=alpha, envelope=envelope)


def _configs(n, n_walkers=16, key=KEY, spread=1.2):
  return jax.random.normal(key, (n_walkers, n, 1)) * spread


# --------------------------------------------------------------------------- #
# Cusp condition
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("lam", [1.0, 1.5, 2.0, 3.0])
def test_cusp_alpha_inverts_the_coupling(lam):
  g = lam * (lam - 1.0)
  assert onp.isclose(float(ansatz1d.cusp_alpha(g)), lam)


def test_a_free_interaction_gives_the_free_fermion_vandermonde_power():
  assert onp.isclose(float(ansatz1d.cusp_alpha(0.0)), 1.0)


# --------------------------------------------------------------------------- #
# Antisymmetry, via physics.nodal
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("n", [2, 3, 4])
def test_the_ansatz_is_exactly_antisymmetric(n):
  r"""Sign continuation off the chamber makes antisymmetry exact, not approximate.

  ``nodal.antisymmetry_error`` checks every same-spin transposition; with all spins equal that is
  every transposition in S_n.
  """
  psi = ansatz1d.wavefunction(_params(n))
  R = _configs(n)
  spins = jnp.zeros(n, dtype=int)
  n_sign_errors, max_log_error = nodal.antisymmetry_error(psi, R, spins)
  assert n_sign_errors == 0
  assert max_log_error < 1e-12


@pytest.mark.parametrize("n", [2, 3])
def test_the_ansatz_sign_is_the_sorting_sign(n):
  psi = ansatz1d.wavefunction(_params(n))
  rng = onp.random.default_rng(0)
  for _ in range(10):
    x = rng.normal(size=n)
    sign, _ = psi(jnp.asarray(x)[:, None])
    assert int(sign) == int(inv.sort_sign(jnp.asarray(x)))


def test_the_node_is_exactly_the_coincidence_set():
  r"""log|psi| diverges to -inf as two coordinates coalesce and nowhere else."""
  psi = ansatz1d.wavefunction(_params(3))
  far = jnp.array([[-1.0], [0.0], [1.0]])
  assert onp.isfinite(float(psi(far)[1]))
  for eps in (1e-2, 1e-4, 1e-6):
    close = jnp.array([[-1.0], [0.0], [eps]])
    assert float(psi(close)[1]) < float(psi(far)[1])
  touching = jnp.array([[-1.0], [0.0], [0.0]])
  assert float(psi(touching)[1]) == -onp.inf


def test_the_node_distance_is_finite_away_from_coincidences():
  psi = ansatz1d.wavefunction(_params(3))
  R = jnp.array([[-1.0], [0.2], [1.5]])
  assert onp.isfinite(float(nodal.node_distance(psi, R)))


# --------------------------------------------------------------------------- #
# Exact Calogero-Sutherland state
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("n, lam", [(3, 1.0), (3, 2.0), (4, 2.0), (4, 1.5)])
def test_the_exact_calogero_state_has_zero_variance_local_energy(n, lam):
  r"""Feeding the analytic ground state through the local-energy code must return E_0 flat.

  This is the sharpest available test of the kinetic term, the potential and the autodiff
  Laplacian at once: any inconsistency shows up as scatter. Needs float64 -- in float32 the
  second derivatives alone leave about 1e-3 of noise.
  """
  log_psi = models1d.calogero_log_psi(OMEGA, lam)
  V = models1d.calogero_potential(OMEGA, lam)
  R = jnp.sort(_configs(n, n_walkers=64, key=jax.random.PRNGKey(n)), axis=1)
  e_loc = vmc.batched_local_energy(log_psi, None, R, V)
  e_exact = models1d.calogero_energy(n, OMEGA, lam)
  assert onp.isclose(float(jnp.mean(e_loc)), e_exact, atol=1e-8)
  assert float(jnp.std(e_loc)) < 1e-8


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_the_free_fermion_limit_recovers_the_harmonic_spectrum(n):
  r"""lam = 1 switches the interaction off; E_0 must be the sum of the lowest n trap levels."""
  assert onp.isclose(
    models1d.calogero_energy(n, OMEGA, 1.0), models1d.free_fermion_energy(n, OMEGA)
  )


def test_the_exact_calogero_wavefunction_is_antisymmetric():
  psi = models1d.calogero_wavefunction(OMEGA, LAM)
  R = _configs(4)
  spins = jnp.zeros(4, dtype=int)
  n_sign_errors, max_log_error = nodal.antisymmetry_error(psi, R, spins)
  assert n_sign_errors == 0
  assert max_log_error < 1e-12


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #

def test_the_chamber_walk_never_leaves_the_chamber():
  n = 4
  params = _params(n)
  R = ansatz1d.init_walkers(jax.random.PRNGKey(1), 128, n)
  assert bool(jnp.all(vmc.chamber(R)))
  R, acc = vmc.metropolis(ansatz1d.log_psi, params, R, jax.random.PRNGKey(2), 20, 0.25)
  assert bool(jnp.all(vmc.chamber(R)))
  assert 0.05 < float(acc) < 0.99


def test_the_closed_form_vandermonde_gradient_matches_the_sampler_autodiff():
  x = jnp.array([-1.0, 0.3, 1.4])
  auto = jax.grad(inv.log_vandermonde)(x)
  assert onp.allclose(onp.asarray(auto), onp.asarray(inv.grad_log_vandermonde(x)))


# --------------------------------------------------------------------------- #
# Optimization
# --------------------------------------------------------------------------- #

def test_calogero_optimization_approaches_the_exact_energy():
  r"""The exact state lies inside the ansatz class, so this must converge, not merely improve.

  alpha is pinned at the cusp value and frozen; the envelope starts deliberately wrong so there
  is something to recover.
  """
  n, n_steps = 3, 300
  e_exact = models1d.calogero_energy(n, OMEGA, LAM)
  alpha = float(ansatz1d.cusp_alpha(LAM * (LAM - 1.0)))
  params = ansatz1d.init_params(jax.random.PRNGKey(3), n, alpha=alpha, envelope=0.6)
  R = ansatz1d.init_walkers(jax.random.PRNGKey(4), 256, n, spread=1.2)
  V = models1d.calogero_potential(OMEGA, LAM)

  params, R, history = vmc.optimize(
    ansatz1d.log_psi, params, R, jax.random.PRNGKey(5), V,
    n_steps=n_steps, n_sweeps=4, step=0.25, lr=5e-3, frozen=("log_alpha",),
  )

  early = onp.mean([row[1] for row in history[:20]])
  late = onp.mean([row[1] for row in history[-50:]])
  late_var = onp.mean([row[2] for row in history[-50:]])

  assert abs(late - e_exact) < abs(early - e_exact)
  assert abs(late - e_exact) < 0.05 * e_exact
  assert late_var < 1.0
  # alpha was frozen, so it must be untouched.
  assert onp.isclose(float(jnp.exp(params["log_alpha"])), alpha)


def test_freezing_a_parameter_group_leaves_it_unchanged():
  n = 3
  params = _params(n)
  before = float(params["log_alpha"])
  grads = jax.tree_util.tree_map(lambda p: jnp.ones_like(p), params)
  state = vmc.adam_init(params)
  state, params = vmc.adam_update(state, params, grads, lr=0.1, frozen=("log_alpha",))
  assert onp.isclose(float(params["log_alpha"]), before)
  assert not onp.isclose(float(params["log_envelope"]), float(_params(n)["log_envelope"]))
