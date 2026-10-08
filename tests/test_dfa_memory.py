# tests/test_dfa_memory.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as onp
import pytest
from jax.scipy.linalg import expm

import physics.dfa_memory as dm
import physics.glauber as gl

P_ZERO = 0.3  # probability of reading a 0 in the div-3 automaton


def _col_stochastic(G):
  G = onp.asarray(G)
  return onp.allclose(G.sum(axis=0), 1.0) and G.min() >= -1e-12


# --------------------------------------------------------------------------- #
# The divisible-by-3 automaton
# --------------------------------------------------------------------------- #

def test_reading_a_bit_multiplies_the_remainder_by_two_and_adds_it():
  P0, P1 = (onp.asarray(P) for P in dm.make_div3_ops())
  for r in range(3):
    assert P0[(2 * r) % 3, r] == 1.0 and P1[(2 * r + 1) % 3, r] == 1.0
  assert _col_stochastic(P0) and _col_stochastic(P1)


def test_a_random_bit_string_tracks_its_value_mod_three():
  P = [onp.asarray(x) for x in dm.make_div3_ops()]
  bits = [1, 0, 1, 1, 0, 1, 0, 0, 1]
  state = onp.array([1.0, 0.0, 0.0])
  for b in bits:
    state = P[b] @ state
  assert int(onp.argmax(state)) == int("".join(map(str, bits)), 2) % 3


@pytest.mark.parametrize("m", [0, 1, 2, 3])
def test_memory_register_lumps_to_the_minimal_automaton_with_a_product_stationary_state(m):
  G = onp.asarray(dm.make_div3_memory_G(m, P_ZERO))
  G_min = onp.asarray(dm.make_div3_memory_G(0, P_ZERO))
  nh = 2**m
  assert _col_stochastic(G) and G.shape == (3 * nh, 3 * nh)
  L = onp.kron(onp.eye(3), onp.ones((1, nh)))                    # forget the history
  assert onp.allclose(L @ G, G_min @ L)
  # remainder is uniform and independent of the last m i.i.d. bits (a 0 has probability P_ZERO)
  history = onp.asarray(dm.stationary_mbit(m, 1.0 - P_ZERO)) if m else onp.ones(1)
  pi = onp.kron(onp.ones(3) / 3, history)
  assert onp.allclose(G @ pi, pi)


def test_loop_and_aperiodic_counters_are_stochastic_and_differ_only_in_when_they_advance():
  L = 3
  G_loop, G_aper = (onp.asarray(f(L, P_ZERO)) for f in (dm.make_div3_loop_G, dm.make_div3_aper_G))
  assert _col_stochastic(G_loop) and _col_stochastic(G_aper)
  assert not onp.allclose(G_loop, G_aper)
  R = onp.asarray(dm.cyclic_loop(L))
  assert onp.array_equal(onp.linalg.matrix_power(R, L), onp.eye(L))
  # always-advancing counter: the counter marginal is deterministic
  assert onp.allclose(onp.kron(onp.ones((1, 3)), onp.eye(L)) @ G_loop, R @ onp.kron(onp.ones((1, 3)), onp.eye(L)))


def test_mathieu_register_variants():
  G = onp.asarray(dm.make_mathieu10_G(P_ZERO))
  assert _col_stochastic(G) and G.shape == (33, 33)
  G2, G3 = onp.asarray(dm.make_mathieu10_G2(P_ZERO)), onp.asarray(dm.make_mathieu10_G3(P_ZERO))
  assert _col_stochastic(G2) and _col_stochastic(G3)
  assert onp.allclose(onp.asarray(dm.make_mathieu10_G(P_ZERO, on_1="identity")), G2)
  assert onp.allclose(onp.asarray(dm.make_mathieu10_G(P_ZERO, on_0="identity")), G3)
  both_off = onp.asarray(dm.make_mathieu10_G(P_ZERO, on_0="identity", on_1="identity"))
  assert onp.allclose(both_off, onp.kron(onp.asarray(dm.make_div3_memory_G(0, P_ZERO)), onp.eye(11)))
  with pytest.raises(ValueError):
    dm.make_mathieu10_G(P_ZERO, on_0="tau")


def test_the_symbolic_matrices_agree_with_the_numeric_ones():
  sp = pytest.importorskip("sympy")
  p, q = sp.Symbol("p"), sp.Symbol("q")
  sub = {p: 1 - P_ZERO, q: P_ZERO}      # the symbolic p is P(read a 1); the numeric p is P(read a 0)
  for m in range(3):
    G = onp.array(dm.make_div3_memory_G_sympy(m, p, q).subs(sub).tolist(), dtype=float)
    assert onp.allclose(G, onp.asarray(dm.make_div3_memory_G(m, P_ZERO)))
  for L in range(1, 4):
    G = onp.array(dm.make_div3_loop_G_sympy(L, p, q).subs(sub).tolist(), dtype=float)
    assert onp.allclose(G, onp.asarray(dm.make_div3_loop_G(L, P_ZERO)))
  G = dm.make_div3_memory_G_sympy(2, p, q)
  assert all(sp.simplify(sum(G[:, j]).subs(q, 1 - p)) == 1 for j in range(G.shape[1]))       # exact column sums


# --------------------------------------------------------------------------- #
# The shift register and the physical write step
# --------------------------------------------------------------------------- #

def test_four_state_automaton_detects_three_ones_in_a_row():
  pa = 0.75
  G = onp.asarray(dm.make_dfa_G(pa))
  assert _col_stochastic(G)
  pb = 1 - pa
  assert onp.allclose(G @ onp.array([0, 0, 1.0, 0]), [pa, 0, 0, pb])      # from state 2 a 1 absorbs
  assert onp.allclose(onp.linalg.matrix_power(G, 3000)[:, 0], [0, 0, 0, 1])


@pytest.mark.parametrize("m", [1, 2, 3])
def test_shift_register_stationary_state_is_iid_bits_and_its_last_bit_is_marginal(m):
  p = 0.35                                                        # probability of a 1
  G = onp.asarray(dm.make_mbit_memory_G(m, p))
  pi = onp.asarray(dm.stationary_mbit(m, p))
  assert _col_stochastic(G) and onp.allclose(G @ pi, pi) and pi.sum() == pytest.approx(1.0)
  assert onp.allclose(onp.asarray(dm.last_bit_projection(m)) @ pi, [1 - p, p])


def test_bit_write_generator_relaxes_to_the_gibbs_state_of_its_target():
  beta, Delta = 1.3, 4.0
  for b in (0, 1):
    K, pi = dm.bit_write_generator(b, beta=beta, Delta=Delta)
    assert onp.allclose(onp.asarray(K).sum(axis=0), 0.0)
    assert onp.allclose(K @ pi, 0.0, atol=1e-12)
    assert int(onp.argmax(onp.asarray(pi))) == b


def test_cyclic_shift_is_a_permutation_of_order_m():
  m = 4
  S = onp.asarray(dm.cyclic_shift_matrix(m))
  assert _col_stochastic(S) and onp.allclose(S.sum(axis=1), 1.0)
  assert onp.array_equal(onp.linalg.matrix_power(S, m), onp.eye(2**m))


def test_write_transition_flips_only_the_newest_bit_with_antisymmetric_energy_change():
  rows, dE, E = dm.make_memory_write_transition(3, 1, 2.0)
  rows, dE = onp.asarray(rows)[0], onp.asarray(dE)[0]
  assert onp.array_equal(rows, onp.arange(8) ^ 1)
  assert onp.allclose(dE, -dE[rows])
  assert set(onp.unique(onp.abs(dE))) == {2.0}


def test_physical_branch_is_the_write_exponential_followed_by_the_register_rotation():
  m, b, tau, beta, Delta = 3, 1, 0.4, 1.1, 3.0
  G_b, pi_b, A_cycle, c_cycle = dm.make_physical_mbit_branch(m, b, tau, beta=beta, Delta=Delta)
  rows, dE, _ = dm.make_memory_write_transition(m, b, Delta)
  K = gl.dense_generator_from_rates(rows, dm.rates_from_dE(dE, beta, 1.0))
  S = dm.cyclic_shift_matrix(m)
  assert _col_stochastic(G_b) and G_b.shape == (8, 8)
  assert onp.allclose(G_b, expm(tau * K) @ S, atol=1e-10)         # the notebook's own consistency check
  assert onp.allclose(c_cycle, beta * A_cycle)
  assert onp.allclose(onp.asarray(pi_b).sum(), 1.0)
