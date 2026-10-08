# physics/dfa_memory.py
r"""Column-stochastic maps of automata read one random bit at a time, with and without a memory register.

Every ``G`` here is column-stochastic, ``p -> G @ p``. The central example is the divisible-by-3
automaton: its three states are the remainder ``r`` of the number read so far, and reading bit ``b``
sends ``r`` to ``2r + b (mod 3)``. In terms of permutation matrices that is ``P0`` (bit 0, swaps
remainders 1 and 2) and ``P1`` (bit 1, swaps 0 and 1), so ``G = p P0 + (1 - p) P1`` when the bit is 0
with probability ``p``. Redundant registers make the same computation carry more state:

- ``make_div3_memory_G(m, p)``: also remember the last ``m`` bits read (``2**m`` histories);
- ``make_div3_loop_G`` / ``make_div3_aper_G``: pair the automaton with a cyclic counter of length
  ``L``, advanced on every step or only when reading a 1;
- ``make_mathieu10_G``: pair it with permutations of 11 points that generate part of the Mathieu
  group ``M_11`` (an 11-cycle and an element of cycle type ``4^2 1^3``);
- the ``*_sympy`` versions return exact symbolic matrices in ``p`` and ``q = 1 - p``, with ``p`` the
  probability of a 1 (the numeric functions use ``p`` for a 0).

The ``m``-bit shift register (``make_mbit_memory_G``), the single-bit write generator and the
physical branch that implements one write step (``make_physical_mbit_branch``) follow the same
convention and feed ``glauber.precompute_ep``.

Moved from ``notebooks/stoch_thermo/DFA.ipynb`` and ``prior.ipynb``.
"""
from __future__ import annotations

import jax.numpy as jnp
import jax.scipy.linalg as jsp

import src.symmetry.permutations as perms
from physics import glauber


def make_div3_ops():
  r""" Permutation matrices ``(P0, P1)`` of reading a 0 and a 1 in the divisible-by-3 automaton. """
  g23 = perms.from_cycles(((2, 3),), degree=3)
  g12 = perms.from_cycles(((1, 2),), degree=3)
  P0 = perms.perm_hom(g23)
  P1 = perms.perm_hom(g12)
  return P0, P1


def make_div3_memory_G(m, p):
  # m = number of redundant remembered bits
  # m = 0 gives the minimal 3-state DFA
  m = int(m)
  nh = 2 ** m
  P0, P1 = make_div3_ops()

  if m == 0:
    return p * P0 + (1 - p) * P1

  # all bits below m-1 bit
  mask = 2 ** (m - 1) - 1

  G_h0, G_h1 = jnp.zeros((nh, nh)), jnp.zeros((nh, nh))
  for h in range(nh):
    tail = h & mask
    # New histories
    h0 = (tail << 1) | 0
    h1 = (tail << 1) | 1
    G_h0 = G_h0.at[h0, h].add(1)
    G_h1 = G_h1.at[h1, h].add(1)

  G0 = jnp.kron(P0, G_h0)
  G1 = jnp.kron(P1, G_h1)
  return p * G0 + (1 - p) * G1


def cyclic_loop(L):
  r""" Permutation matrix of the cycle ``a -> a + 1 (mod L)``. """
  perm = perms.from_cycles((tuple(range(1, L + 1)),))
  return perms.perm_hom(perm)


def make_div3_loop_G(L, p):
  r""" The minimal automaton times a counter that advances on every step. """
  P0, P1 = make_div3_ops()
  G_min = p * P0 + (1 - p) * P1
  R = cyclic_loop(L)
  return jnp.kron(G_min, R)


def make_div3_aper_G(L, p):
  r""" The automaton with a counter that advances only when a 1 is read. """
  P0, P1 = make_div3_ops()
  I = jnp.eye(L)
  R = cyclic_loop(L)
  G0 = p * jnp.kron(P0, I)
  G1 = (1 - p) * jnp.kron(P1, R)
  return G0 + G1


def make_mathieu10_G(p, *, on_0: str = "sigma", on_1: str = "tau"):
  r"""The automaton times an 11-point register acted on by ``sigma`` (11-cycle) and ``tau`` (``4^2 1^3``).

  ``on_0`` is what reading a 0 does to the register, ``"sigma"`` or ``"identity"``; ``on_1`` is what
  reading a 1 does, ``"tau"`` or ``"identity"``.
  """
  if on_0 not in ("sigma", "identity") or on_1 not in ("tau", "identity"):
    raise ValueError("on_0 must be 'sigma' or 'identity' and on_1 must be 'tau' or 'identity'")
  P0, P1 = make_div3_ops()
  sigma = perms.from_cycles((tuple(range(1, 12)),))
  tau = perms.from_cycles(((3, 7, 11, 8), (4, 10, 5, 6)), degree=11)
  R0 = perms.perm_hom(sigma) if on_0 == "sigma" else jnp.eye(11)
  R1 = perms.perm_hom(tau) if on_1 == "tau" else jnp.eye(11)
  return p * jnp.kron(P0, R0) + (1 - p) * jnp.kron(P1, R1)


def make_mathieu10_G2(p):
  r""" ``make_mathieu10_G`` with the register untouched by a 1. """
  return make_mathieu10_G(p, on_1="identity")


def make_mathieu10_G3(p):
  r""" ``make_mathieu10_G`` with the register untouched by a 0. """
  return make_mathieu10_G(p, on_0="identity")


# --------------------------------------------------------------------------- #
# Exact symbolic versions
# --------------------------------------------------------------------------- #

def make_div3_memory_G_sympy(m, p, q):
  r"""Exact version of ``make_div3_memory_G``, with symbols ``p`` and ``q``.

  Note the convention: here ``q`` multiplies reading a 0 and ``p`` reading a 1, whereas in the
  numeric functions ``p`` is the probability of a 0. Substitute ``p -> 1 - p_numeric`` and
  ``q -> p_numeric`` to compare. The same holds for the other ``*_sympy`` functions.
  """
  import sympy as sp

  nh = 1 if m == 0 else 2**m
  n = 3 if m == 0 else 3 * nh
  G = sp.MutableDenseMatrix.zeros(n, n)

  def idx(r, h=0):
    return r if m == 0 else r * nh + h

  mask = 0 if m == 0 else 2**(m - 1) - 1

  for r in range(3):
    for h in range(nh):
      source = idx(r, h)
      tail = h & mask
      h0 = 0 if m == 0 else (tail << 1)
      h1 = 0 if m == 0 else ((tail << 1) | 1)

      r0 = (2 * r) % 3
      r1 = (2 * r + 1) % 3

      G[idx(r0, h0), source] += q
      G[idx(r1, h1), source] += p

  return sp.Matrix(G)


def cyclic_loop_sympy(L):
  import sympy as sp

  R = sp.zeros(L, L)
  for a in range(L):
    R[(a + 1) % L, a] = 1
  return R


def make_div3_G_sympy(p, q):
  import sympy as sp

  G = sp.zeros(3, 3)
  for r in range(3):
    r0 = (2 * r) % 3
    r1 = (2 * r + 1) % 3
    G[r0, r] += q
    G[r1, r] += p
  return G


def make_div3_loop_G_sympy(L, p, q):
  import sympy as sp

  G_min = make_div3_G_sympy(p, q)
  R = cyclic_loop_sympy(L)
  return sp.kronecker_product(G_min, R)


# --------------------------------------------------------------------------- #
# A four-state automaton and the m-bit shift register
# --------------------------------------------------------------------------- #

def make_dfa_G(pa):
  r"""Four-state automaton that detects three 1s in a row: a 0 (probability ``pa``) returns to state 0,
  a 1 advances ``0 -> 1 -> 2 -> 3``, and state 3 is absorbing."""
  pb = 1 - pa
  G = jnp.array([
    [pa, pb, 0.0, 0.0],
    [pa, 0.0, pb, 0.0],
    [pa, 0.0, 0.0, pb],
    [0.0, 0.0, 0.0, pa + pb],
  ]).T
  return G


def make_mbit_memory_G(m, p):
  r""" Shift register of the last ``m`` bits, a 1 arriving with probability ``p`` (least significant bit newest). """
  n = 2**m

  A0 = jnp.zeros((n, n))
  A1 = jnp.zeros((n, n))

  for x in range(n):
    # retain the last m-1 bits
    tail = x & (2**(m-1) - 1)

    next0 = (tail << 1) | 0
    next1 = (tail << 1) | 1

    A0 = A0.at[next0, x].set(1.0)
    A1 = A1.at[next1, x].set(1.0)
  return (1-p) * A0 + p * A1


def stationary_mbit(m, p):
  r""" Stationary distribution of ``make_mbit_memory_G``: ``m`` independent bits. """
  pi = jnp.array([1 - p, p])
  for _ in range(m - 1):
    pi = jnp.kron(pi, jnp.array([1 - p, p]))
  return pi


def last_bit_projection(m):
  r""" ``2 x 2**m`` matrix marginalizing the register onto its newest bit. """
  n = 2**m
  C = jnp.zeros((2, n))
  for x in range(n):
    bit = x & 1
    C = C.at[bit, x].set(1.0)
  return C


# --------------------------------------------------------------------------- #
# Physical implementation of writing a bit
# --------------------------------------------------------------------------- #

def make_memory_write_transition(m, b, Delta):
  r"""Edge data ``(rows, dE, E)`` for flipping the newest bit of an ``m``-bit register toward ``b``.

  The newest bit has energy ``0`` when it equals ``b`` and ``Delta`` otherwise; ``rows[0, x]`` is
  ``x`` with that bit flipped and ``dE[0, x]`` the energy change of the flip.
  """
  n = 2**m
  states = jnp.arange(n)

  # Flip least-significant bit = most recent memory bit
  rows = jnp.array([states ^ 1])  # shape (1, n)

  if b == 0:
    E_bit = jnp.array([0.0, Delta])
  else:
    E_bit = jnp.array([Delta, 0.0])

  last_bit = states & 1
  E = E_bit[last_bit]

  # energy change source -> flipped destination
  dE = E[rows] - E[None, :]      # shape (1, n)

  return rows, dE, E


def rates_from_dE(dE, beta, gamma=1.0):
  return gamma * jnp.exp(-0.5 * beta * dE)


def bit_write_generator(b, beta=1.0, Delta=5.0, gamma=1.0):
  r""" Generator ``K`` of a two-level bit relaxing toward ``b`` and its Gibbs state ``pi``. """
  # target b has low energy
  E = Delta * (jnp.arange(2) != b)

  dE = E[:, None] - E[None, :]

  K = gamma * jnp.exp(-0.5 * beta * dE)
  K = K.at[jnp.diag_indices(2)].set(0.0)

  # column-generator convention
  K = K.at[jnp.diag_indices(2)].set(-K.sum(axis=0))

  pi = jnp.exp(-beta * E)
  pi = pi / pi.sum()

  return K, pi


def cyclic_shift_matrix(m):
  r""" Rotate the ``m`` register bits so the oldest becomes the newest. """
  n = 2**m
  S = jnp.zeros((n, n))

  for x in range(n):
    first = (x >> (m - 1)) & 1
    tail = x & (2**(m - 1) - 1)

    y = (tail << 1) | first
    S = S.at[y, x].set(1.0)

  return S


def make_physical_mbit_branch(m, b, tau, beta=1.0, Delta=5.0, gamma=1.0):
  r"""One write-``b`` step of the register as a physical process of duration ``tau``.

  Returns ``(G_b, pi_b, A_cycle, c_cycle)``: the stochastic map of the step (write, then rotate),
  the bit's Gibbs state, and the heat-weight vectors from ``glauber.precompute_ep`` for the cycle.
  """
  Kb, pi_b = bit_write_generator(b, beta=beta, Delta=Delta, gamma=gamma)
  rows, dE, E = make_memory_write_transition(m, b, Delta)
  rates = rates_from_dE(dE, beta, 1.0)
  K = glauber.dense_generator_from_rates(rows, rates)

  # one bath,
  g = glauber.heat_rate_vectors(rates[None, :, :], dE)
  EM_dt, A = glauber.precompute_ep(K, g, tau)
  c = jnp.array([beta]) @ A

  Rb = jsp.expm(tau * Kb)
  S = cyclic_shift_matrix(m)

  G_b = jnp.kron(jnp.eye(2**(m-1)), Rb) @ S

  A_cycle = A[0] @ S
  c_cycle = c @ S
  return G_b, pi_b, A_cycle, c_cycle
