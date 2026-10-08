# physics/sk_theory.py
r"""Couplings and mean-field theory for the Sherrington-Kirkpatrick-type spin glass.

Couplings are ``J_ij = J0 / N + DeltaJ / sqrt(N) Z_ij`` with ``Z`` standard normal and no
self-coupling: ``J0`` is the ferromagnetic mean and ``DeltaJ`` the disorder. The mean magnetization
of the self-consistent closure used with these dynamics solves

  ``m = E_z tanh(beta (J0 m + DeltaJ z)),   z ~ N(0, 1)``,

by damped iteration from ``m = 1`` (``solve_m``), with the Gaussian average done by Gauss-Hermite
quadrature. Linearizing the same map at ``m = 0`` gives ``1 = beta J0 E_z sech^2(beta DeltaJ z)``,
whose root in ``DeltaJ`` is where the ordered solution disappears (``critical_DeltaJ``).

The dynamics are ``glauber.make_asymmetric_sk_graph`` and ``glauber.get_sk_rates``.

Moved from ``notebooks/stoch_thermo/ising_SK.ipynb``.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import brentq


def Jij_from_disorder(Z, J0: float, DeltaJ):
  r""" Couplings from a standard-normal matrix ``Z``, traceable under ``jit`` and ``grad``. """
  N = Z.shape[0]
  Jij = J0 / N + DeltaJ / jnp.sqrt(N) * Z
  return jnp.fill_diagonal(Jij, 0.0, inplace=False)


def make_Jij(Z, J0, DeltaJ):
  r""" Same couplings built on the host; the diagonal is zeroed. """
  N = Z.shape[0]

  Jij = J0 / N + DeltaJ / np.sqrt(N) * Z

  Jij = np.asarray(Jij).copy()
  np.fill_diagonal(Jij, 0.0)

  return jnp.asarray(Jij)


xgh, wgh = hermgauss(80)

zgh = np.sqrt(2.0) * xgh
wgh = wgh / np.sqrt(np.pi)
r""" Nodes and weights such that ``sum(wgh * f(zgh))`` is the standard-normal expectation of ``f``. """


def gaussian_expectation(f):
  return np.sum(wgh * f(zgh))


def solve_m(beta, DeltaJ, J0=1.0, tol=1e-12, maxiter=10000):
  # start positive to select the
  # positive broken-symmetry branch
  m = 1.0

  for _ in range(maxiter):
    m_new = gaussian_expectation(lambda z: np.tanh(beta * (J0 * m + DeltaJ * z)))
    if abs(m_new - m) < tol:
      return m_new

    m = 0.5 * m + 0.5 * m_new

  return m


def critical_DeltaJ(beta, J0=1.0, upper=2.0):
  r""" Disorder at which the ordered solution of ``solve_m`` vanishes; ``nan`` if ``beta J0 <= 1`` (never ordered). """
  if beta * J0 <= 1.0:
    return np.nan

  target = 1.0 / (beta * J0)

  def f(DeltaJ):
    rhs = gaussian_expectation(lambda z: 1.0 / np.cosh(beta * DeltaJ * z)**2)
    return rhs - target

  return brentq(f, 0.0, upper)
