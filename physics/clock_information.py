# physics/clock_information.py
r"""Clock information and the time-averaged mismatch cost of a time-homogeneous process.

Conventions follow ``physics.markov``: a column generator ``K`` with ``dp/dt = K @ p``, and a
trajectory ``P`` with time on axis 0 and states on axis 1 (``P[i] = p_{t_i}``). A window is a vector
``w`` of weights on the sample times summing to one; ``uniform_weights`` gives the trapezoid weights
of the uniform window on an endpoint-inclusive grid. Continuous state spaces are sampled on a grid
and carried as probability masses per cell, so ``markov.KL`` applies unchanged. Library code never
sets ``jax_enable_x64``; the identities checked below need float64, so scripts and tests set it.

Windows and clock information:

- ``time_average``: the time-averaged prior ``qbar = sum_i w_i p_{t_i}``.
- ``clock_information``: ``I(tau; X) = sum_i w_i KL(p_{t_i} || qbar)``, the ``w``-weighted
  Jensen-Shannon divergence (Sibson's information radius) of the trajectory.
- ``log_ratio``: ``log(p_t(x) / qbar(x))``, the pointwise mutual information between time and state.
- ``uniform_window_mmc``: minimal total mismatch cost over a uniform window by the boundary formula
  ``D(p_0 || qbar) - D(p_T || qbar)``.

Dynamics: ``ctmc_trajectory``, ``contraction_rate`` (``D_K(p || q)`` for a CTMC),
``clock_information_decay`` (``I(tau; X_{tau+s})``, whose slope at ``s = 0`` is minus the minimal
cost rate), ``ring_generator`` (with ``potential_ring_generator``) and ``coherence_bound``, and the Ornstein-Uhlenbeck pair
``ou_trajectory`` / ``ou_contraction_rate`` (relative Fisher information). The grid versions
differentiate sampled arrays by central differences; ``ou_log_density``, ``ou_time_fisher_exact`` and
``ou_contraction_rate_exact`` take the derivatives of the closed-form OU density by autodiff instead,
which removes the first-order error ``jnp.gradient`` makes at the ends of the time grid.

Spectral and smoothing helpers: ``mode_filter_uniform`` (``g``) and ``mode_filter_exponential``
(``h``) give each relaxation mode's share of ``D(p_0 || pi)`` near equilibrium; ``heat_smooth`` is
Gaussian smoothing in time (the heat equation in ``t`` with reflecting ends, scale
``s = sigma^2 / 2``), which leaves ``qbar`` unchanged; ``time_fisher_information`` is ``J(t)``;
``time_harmonic`` is the amplitude ``a_k(x)`` of each cosine mode in time, which blurring damps by
``exp(-(k pi / T)^2 s)``, and ``nodal_cross`` uses ``a_1`` and ``a_2`` to predict the heavy-blur nodal
set of ``p^sigma_t = qbar``: the lines ``t = T/2`` and ``x = x*`` (``a_1(x*) = 0``) opened into hyperbolas.

The dynamical Hamiltonian ``H_dyn = D^dagger D``, ``D = d_t - K``, is the operator of the least-squares
form ``int_0^T ||d_t p - K p||^2 dt`` in the ``1/pi`` inner product (``dynamical_hamiltonian``). Its zero
level is the set of solutions of the master equation, one per initial condition. For a reversible ``K``,
``H_dyn = -d_t^2 + K^2`` separates and its other levels are ``(k pi / T)^2 + lambda_j^2`` (``normal_levels``);
a driven ``K`` couples ``t`` and ``x`` through ``(K - K^dagger) d_t``. ``crossing_gap`` measures whether
the nodal lines of an eigenfunction cross or have reconnected.

Six analyses return dataclasses that ``plotting.clock_information`` draws (this module does not
import it): ``ou_space_time``, ``ou_clock_decay``, ``ou_window_scan`` / ``ctmc_window_scan``,
``ou_blur_scan``, ``ring_comparison`` and ``ring_hamiltonian``. Each checks the identity it relies on and
raises ``AssertionError`` rather than return a number that violates it.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as onp
from jax import Array
from jax.scipy.fft import dct, idct
from jax.scipy.linalg import expm
from jax.scipy.special import logsumexp
from scipy.signal import resample
from physics.markov import KL, ctmc_trajectory, contraction_rate, stationary_distribution


# --------------------------------------------------------------------------- #
# Windows and clock information
# --------------------------------------------------------------------------- #

def uniform_weights(n: int) -> Array:
  r""" Trapezoid weights of the uniform window on ``n`` equally spaced times, endpoints included. """
  w = jnp.ones(n).at[0].set(0.5).at[-1].set(0.5)
  return w / w.sum()


def time_average(P: Array, w: Array) -> Array:
  r""" ``qbar = sum_i w_i P[i]``, the optimal fixed prior for the window ``w``. """
  return w @ P


def clock_information(P: Array, w: Array) -> Array:
  r""" ``I(tau; X) = sum_i w_i KL(P[i] || qbar)``; zero exactly when ``P`` is constant on the window. """
  return jnp.sum(w * KL(P, time_average(P, w), axis=1))


def log_ratio(P: Array, w: Array) -> Array:
  r""" ``log(P[i, x] / qbar[x])``: positive where state ``x`` is evidence for time ``t_i``. """
  q = time_average(P, w)
  return jnp.log(P) - jnp.log(q)


def uniform_window_mmc(P: Array) -> Array:
  r""" Minimal total mismatch cost ``D(p_0 || qbar) - D(p_T || qbar)`` for the uniform window over ``P``. """
  q = time_average(P, uniform_weights(P.shape[0]))
  return KL(P[0], q) - KL(P[-1], q)


# --------------------------------------------------------------------------- #
# Finite state spaces
# --------------------------------------------------------------------------- #


def clock_information_decay(K: Array, P: Array, w: Array, shifts: Array) -> Array:
  r""" ``I(tau; X_{tau+s})`` for each shift ``s``: the window's clock information after extra evolution. """
  return jax.vmap(lambda s: clock_information(P @ expm(K * s).T, w))(jnp.asarray(shifts))



def ring_generator(n: int, k_plus: float, k_minus: float) -> Array:
  r""" Uniform ring: ``x -> x+1`` at rate ``k_plus`` and ``x -> x-1`` at rate ``k_minus`` (mod ``n``). """
  idx = jnp.arange(n)
  K = jnp.zeros((n, n)).at[(idx + 1) % n, idx].add(k_plus).at[(idx - 1) % n, idx].add(k_minus)
  return K - jnp.diag(K.sum(axis=0))


def coherence_bound(n: int, k_plus: float, k_minus: float) -> float:
  r""" ``cot(pi/n) tanh(A/(2n))`` with affinity ``A = n log(k_plus/k_minus)``; exact for the uniform ring. """
  affinity = n * onp.log(k_plus / k_minus)
  return float(onp.tanh(affinity / (2 * n)) / onp.tan(onp.pi / n))


def potential_ring_generator(n: int, k_plus: float, k_minus: float, depth: float) -> Array:
  r""" Ring generator with rates out of ``x`` scaled by ``exp(-(V(x +- 1) - V(x)) / 2)``, ``V = depth cos(2 pi x / n)``.

  Local detailed balance holds with Boltzmann factor ``e^{-V}``: ``k_plus = k_minus`` is reversible with ``pi ~ e^{-V}``
  and ``k_plus != k_minus`` drives a current. ``depth = 0`` is ``ring_generator``.
  """
  V = depth * jnp.cos(2 * jnp.pi * jnp.arange(n) / n)
  return ring_generator(n, k_plus * jnp.exp(-0.5 * (jnp.roll(V, -1) - V)), k_minus * jnp.exp(-0.5 * (jnp.roll(V, 1) - V)))


# --------------------------------------------------------------------------- #
# Ornstein-Uhlenbeck on a grid
# --------------------------------------------------------------------------- #

def ou_trajectory(ts: Array, xs: Array, *, a: float, m0: float, v0: float, v_inf: float = 1.0) -> Array:
  r""" Gaussian OU solutions ``N(m0 e^{-at}, v_inf + (v0 - v_inf) e^{-2at})`` as masses on the grid ``xs``. """
  t = jnp.asarray(ts)[:, None]
  x = jnp.asarray(xs)[None, :]
  m = m0 * jnp.exp(-a * t)
  v = v_inf + (v0 - v_inf) * jnp.exp(-2 * a * t)
  dens = jnp.exp(-(x - m) ** 2 / (2 * v))
  return dens / dens.sum(axis=1, keepdims=True)


def ou_kl_to_stationary(*, m0: float, v0: float, v_inf: float = 1.0) -> float:
  r""" Closed-form ``D(N(m0, v0) || N(0, v_inf))``. """
  return 0.5 * (onp.log(v_inf / v0) + (v0 + m0 ** 2) / v_inf - 1.0)


def ou_contraction_rate(P: Array, q: Array, xs: Array, *, a: float, v_inf: float = 1.0, floor: float = 1e-12) -> Array:
  r""" ``D_K(p || q) = D int p (d/dx log(p/q))^2`` with diffusion ``D = a v_inf``, per row of ``P``. """
  dx = xs[1] - xs[0]
  P_safe = jnp.maximum(P, floor)
  grad = jnp.gradient(jnp.log(P_safe) - jnp.log(jnp.maximum(q, floor)), dx, axis=-1)
  return a * v_inf * jnp.sum(jnp.where(P > floor, P * grad ** 2, 0.0), axis=-1)


def ou_log_density(t, x, *, a: float, m0: float, v0: float, v_inf: float = 1.0):
  r""" Closed-form log-density of the OU solution at scalar time ``t`` and point ``x``. """
  m = m0 * jnp.exp(-a * t)
  v = v_inf + (v0 - v_inf) * jnp.exp(-2 * a * t)
  return -0.5 * (x - m) ** 2 / v - 0.5 * jnp.log(2 * jnp.pi * v)


def ou_time_fisher_exact(ts: Array, xs: Array, *, a: float, m0: float, v0: float, v_inf: float = 1.0) -> Array:
  r""" ``J(t) = sum_x p_t(x) (d_t log p_t(x))^2``: time derivative by autodiff, sum over grid masses. """
  logp = partial(ou_log_density, a=a, m0=m0, v0=v0, v_inf=v_inf)
  dlogp_dt = jax.vmap(jax.grad(logp, argnums=0), (None, 0))

  def J(t):
    lp = jax.vmap(lambda x: logp(t, x))(xs)
    p = jnp.exp(lp - logsumexp(lp))                          # masses on the grid
    return jnp.sum(p * dlogp_dt(t, xs) ** 2)

  return jax.vmap(J)(jnp.asarray(ts))


def ou_contraction_rate_exact(ts: Array, xs: Array, w: Array, *, a: float, m0: float, v0: float,
                              v_inf: float = 1.0) -> Array:
  r""" ``D_K(p_t || qbar)`` for each ``t`` in ``ts``, ``qbar = sum_i w_i p_{t_i}``, scores by autodiff.

  ``log qbar`` is the ``logsumexp`` of the window mixture with the same weights as ``time_average``,
  so it agrees with the grid version and stays finite in the tails.
  """
  logp = partial(ou_log_density, a=a, m0=m0, v0=v0, v_inf=v_inf)
  log_q = lambda x: logsumexp(jax.vmap(lambda s: logp(s, x))(ts), b=w)

  def rate(t):
    lp = jax.vmap(lambda x: logp(t, x))(xs)
    p = jnp.exp(lp - logsumexp(lp))
    score = jax.vmap(jax.grad(lambda x: logp(t, x) - log_q(x)))(xs)
    return a * v_inf * jnp.sum(p * score ** 2)

  return jax.vmap(rate)(jnp.asarray(ts))


# --------------------------------------------------------------------------- #
# Spectral filters and temporal smoothing
# --------------------------------------------------------------------------- #

def mode_filter_uniform(x: Array) -> Array:
  r""" ``g(x) = (1 - e^{-2x})/2 - (1 - e^{-x})^2 / x``: a mode with ``|lambda| T = x`` contributes ``|c|^2 g``. """
  x = jnp.asarray(x, dtype=float)
  safe = jnp.where(x > 1e-3, x, 1.0)
  full = 0.5 * (1.0 - jnp.exp(-2.0 * safe)) - (1.0 - jnp.exp(-safe)) ** 2 / safe
  return jnp.where(x > 1e-3, full, x ** 3 / 12.0)


def mode_filter_exponential(y: Array) -> Array:
  r""" ``h(y) = y^3 / ((1 + 2y)(1 + y)^2)``: the same share for an exponential window, ``y = |lambda| / rate``. """
  y = jnp.asarray(y, dtype=float)
  return y ** 3 / ((1.0 + 2.0 * y) * (1.0 + y) ** 2)


def midpoint_times(T: float, n: int) -> Array:
  r""" ``t_i = (i + 1/2) T / n``, the grid on which ``heat_smooth`` has reflecting ends. """
  return (jnp.arange(n) + 0.5) * T / n


def heat_smooth(P: Array, s: float, T: float) -> Array:
  r""" Heat flow ``d_s P = d_t^2 P`` for time ``s`` with reflecting ends (Gaussian blur of std ``sqrt(2s)``).

  ``P`` must be sampled on ``midpoint_times(T, n)``; the flow is diagonal in the type-II cosine
  basis, and it preserves the uniform time average ``P.mean(axis=0)`` exactly.
  """
  k = jnp.arange(P.shape[0])
  decay = jnp.exp(-((k * jnp.pi / T) ** 2) * s)
  return idct(dct(P, type=2, axis=0, norm="ortho") * decay[:, None], type=2, axis=0, norm="ortho")


def time_fisher_information(P: Array, ts: Array, *, floor: float = 1e-12) -> Array:
  r""" ``J(t_i) = sum_x (d_t P)^2 / P``, the Fisher information about time, by central differences. """
  dP = jnp.gradient(P, jnp.asarray(ts), axis=0)
  return jnp.sum(jnp.where(P > floor, dP ** 2 / jnp.maximum(P, floor), 0.0), axis=1)


def time_harmonic(P: Array, k: int) -> Array:
  r""" ``a_k(x) = (2/n) sum_i P[i, x] cos(k pi (i + 1/2) / n)``: amplitude of the ``k``-th cosine mode in time.

  For ``P`` on ``midpoint_times(T, n)``, ``P = qbar + sum_{k>=1} a_k(x) cos(k pi t / T)`` exactly, and
  ``heat_smooth(P, s, T)`` multiplies ``a_k`` by ``exp(-(k pi / T)^2 s)``.
  """
  n = P.shape[0]
  return (2.0 / n) * jnp.cos(k * jnp.pi * (jnp.arange(n) + 0.5) / n) @ P


@dataclass(frozen=True)
class NodalCross:
  r""" Heavy-blur limit of the nodal set ``p^sigma_t(x) = qbar(x)``, see ``nodal_cross``.

  ``a1`` and ``a2`` are ``time_harmonic(P, 1)`` and ``time_harmonic(P, 2)`` on ``xs``; ``x_star`` holds the
  zeros of ``a1`` and ``c`` the hyperbola constant ``-a_2(x*) T / (pi a_1'(x*))`` at each of them.
  """
  T: float
  xs: onp.ndarray
  a1: onp.ndarray
  a2: onp.ndarray
  x_star: onp.ndarray
  c: onp.ndarray

  def product(self, sigma: float) -> onp.ndarray:
    r""" ``c exp(-3 pi^2 sigma^2 / (2 T^2))``: ``(t - T/2)(x - x*)`` on the nodal set near each crossing. """
    return self.c * onp.exp(-1.5 * (onp.pi * sigma / self.T) ** 2)


def nodal_cross(P: Array, xs: Array, T: float) -> NodalCross:
  r""" Where the nodal set of the clock-blurred field goes as ``sigma`` grows; ``P`` is on ``midpoint_times(T, n)``.

  Blurring by ``sigma`` damps harmonic ``k`` of ``p_t(x) - qbar(x)`` by ``exp(-(k pi / T)^2 sigma^2 / 2)``, so
  the slowest wins: ``p^sigma_t(x) - qbar(x) ~ a_1(x) cos(pi t / T)``, a function of ``t`` times a function
  of ``x``. Its zero set is the line ``t = T/2`` (the window's reflection symmetry) crossed with the lines
  ``x = x*``, ``a_1(x*) = 0``. Keeping ``k = 2`` and expanding about ``(T/2, x*)`` opens each crossing into
  the hyperbola ``(t - T/2)(x - x*) = c exp(-3 pi^2 sigma^2 / (2 T^2))``, ``c = -a_2(x*) T / (pi a_1'(x*))``;
  the sign of ``c`` picks which pair of opposite quadrants the branches occupy. The prediction is asymptotic
  in ``sigma``, accurate once the ``k >= 3`` harmonics are negligible against ``k = 2``.
  """
  xs = onp.asarray(xs)
  a1, a2 = onp.asarray(time_harmonic(P, 1)), onp.asarray(time_harmonic(P, 2))
  sign = onp.sign(a1)
  j = onp.flatnonzero(sign[:-1] * sign[1:] < 0)
  x_star = xs[j] - a1[j] * (xs[j + 1] - xs[j]) / (a1[j + 1] - a1[j])
  slope = onp.interp(x_star, xs, onp.gradient(a1, xs, edge_order=2))
  c = -onp.interp(x_star, xs, a2) * T / (onp.pi * slope)
  return NodalCross(float(T), xs, a1, a2, x_star, c)


# --------------------------------------------------------------------------- #
# The dynamical Hamiltonian
# --------------------------------------------------------------------------- #

def dynamical_hamiltonian(K: Array, T: float, n_t: int, pi: Array | None = None) -> tuple[Array, Array]:
  r""" Matrix of ``H_dyn = D^dagger D``, ``D = d_t - K``, on ``n_t`` times in ``[0, T]``, and the scale of its fields.

  ``H_dyn`` is the operator of the quadratic form ``Q[p] = int_0^T ||d_t p - K p||^2 dt`` with
  ``||u||^2 = sum_x u_x^2 / pi_x``, the inner product in which a reversible ``K`` is self-adjoint (``pi``
  defaults to the stationary state). The ends are free, so the natural boundary condition is ``d_t p = K p``
  at ``t = 0`` and ``T``: the zero-energy states are exactly the solutions of the master equation, one per
  initial condition, and the zero level is ``n``-fold degenerate. Each time interval contributes its midpoint
  residual ``(p_{i+1} - p_i) / h - K (p_i + p_{i+1}) / 2`` and the norm has trapezoid weights ``w_i``, so
  levels carry an ``O(h^2)`` error. Returns ``(H, scale)``: ``H`` is symmetric and positive semidefinite, its
  eigenvalues are the energies, and an eigenvector ``phi`` is the field ``p = phi.reshape(n_t, n) * scale``.
  Needs float64 for a sharp zero level.
  """
  K = jnp.asarray(K)
  n = K.shape[0]
  pi = jnp.asarray(stationary_distribution(K) if pi is None else pi)
  Kq = K * jnp.sqrt(pi)[None, :] / jnp.sqrt(pi)[:, None]                  # K on q = p / sqrt(pi), Euclidean norm
  h = T / (n_t - 1)
  forward = jnp.eye(n_t - 1, n_t, 1)
  d_t = (forward - jnp.eye(n_t - 1, n_t)) / h
  mid = (forward + jnp.eye(n_t - 1, n_t)) / 2
  B = jnp.kron(d_t, jnp.eye(n)) - jnp.kron(mid, Kq)
  w = jnp.full(n_t, h).at[0].set(0.5 * h).at[-1].set(0.5 * h)
  s = jnp.kron(jnp.sqrt(w), jnp.ones(n))
  return h * B.T @ B / s[:, None] / s[None, :], jnp.sqrt(pi)[None, :] / jnp.sqrt(w)[:, None]


def normal_levels(K: Array, T: float, n_levels: int) -> onp.ndarray:
  r""" Lowest ``n_levels`` of ``H_dyn`` if ``K`` is normal in the ``1/pi`` inner product (reversible, or a uniform ring).

  Each eigenvalue ``mu_j`` of ``K`` gives the zero mode ``e^{mu_j t}`` and the tower ``(k pi / T)^2 + (Re mu_j)^2``,
  ``k >= 1``. With free ends the time profile of a tower state is ``cos(k pi t / T + delta)``,
  ``tan(delta) = -Re(mu_j) T / (k pi)``, so the ``k = 0`` level of the Neumann problem (``(Re mu_j)^2``) is
  replaced by the zero mode. ``Im mu_j`` is a gauge phase ``e^{i Im(mu_j) t}`` and moves no level. For a
  non-normal ``K`` the formula fails.
  """
  mu = onp.linalg.eigvals(onp.asarray(K))
  towers = [(onp.pi * k / T) ** 2 + mu.real ** 2 for k in range(1, n_levels + 1)]
  return onp.sort(onp.concatenate([onp.zeros(len(mu)), *towers]))[:n_levels]


def ring_interpolate(F: onp.ndarray, up: int = 12) -> onp.ndarray:
  r""" Trigonometric interpolation of ``F[t, x]`` along its periodic site axis onto ``up`` times as many points. """
  return resample(F, F.shape[1] * up, axis=1)


def crossing_gap(F: onp.ndarray, ts: onp.ndarray, up: int = 12) -> float:
  r""" Smallest ``|F|`` at a saddle of ``F(t, x)`` in units of ``max |F|``: zero if nodal lines cross, positive if avoided.

  Level sets of a smooth function cross only at a saddle whose value is exactly zero; where the lines have
  reconnected into two hyperbola branches the saddle sits at ``+-epsilon``. ``F`` has periodic ``x`` and is
  interpolated by ``ring_interpolate``. Saddles are the grid points where ``|grad F|`` is locally smallest
  with ``det Hess F < 0``, moved to the critical point by one Newton step (``nan`` if there is none).
  """
  G = ring_interpolate(F, up)
  dt, dx = ts[1] - ts[0], 1.0 / up
  Ft = onp.gradient(G, dt, axis=0)
  Fx = (onp.roll(G, -1, 1) - onp.roll(G, 1, 1)) / (2 * dx)
  Ftt = onp.gradient(Ft, dt, axis=0)
  Fxx = (onp.roll(G, -1, 1) - 2 * G + onp.roll(G, 1, 1)) / dx ** 2
  Ftx = (onp.roll(Ft, -1, 1) - onp.roll(Ft, 1, 1)) / (2 * dx)
  g2 = Ft ** 2 + Fx ** 2
  near = [onp.roll(onp.roll(g2, i, 0), j, 1) for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0)]
  saddle = onp.all([g2 < other for other in near], axis=0) & (Ftt * Fxx - Ftx ** 2 < 0)
  saddle[[0, -1], :] = False
  values = []
  for i, j in zip(*onp.nonzero(saddle)):
    grad, hess = onp.array([Ft[i, j], Fx[i, j]]), onp.array([[Ftt[i, j], Ftx[i, j]], [Ftx[i, j], Fxx[i, j]]])
    step = onp.linalg.solve(hess, grad)
    if abs(step[0]) <= dt and abs(step[1]) <= dx:
      values.append(abs(G[i, j] - 0.5 * grad @ step))
  return float(min(values) / onp.abs(G).max()) if values else float("nan")


# --------------------------------------------------------------------------- #
# Analyses
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SpaceTimeWindow:
  r""" OU trajectory over a uniform window: masses ``P[t, x]``, ``qbar``, the log-ratio field and two totals. """
  ts: onp.ndarray
  xs: onp.ndarray
  P: onp.ndarray
  qbar: onp.ndarray
  log_ratio: onp.ndarray
  info: float          # I(tau; X)
  mmc: float           # minimal total mismatch cost D(p_0||qbar) - D(p_T||qbar)


def ou_space_time(*, T: float = 3.0, n_t: int = 301, xs: Array, a: float = 1.0, m0: float = 2.0,
                  v0: float = 0.25, v_inf: float = 1.0) -> SpaceTimeWindow:
  r""" The space-time density of an OU relaxation and its log-ratio to the time average. """
  ts = jnp.linspace(0.0, T, n_t)
  P = ou_trajectory(ts, xs, a=a, m0=m0, v0=v0, v_inf=v_inf)
  w = uniform_weights(n_t)
  return SpaceTimeWindow(
    onp.asarray(ts), onp.asarray(xs), onp.asarray(P), onp.asarray(time_average(P, w)),
    onp.asarray(log_ratio(P, w)), float(clock_information(P, w)), float(uniform_window_mmc(P)),
  )


@dataclass(frozen=True)
class ClockDecay:
  r""" ``I(tau; X_{tau+s})`` against ``s``, and the minimal cost rate (minus its slope at ``s = 0``). """
  shifts: onp.ndarray
  info: onp.ndarray
  rate: float          # (D(p_0||qbar) - D(p_T||qbar)) / T
  fd_rate: float       # -(I(h) - I(0)) / h, the same number by finite differences


def ou_clock_decay(shifts: Sequence[float], *, T: float = 3.0, n_t: int = 301, xs: Array, a: float = 1.0,
                   m0: float = 2.0, v0: float = 0.25, v_inf: float = 1.0, h: float = 1e-4) -> ClockDecay:
  r""" Clock information erased by the homogeneous dynamics; checks ``-T dI/ds = D(p_0||qbar) - D(p_T||qbar)``. """
  ts = jnp.linspace(0.0, T, n_t)
  w = uniform_weights(n_t)
  info_at = lambda s: clock_information(ou_trajectory(ts + s, xs, a=a, m0=m0, v0=v0, v_inf=v_inf), w)
  rate = float(uniform_window_mmc(ou_trajectory(ts, xs, a=a, m0=m0, v0=v0, v_inf=v_inf))) / T
  fd_rate = -float(info_at(h) - info_at(0.0)) / h
  assert abs(fd_rate - rate) <= 1e-2 * abs(rate), (fd_rate, rate)
  info = onp.array([float(info_at(s)) for s in shifts])
  return ClockDecay(onp.asarray(shifts, dtype=float), info, rate, fd_rate)


@dataclass(frozen=True)
class WindowScan:
  r""" Minimal total mismatch cost and clock information for windows ``[0, T]``, with the budget ``D(p_0||pi)``. """
  Ts: onp.ndarray
  mmc: onp.ndarray
  info: onp.ndarray
  budget: float


def ou_window_scan(Ts: Sequence[float], *, xs: Array, a: float = 1.0, m0: float = 2.0, v0: float = 0.25,
                   v_inf: float = 1.0, n_t: int = 1201) -> WindowScan:
  r""" ``MC*_T`` against ``T`` for an OU relaxation; it rises to ``D(p_0||pi)`` from below.

  Every window uses ``n_t`` samples, so the jitted pieces compile once for the whole scan.
  """
  budget = ou_kl_to_stationary(m0=m0, v0=v0, v_inf=v_inf)
  mmc, info = [], []
  for T in Ts:
    ts = jnp.linspace(0.0, T, n_t)
    P = ou_trajectory(ts, xs, a=a, m0=m0, v0=v0, v_inf=v_inf)
    mmc.append(float(uniform_window_mmc(P)))
    info.append(float(clock_information(P, uniform_weights(n_t))))
  mmc = onp.array(mmc)
  assert onp.all(mmc <= budget + 1e-6), "minimal cost exceeded D(p0||pi), the cost with prior pi"
  return WindowScan(onp.asarray(Ts, dtype=float), mmc, onp.array(info), float(budget))


def ctmc_window_scan(K: Array, p0: Array, Ts: Sequence[float], *, n_t: int = 1201) -> WindowScan:
  r""" ``MC*_T`` against ``T`` for a CTMC, with budget ``D(p_0||pi)``. """
  budget = float(KL(p0, jnp.asarray(stationary_distribution(K))))
  mmc, info = [], []
  for T in Ts:
    ts = jnp.linspace(0.0, T, n_t)
    P = ctmc_trajectory(K, p0, ts)
    mmc.append(float(uniform_window_mmc(P)))
    info.append(float(clock_information(P, uniform_weights(n_t))))
  mmc = onp.array(mmc)
  assert onp.all(mmc <= budget + 1e-6), "minimal cost exceeded D(p0||pi), the cost with prior pi"
  return WindowScan(onp.asarray(Ts, dtype=float), mmc, onp.array(info), budget)


@dataclass(frozen=True)
class BlurScan:
  r""" Clock information and time-averaged minimal cost rate after blurring time by ``sigma``.

  ``info[k]`` and ``mmc[k]`` are for ``sigmas[k]``; ``log_ratios[j]`` and ``display_P[j]`` are the
  smoothed log-ratio field and masses at ``display_sigmas[j]`` on ``ts`` (midpoints) by ``xs``; ``cross``
  is the heavy-blur prediction for their nodal set.
  """
  sigmas: onp.ndarray
  info: onp.ndarray
  mmc: onp.ndarray
  ts: onp.ndarray
  xs: onp.ndarray
  display_sigmas: onp.ndarray
  log_ratios: onp.ndarray
  display_P: onp.ndarray
  cross: NodalCross


def ou_blur_scan(sigmas: Sequence[float], *, T: float = 3.0, n_t: int = 400, xs: Array, a: float = 1.0,
                 m0: float = 2.0, v0: float = 0.25, v_inf: float = 1.0,
                 display_sigmas: Sequence[float] = (0.0, 0.3, 0.8)) -> BlurScan:
  r""" Gaussian clock blur of an OU trajectory; checks that ``qbar`` is unchanged by the blur and that the
  displayed fields have their first time harmonic damped by exactly ``exp(-(pi sigma / T)^2 / 2)``. """
  ts = midpoint_times(T, n_t)
  P = ou_trajectory(ts, xs, a=a, m0=m0, v0=v0, v_inf=v_inf)
  q = P.mean(axis=0)

  def blurred(sigma):
    Ps = jnp.clip(heat_smooth(P, 0.5 * sigma ** 2, T), 0.0, None)
    return Ps / Ps.sum(axis=1, keepdims=True)

  w = jnp.full(n_t, 1.0 / n_t)
  info, mmc = [], []
  for sigma in sigmas:
    Ps = blurred(sigma)
    assert float(jnp.max(jnp.abs(Ps.mean(axis=0) - q))) < 1e-8, "the blur moved qbar"
    info.append(float(clock_information(Ps, w)))
    mmc.append(float(jnp.mean(ou_contraction_rate(Ps, q, xs, a=a, v_inf=v_inf))))
  display_P = onp.stack([onp.asarray(blurred(s)) for s in display_sigmas])
  log_ratios = onp.stack([onp.asarray(log_ratio(jnp.maximum(Ps, 1e-300), w)) for Ps in display_P])
  cross = nodal_cross(P, xs, T)
  for sigma, Ps in zip(display_sigmas, display_P):
    damped = cross.a1 * onp.exp(-0.5 * (onp.pi * sigma / T) ** 2)
    assert onp.allclose(time_harmonic(jnp.asarray(Ps), 1), damped, rtol=0.0, atol=1e-9 * onp.abs(cross.a1).max()), \
      "the blur did not damp the first time harmonic by exp(-(pi sigma / T)^2 / 2)"
  return BlurScan(onp.asarray(sigmas, dtype=float), onp.array(info), onp.array(mmc), onp.asarray(ts),
                  onp.asarray(xs), onp.asarray(display_sigmas, dtype=float), log_ratios, display_P, cross)


@dataclass(frozen=True)
class RingComparison:
  r""" A driven ring and its reversible twin (same escape rate and stationary state), started at state 0.

  Row 0 is driven, row 1 reversible. ``mode`` is the first Fourier amplitude
  ``sum_x p_t(x) e^{-2 pi i x / n}`` (1 at the start, 0 at ``pi``); ``slowest`` are the slowest
  nonzero eigenvalues.
  """
  ts: onp.ndarray
  start_prob: onp.ndarray
  mode: onp.ndarray
  slowest: onp.ndarray
  coherence: float
  scans: tuple[WindowScan, WindowScan]


def ring_comparison(n: int = 10, k_plus: float = 1.0, k_minus: float = 0.01, *, t_max: float = 30.0,
                    n_t: int = 301, Ts: Sequence[float] = (1, 2, 4, 6, 8, 10, 15, 20, 30, 40, 60)) -> RingComparison:
  r""" Ringing versus overdamped relaxation; checks the slowest mode sits on the coherence bound. """
  k_rev = 0.5 * (k_plus + k_minus)
  Ks = (ring_generator(n, k_plus, k_minus), ring_generator(n, k_rev, k_rev))
  p0 = jnp.zeros(n).at[0].set(1.0)
  ts = jnp.linspace(0.0, t_max, n_t)
  phase = jnp.exp(-2j * jnp.pi * jnp.arange(n) / n)
  start, mode, slowest = [], [], []
  for K in Ks:
    P = ctmc_trajectory(K, p0, ts)
    start.append(onp.asarray(P[:, 0]))
    mode.append(onp.asarray(P @ phase))
    vals = onp.linalg.eigvals(onp.asarray(K))
    vals = vals[onp.argsort(-vals.real)]
    slowest.append(vals[1] if vals[1].imag >= 0 else onp.conj(vals[1]))
  coherence = coherence_bound(n, k_plus, k_minus)
  assert onp.isclose(abs(slowest[0].imag / slowest[0].real), coherence, rtol=1e-6)
  assert onp.isclose(slowest[0].real, slowest[1].real, rtol=1e-6) and abs(slowest[1].imag) < 1e-9
  scans = tuple(ctmc_window_scan(K, p0, Ts) for K in Ks)
  return RingComparison(onp.asarray(ts), onp.stack(start), onp.stack(mode), onp.array(slowest), coherence, scans)


@dataclass(frozen=True)
class HamiltonianSpectrum:
  r""" Lowest levels of ``H_dyn`` for one ring generator, and its first eigenfunction with nodes in the state.

  ``state`` is ``p(t, x)`` of the lowest nonzero level that changes sign along ``x`` (``max |p| = 1``), the level's
  response to a source at ``t = 0``, which fixes its phase inside a degenerate level. ``rank_residual`` is the
  ratio ``s_2 / s_1`` of its singular values: zero for a product of a function of ``t`` and one of ``x``.
  ``trajectory`` is the Rayleigh quotient of the exact trajectory from site 0, zero up to ``O(h^2)``.
  """
  label: str
  energies: onp.ndarray
  normal: onp.ndarray       # normal_levels for the same generator
  level: float
  state: onp.ndarray
  rank_residual: float
  gap: float                # crossing_gap(state)
  trajectory: float


def _hamiltonian_spectrum(label: str, K: Array, ts: Array, n_levels: int) -> HamiltonianSpectrum:
  n, n_t, T = K.shape[0], len(ts), float(ts[-1])
  H, scale = map(onp.asarray, dynamical_hamiltonian(K, T, n_t))
  w, V = onp.linalg.eigh(H)
  assert onp.max(onp.abs(w[:n])) < 1e-6 * w[n], "the zero level of H_dyn is not n-fold"
  i = n
  while i < len(w):
    j = i + 1
    while j < len(w) and abs(w[j] - w[i]) < 1e-6 * w[i]:
      j += 1
    p = (V[:, i:j] * scale.reshape(-1, 1)).reshape(n_t, n, j - i)
    x0 = int(onp.argmax((p[0] ** 2).sum(axis=-1) >= (1 - 1e-6) * (p[0] ** 2).sum(axis=-1).max()))
    state = p @ p[0, x0]
    state = state / onp.abs(state).max()
    if state[0].max() > 1e-3 and state[0].min() < -1e-3:
      break
    i = j
  assert i < len(w), "no level of H_dyn changes sign along x"
  sv = onp.linalg.svd(state, compute_uv=False)
  phi = (onp.asarray(ctmc_trajectory(K, jnp.zeros(n).at[0].set(1.0), ts)) / scale).ravel()
  return HamiltonianSpectrum(label, w[:n_levels], normal_levels(K, T, n_levels), float(w[i]), state, float(sv[1] / sv[0]),
                             crossing_gap(state, onp.asarray(ts)), float(phi @ H @ phi / (phi @ phi)))


@dataclass(frozen=True)
class RingHamiltonian:
  r""" ``H_dyn`` for four rings on ``ts``: uniform reversible, uniform driven, reversible in a potential, driven in it. """
  ts: onp.ndarray
  systems: tuple[HamiltonianSpectrum, HamiltonianSpectrum, HamiltonianSpectrum, HamiltonianSpectrum]


def ring_hamiltonian(n: int = 10, k_plus: float = 1.0, k_minus: float = 0.01, *, T: float = 8.0, n_t: int = 161,
                     depth: float = 3.0, n_levels: int = 30) -> RingHamiltonian:
  r""" ``H_dyn`` on a uniform ring and on the ring in the potential ``depth cos(2 pi x / n)``, each reversible and driven.

  The driven rings have rates ``k_plus`` and ``k_minus`` (``1`` and ``0.01`` is the ring of ``ring_comparison``), the
  reversible ones ``(k_plus + k_minus) / 2`` both ways. Every ring has an ``n``-fold zero level, checked. A reversible
  ring separates: its eigenfunctions are products, their nodal lines cross, and the levels are ``normal_levels``,
  also in the potential, where only the nodes move. A uniform driven ring is normal, so the levels are the same and
  the eigenfunctions are waves ``g(t) cos(2 pi m x / n + omega t)`` with tilted nodal lines that still cross. In
  the potential the driven ring is not normal: the levels leave ``normal_levels`` and the crossings are avoided.
  """
  ts = jnp.linspace(0.0, T, n_t)
  k_rev = 0.5 * (k_plus + k_minus)
  rings = (("uniform, reversible", k_rev, k_rev, 0.0), ("uniform, driven", k_plus, k_minus, 0.0),
           ("potential, reversible", k_rev, k_rev, depth), ("potential, driven", k_plus, k_minus, depth))
  return RingHamiltonian(onp.asarray(ts), tuple(
    _hamiltonian_spectrum(label, potential_ring_generator(n, kp, km, d), ts, n_levels) for label, kp, km, d in rings))
