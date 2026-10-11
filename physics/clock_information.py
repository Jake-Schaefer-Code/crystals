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
cost rate), ``ring_generator`` and ``coherence_bound``, and the Ornstein-Uhlenbeck pair
``ou_trajectory`` / ``ou_contraction_rate`` (relative Fisher information).

Spectral and smoothing helpers: ``mode_filter_uniform`` (``g``) and ``mode_filter_exponential``
(``h``) give each relaxation mode's share of ``D(p_0 || pi)`` near equilibrium; ``heat_smooth`` is
Gaussian smoothing in time (the heat equation in ``t`` with reflecting ends, scale
``s = sigma^2 / 2``), which leaves ``qbar`` unchanged; ``time_fisher_information`` is ``J(t)``.

Five analyses return dataclasses that ``plotting.clock_information`` draws (this module does not
import it): ``ou_space_time``, ``ou_clock_decay``, ``ou_window_scan`` / ``ctmc_window_scan``,
``ou_blur_scan`` and ``ring_comparison``. Each checks the identity it relies on and raises
``AssertionError`` rather than return a number that violates it.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as onp
from jax import Array
from jax.scipy.fft import dct, idct
from jax.scipy.linalg import expm

from physics.markov import KL


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

def ctmc_trajectory(K: Array, p0: Array, ts: Array) -> Array:
  r""" ``P[i] = expm(K t_i) @ p0``. """
  return jax.vmap(lambda t: expm(K * t) @ p0)(jnp.asarray(ts))


def contraction_rate(K: Array, p: Array, q: Array) -> Array:
  r""" ``D_K(p || q) = -d/ds KL(e^{sK} p || e^{sK} q)`` at ``s = 0``; needs ``p, q`` of full support. """
  Kp, Kq = K @ p, K @ q
  return -jnp.sum(Kp * (jnp.log(p) - jnp.log(q))) + jnp.sum(p * Kq / q)


def clock_information_decay(K: Array, P: Array, w: Array, shifts: Array) -> Array:
  r""" ``I(tau; X_{tau+s})`` for each shift ``s``: the window's clock information after extra evolution. """
  return jax.vmap(lambda s: clock_information(P @ expm(K * s).T, w))(jnp.asarray(shifts))


def stationary_distribution(K: Array) -> onp.ndarray:
  r""" The normalized null vector of ``K`` (eigenvalue nearest zero), clipped to be nonnegative. """
  vals, vecs = onp.linalg.eig(onp.asarray(K))
  v = onp.real(vecs[:, onp.argmin(onp.abs(vals))])
  return onp.clip(v / v.sum(), 0.0, None)


def ring_generator(n: int, k_plus: float, k_minus: float) -> Array:
  r""" Uniform ring: ``x -> x+1`` at rate ``k_plus`` and ``x -> x-1`` at rate ``k_minus`` (mod ``n``). """
  idx = jnp.arange(n)
  K = jnp.zeros((n, n)).at[(idx + 1) % n, idx].add(k_plus).at[(idx - 1) % n, idx].add(k_minus)
  return K - jnp.diag(K.sum(axis=0))


def coherence_bound(n: int, k_plus: float, k_minus: float) -> float:
  r""" ``cot(pi/n) tanh(A/(2n))`` with affinity ``A = n log(k_plus/k_minus)``; exact for the uniform ring. """
  affinity = n * onp.log(k_plus / k_minus)
  return float(onp.tanh(affinity / (2 * n)) / onp.tan(onp.pi / n))


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

  ``info[k]`` and ``mmc[k]`` are for ``sigmas[k]``; ``log_ratios[j]`` is the smoothed log-ratio field
  at ``display_sigmas[j]`` on ``ts`` (midpoints) by ``xs``.
  """
  sigmas: onp.ndarray
  info: onp.ndarray
  mmc: onp.ndarray
  ts: onp.ndarray
  xs: onp.ndarray
  display_sigmas: onp.ndarray
  log_ratios: onp.ndarray


def ou_blur_scan(sigmas: Sequence[float], *, T: float = 3.0, n_t: int = 400, xs: Array, a: float = 1.0,
                 m0: float = 2.0, v0: float = 0.25, v_inf: float = 1.0,
                 display_sigmas: Sequence[float] = (0.0, 0.3, 0.8)) -> BlurScan:
  r""" Gaussian clock blur of an OU trajectory; checks that ``qbar`` is unchanged by the blur. """
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
  log_ratios = onp.stack([onp.asarray(log_ratio(jnp.maximum(blurred(s), 1e-300), w)) for s in display_sigmas])
  return BlurScan(onp.asarray(sigmas, dtype=float), onp.array(info), onp.array(mmc), onp.asarray(ts),
                  onp.asarray(xs), onp.asarray(display_sigmas, dtype=float), log_ratios)


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
