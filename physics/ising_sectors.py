# physics/ising_sectors.py
r"""Exact Curie-Weiss dynamics lumped by magnetization, and the periodic-optimal-prior (POP) analyses.

The reduced state ``k = 0, ..., n_spins`` is the number of up spins. For an exchangeable
distribution this reduction preserves every KL divergence exactly, and microstate Shannon entropy
is recovered by adding the binomial degeneracy of each sector (``microstate_entropy``). It is the
numpy twin of ``glauber.build_mean_field_reduced_system``, built independently so that
``validate_sector_reduction`` can check one against the other.

Four analyses sit on top of it, each returning a dataclass of arrays that
``plotting.ising_pop`` draws (this module does not import it):

1. ``hidden_housekeeping``: identical state dynamics implemented by one effective bath or two
   physical baths, separating transient information loss from hidden housekeeping EP.
2. ``symmetry_bit_relaxation``: forgetting a single symmetry-breaking bit, for which the asymptotic
   POP mismatch is exactly ``log(2)``.
3. ``critical_scan``: critical slowing of POP saturation across the Curie-Weiss transition.
4. ``periodic_drive_scan``: a genuinely periodic square-wave field protocol and its Floquet POP bound.

Each analysis checks the identities it relies on (stationary state is Gibbs, first law, mismatch
bounded by entropy production) and raises ``AssertionError`` rather than return a number that
violates them.

Moved out of ``notebooks/stoch_thermo/ising_pop_physics.py``; the figure code is
``plotting/ising_pop.py`` and the command line is ``examples/ising_pop_physics.py``.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigvals, expm
from scipy.special import gammaln, logsumexp

import physics.glauber as glauber
import physics.markov as markov

Array = np.ndarray


@dataclass(frozen=True)
class SectorModel:
  """Exact Curie-Weiss dynamics lumped by the number of up spins."""

  n_spins: int
  J: float
  h: float
  betas: Array
  gammas: Array
  k: Array
  magnetization: Array
  energy: Array
  log_degeneracy: Array
  generator: Array
  heat_rate: Array


@dataclass(frozen=True)
class StepMap:
  """Finite-time stochastic map and integrated bath-heat maps."""

  G: Array
  heat: Array


@dataclass(frozen=True)
class RepeatedMetrics:
  """Cycle-resolved information and thermodynamic observables."""

  states: Array
  qbars: Array
  pmmc: Array
  total_ep: Array
  stationary_kl_drop: Array
  ep_remainder: Array
  cumulative_heat: Array
  cumulative_work: Array
  stationary: Array


def _as_1d(values: Iterable[float]) -> Array:
  result = np.asarray(tuple(values), dtype=float)
  if result.ndim != 1 or result.size == 0:
    raise ValueError("expected a nonempty one-dimensional parameter array")
  return result


def shannon(prob: Array, axis: int = -1) -> Array:
  """Shannon entropy with the convention ``0 log 0 = 0``."""

  prob = np.asarray(prob, dtype=float)
  terms = np.zeros_like(prob)
  positive = prob > 0.0
  terms[positive] = prob[positive] * np.log(prob[positive])
  return -np.sum(terms, axis=axis)


def microstate_entropy(sector_prob: Array, log_degeneracy: Array) -> Array:
  """Entropy of a distribution uniform within each magnetization sector."""

  sector_prob = np.asarray(sector_prob, dtype=float)
  return shannon(sector_prob, axis=-1) + sector_prob @ log_degeneracy


def relative_entropy(p: Array, q: Array) -> float:
  """KL divergence between sector distributions."""

  p = np.asarray(p, dtype=float)
  q = np.asarray(q, dtype=float)
  positive = p > 0.0
  if np.any(q[positive] <= 0.0):
    return float("inf")
  return float(np.sum(p[positive] * (np.log(p[positive]) - np.log(q[positive]))))


def build_sector_model(
  n_spins: int,
  J: float,
  h: float,
  betas: Iterable[float],
  gammas: Iterable[float],
) -> SectorModel:
  """Construct the exact ``n_spins + 1`` Curie-Weiss birth-death chain.

  The energy and symmetric Arrhenius rates match ``make_mean_field_graph``
  and ``get_rates`` in ``physics/glauber.py``:

  ``H = -J (M^2 - n)/(2n) - h M`` and
  ``k_nu(Delta E) = gamma_nu exp(-beta_nu Delta E / 2)``.
  """

  if n_spins < 1:
    raise ValueError("n_spins must be positive")
  betas_array = _as_1d(betas)
  gammas_array = _as_1d(gammas)
  if betas_array.shape != gammas_array.shape:
    raise ValueError("betas and gammas must have the same shape")
  if np.any(gammas_array <= 0.0):
    raise ValueError("all bath coupling rates must be positive")

  k = np.arange(n_spins + 1, dtype=float)
  total_magnetization = 2.0 * k - n_spins
  magnetization = total_magnetization / n_spins
  energy = (
    -J * (total_magnetization**2 - n_spins) / (2.0 * n_spins)
    - h * total_magnetization
  )
  log_degeneracy = (
    gammaln(n_spins + 1.0) - gammaln(k + 1.0) - gammaln(n_spins - k + 1.0)
  )

  dimension = n_spins + 1
  generator = np.zeros((dimension, dimension), dtype=float)
  heat_rate = np.zeros((betas_array.size, dimension), dtype=float)

  for source in range(dimension):
    if source < n_spins:
      delta_energy = energy[source + 1] - energy[source]
      channel_rates = gammas_array * np.exp(-0.5 * betas_array * delta_energy)
      multiplicity = n_spins - source
      macro_rates = multiplicity * channel_rates
      generator[source + 1, source] += np.sum(macro_rates)
      heat_rate[:, source] += macro_rates * delta_energy

    if source > 0:
      delta_energy = energy[source - 1] - energy[source]
      channel_rates = gammas_array * np.exp(-0.5 * betas_array * delta_energy)
      multiplicity = source
      macro_rates = multiplicity * channel_rates
      generator[source - 1, source] += np.sum(macro_rates)
      heat_rate[:, source] += macro_rates * delta_energy

  generator[np.diag_indices(dimension)] = -np.sum(generator, axis=0)
  return SectorModel(
    n_spins=n_spins,
    J=float(J),
    h=float(h),
    betas=betas_array,
    gammas=gammas_array,
    k=k,
    magnetization=magnetization,
    energy=energy,
    log_degeneracy=log_degeneracy,
    generator=generator,
    heat_rate=heat_rate,
  )


def make_step_map(model: SectorModel, duration: float) -> StepMap:
  """Integrate the CTMC and each bath heat over one constant stage."""

  if duration <= 0.0:
    raise ValueError("duration must be positive")
  dimension = model.generator.shape[0]
  n_baths = model.heat_rate.shape[0]
  augmented = np.zeros((dimension + n_baths, dimension + n_baths), dtype=float)
  augmented[:dimension, :dimension] = model.generator.T
  augmented[:dimension, dimension:] = model.heat_rate.T
  propagated = expm(augmented * duration)
  G = propagated[:dimension, :dimension].T
  heat = propagated[:dimension, dimension:].T
  column_error = float(np.max(np.abs(G.sum(axis=0) - 1.0)))
  if column_error > 2e-11 or float(G.min()) < -2e-11:
    raise AssertionError(
      "finite-time map is not stochastic: "
      f"column error={column_error:.3e}, minimum={G.min():.3e}"
    )
  return StepMap(G=G, heat=heat)


def stationary_distribution(operator: Array, *, continuous: bool) -> Array:
  """Return the normalized stationary right eigenvector of ``K`` or ``G``."""

  operator = np.asarray(operator, dtype=float)
  values, vectors = np.linalg.eig(operator)
  target = 0.0 if continuous else 1.0
  index = int(np.argmin(np.abs(values - target)))
  stationary = np.real(vectors[:, index])
  if stationary.sum() < 0.0:
    stationary = -stationary
  stationary[np.abs(stationary) < 1e-15] = 0.0
  if stationary.min() < -1e-9:
    raise RuntimeError("stationary eigenvector has a significant negative component")
  stationary = np.maximum(stationary, 0.0)
  total = float(stationary.sum())
  if not np.isfinite(total) or total <= 0.0:
    raise RuntimeError("stationary eigenvector cannot be normalized")
  stationary /= total
  residual_vector = operator @ stationary
  if not continuous:
    residual_vector = residual_vector - stationary
  residual = float(np.linalg.norm(residual_vector, ord=1))
  scale = max(1.0, float(np.linalg.norm(operator, ord=1)))
  if residual > 2e-10 * scale:
    raise RuntimeError(f"stationary eigenvector residual is {residual:.3e}")
  return stationary


def gibbs_sector_distribution(model: SectorModel, beta: float) -> Array:
  """Canonical sector probabilities, including binomial multiplicities."""

  log_weights = model.log_degeneracy - beta * model.energy
  return np.exp(log_weights - logsumexp(log_weights))


def spectral_gap(generator: Array) -> float:
  """Smallest nonzero relaxation rate of an irreducible generator."""

  values = np.real_if_close(eigvals(generator), tol=1000)
  values = np.real(values)
  values.sort()
  values = values[::-1]
  if values.size < 2:
    return float("nan")
  return float(max(0.0, -values[1]))


def repeat_map(
  step: StepMap,
  p0: Array,
  n_cycles: int,
  log_degeneracy: Array,
  betas: Iterable[float],
  *,
  work_map: Array | None = None,
) -> RepeatedMetrics:
  """Evaluate POP mismatch and thermodynamic costs over repeated cycles."""

  if n_cycles < 1:
    raise ValueError("n_cycles must be positive")
  G = np.asarray(step.G, dtype=float)
  heat_map = np.asarray(step.heat, dtype=float)
  p0 = np.asarray(p0, dtype=float)
  log_degeneracy = np.asarray(log_degeneracy, dtype=float)
  betas_array = _as_1d(betas)
  if G.ndim != 2 or G.shape[0] != G.shape[1]:
    raise ValueError("G must be a square matrix")
  if p0.shape != (G.shape[0],) or log_degeneracy.shape != p0.shape:
    raise ValueError("p0 and log_degeneracy must match the map dimension")
  if heat_map.ndim != 2 or heat_map.shape[1] != p0.size:
    raise ValueError("heat map width must match the map dimension")
  if heat_map.shape[0] != betas_array.size:
    raise ValueError("one inverse temperature is required per heat channel")
  if (
    not np.all(np.isfinite(p0))
    or float(p0.min()) < 0.0
    or abs(float(p0.sum()) - 1.0) > 2e-12
  ):
    raise ValueError("p0 must be a normalized nonnegative distribution")
  if (
    float(G.min()) < -2e-12
    or np.max(np.abs(G.sum(axis=0) - 1.0)) > 2e-11
  ):
    raise ValueError("G must be column stochastic")
  if work_map is None:
    work_map = np.zeros_like(p0)
  work_map = np.asarray(work_map, dtype=float)
  if work_map.shape != p0.shape:
    raise ValueError("work map must match the map dimension")

  dimension = p0.size
  states = np.zeros((n_cycles + 1, dimension), dtype=float)
  qbars = np.zeros((n_cycles, dimension), dtype=float)
  pmmc = np.zeros(n_cycles, dtype=float)
  total_ep = np.zeros(n_cycles, dtype=float)
  cumulative_heat = np.zeros((n_cycles, betas_array.size), dtype=float)
  cumulative_work = np.zeros(n_cycles, dtype=float)

  states[0] = p0
  state_sum = np.zeros(dimension, dtype=float)
  heat_sum = np.zeros(betas_array.size, dtype=float)
  work_sum = 0.0
  entropy0 = float(microstate_entropy(p0, log_degeneracy))

  for index in range(n_cycles):
    p = states[index]
    state_sum += p
    heat_sum += heat_map @ p
    work_sum += float(work_map @ p)
    p_next = G @ p
    if float(p_next.min()) < -2e-12 or abs(float(p_next.sum()) - 1.0) > 2e-10:
      raise RuntimeError("repeated map produced an invalid probability vector")
    states[index + 1] = p_next

    count = index + 1
    qbar = state_sum / count
    qbars[index] = qbar
    Gqbar = G @ qbar
    entropy_next = float(microstate_entropy(p_next, log_degeneracy))
    pmmc[index] = (
      entropy_next
      - entropy0
      + count
      * (
        float(microstate_entropy(qbar, log_degeneracy))
        - float(microstate_entropy(Gqbar, log_degeneracy))
      )
    )
    cumulative_heat[index] = heat_sum
    cumulative_work[index] = work_sum
    total_ep[index] = entropy_next - entropy0 - float(betas_array @ heat_sum)

  stationary = stationary_distribution(G, continuous=False)
  initial_kl = relative_entropy(p0, stationary)
  stationary_kl_drop = np.array(
    [initial_kl - relative_entropy(p, stationary) for p in states[1:]],
    dtype=float,
  )
  ep_remainder = total_ep - stationary_kl_drop
  return RepeatedMetrics(
    states=states,
    qbars=qbars,
    pmmc=pmmc,
    total_ep=total_ep,
    stationary_kl_drop=stationary_kl_drop,
    ep_remainder=ep_remainder,
    cumulative_heat=cumulative_heat,
    cumulative_work=cumulative_work,
    stationary=stationary,
  )


def validate_sector_reduction() -> dict[str, float]:
  """Check the reduced generator and heat vectors against ``physics/glauber.py``."""

  n_spins = 5
  J = 0.37
  h = -0.11
  betas = np.array([2.4, 0.8])
  gammas = np.array([1.0, 0.7])
  reduced = build_sector_model(n_spins, J, h, betas, gammas)

  graph = glauber.make_mean_field_graph(n_spins)
  rows, spins = glauber.make_state_space_graph(n_spins)
  rates_nu = np.asarray(glauber.get_rates(graph, J, h, betas, gammas))
  micro_generator = np.asarray(
    glauber.dense_generator_from_rates(rows, rates_nu.sum(axis=0))
  )
  d_energy = J * np.asarray(graph[1]) + h * np.asarray(graph[2])
  micro_heat = np.asarray(glauber.heat_rate_vectors(rates_nu, d_energy))

  up_count = ((np.asarray(spins).sum(axis=0) + n_spins) / 2).astype(int)
  dimension = n_spins + 1
  projected_generator = np.zeros((dimension, dimension), dtype=float)
  projected_heat = np.zeros((betas.size, dimension), dtype=float)

  for source_sector in range(dimension):
    sources = np.flatnonzero(up_count == source_sector)
    for destination_sector in range(dimension):
      destinations = np.flatnonzero(up_count == destination_sector)
      projected_generator[destination_sector, source_sector] = np.sum(
        micro_generator[np.ix_(destinations, sources)]
      ) / sources.size
    projected_heat[:, source_sector] = np.mean(micro_heat[:, sources], axis=1)

  generator_error = float(np.max(np.abs(projected_generator - reduced.generator)))
  heat_error = float(np.max(np.abs(projected_heat - reduced.heat_rate)))
  reduced_step = make_step_map(reduced, 0.13)
  p0 = all_up(n_spins)
  n_cycles = 7
  reduced_metrics = repeat_map(
    reduced_step,
    p0,
    n_cycles,
    reduced.log_degeneracy,
    betas,
  )
  reference_pmmc = float(markov.bound2(p0, reduced_step.G, n_cycles)[0])
  pmmc_error = abs(float(reduced_metrics.pmmc[-1]) - reference_pmmc)
  if generator_error > 2e-5 or heat_error > 2e-5 or pmmc_error > 2e-5:
    raise AssertionError(
      "magnetization reduction disagrees with physics/glauber.py: "
      f"generator={generator_error:.3e}, heat={heat_error:.3e}, "
      f"POP={pmmc_error:.3e}"
    )
  return {
    "generator_error": generator_error,
    "heat_error": heat_error,
    "pmmc_error": pmmc_error,
  }


def all_up(n_spins: int) -> Array:
  p0 = np.zeros(n_spins + 1, dtype=float)
  p0[-1] = 1.0
  return p0


def first_crossing(values: Array, threshold: float, dt: float) -> float:
  indices = np.flatnonzero(values >= threshold)
  return float((indices[0] + 1) * dt) if indices.size else float("nan")


def make_alternating_field_cycle(
  n_spins: int,
  J: float,
  beta: float,
  h0: float,
  period: float,
) -> tuple[SectorModel, StepMap, Array, float]:
  """Build one cycle with ``+h0`` followed by ``-h0`` and two quenches."""

  plus_model = build_sector_model(n_spins, J, h0, [beta], [1.0])
  minus_model = build_sector_model(n_spins, J, -h0, [beta], [1.0])
  plus_step = make_step_map(plus_model, period / 2.0)
  minus_step = make_step_map(minus_model, period / 2.0)

  G_cycle = minus_step.G @ plus_step.G
  heat_cycle = plus_step.heat + minus_step.heat @ plus_step.G
  quench_plus_to_minus = minus_model.energy - plus_model.energy
  quench_minus_to_plus = plus_model.energy - minus_model.energy
  work_map = (
    quench_plus_to_minus @ plus_step.G
    + quench_minus_to_plus @ G_cycle
  )

  energy_change_map = plus_model.energy @ G_cycle - plus_model.energy
  first_law_error = float(
    np.max(np.abs(energy_change_map - (np.sum(heat_cycle, axis=0) + work_map)))
  )
  if first_law_error > 2e-10:
    raise AssertionError(f"periodic-cycle first-law error is {first_law_error:.3e}")
  return plus_model, StepMap(G_cycle, heat_cycle), work_map, first_law_error


# --------------------------------------------------------------------------- #
# Analyses
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class HiddenHousekeeping:
  r"""Two baths versus one effective bath for the same state dynamics, plus a contrast scan."""
  n_spins: int
  J: float
  beta_bar: float
  model: SectorModel
  metrics: RepeatedMetrics
  times: Array
  matched_one_bath_ep: Array
  hidden_ep: Array
  steady_heat: Array
  steady_ep_rate: float
  information_budget: float
  stationary_error: float
  contrasts: Array
  currents: Array
  ep_rates: Array
  quadratic_coefficient: float

  @property
  def crossover_time(self) -> float:
    return self.information_budget / self.steady_ep_rate

  def summary(self) -> dict:
    return {
      "steady_heat": self.steady_heat,
      "steady_ep_rate": self.steady_ep_rate,
      "information_budget": self.information_budget,
      "crossover_time": self.crossover_time,
      "quadratic_coefficient": self.quadratic_coefficient,
      "stationary_error": self.stationary_error,
    }


def hidden_housekeeping(
  *,
  n_spins: int = 9,
  J: float = 0.2,
  beta_bar: float = 2.0,
  delta_beta: float = 2.0,
  dt: float = 0.1,
  n_cycles: int = 500,
  contrasts: Iterable[float] | None = None,
) -> HiddenHousekeeping:
  r"""Compare one- and two-bath implementations of identical state dynamics.

  The aggregate rates of the two baths at ``beta_bar +- delta_beta / 2`` obey detailed balance at
  ``beta_bar``, so reinterpreting them as one channel keeps ``K``, ``G`` and POP unchanged. What
  differs is the entropy production: the physical two-bath EP exceeds the matched one-bath EP by
  the hidden-channel housekeeping, which grows linearly at the steady rate ``-betas @ heat``.
  ``contrasts`` scans ``delta_beta`` for the steady heat currents and their quadratic EP rate.
  """
  betas = np.array([beta_bar + delta_beta / 2, beta_bar - delta_beta / 2])
  gammas = np.ones(2)
  model = build_sector_model(n_spins, J, 0.0, betas, gammas)
  step = make_step_map(model, dt)
  metrics = repeat_map(step, all_up(n_spins), n_cycles, model.log_degeneracy, betas)
  times = dt * np.arange(1, n_cycles + 1)
  entropy_change = (
    microstate_entropy(metrics.states[1:], model.log_degeneracy)
    - microstate_entropy(metrics.states[0], model.log_degeneracy)
  )
  matched_one_bath_ep = entropy_change - beta_bar * np.sum(metrics.cumulative_heat, axis=1)
  hidden_ep = metrics.total_ep - matched_one_bath_ep

  stationary = stationary_distribution(model.generator, continuous=True)
  gibbs = gibbs_sector_distribution(model, beta_bar)
  stationary_error = float(np.max(np.abs(stationary - gibbs)))
  steady_heat = model.heat_rate @ stationary
  steady_ep_rate = float(-betas @ steady_heat)
  information_budget = relative_entropy(all_up(n_spins), stationary)
  if stationary_error > 2e-11:
    raise AssertionError(
      f"two-bath aggregate stationary state is not Gibbs: {stationary_error:.3e}"
    )
  if abs(float(np.sum(steady_heat))) > 2e-11 or steady_ep_rate < -2e-11:
    raise AssertionError("stationary heat or entropy-production balance failed")
  if (
    np.min(hidden_ep) < -2e-9
    or np.max(metrics.pmmc - matched_one_bath_ep) > 2e-8
    or np.max(np.abs(hidden_ep - metrics.ep_remainder)) > 2e-8
  ):
    raise AssertionError("hidden-channel or matched-bath mismatch inequality failed")

  contrasts = np.linspace(0.0, 2.0, 21) if contrasts is None else np.asarray(tuple(contrasts), dtype=float)
  currents = np.zeros((contrasts.size, 2), dtype=float)
  ep_rates = np.zeros(contrasts.size, dtype=float)
  for index, contrast in enumerate(contrasts):
    scan_betas = np.array([beta_bar + contrast / 2, beta_bar - contrast / 2])
    scan_model = build_sector_model(n_spins, J, 0.0, scan_betas, gammas)
    scan_stationary = stationary_distribution(scan_model.generator, continuous=True)
    currents[index] = scan_model.heat_rate @ scan_stationary
    ep_rates[index] = -scan_betas @ currents[index]

  fit_mask = (contrasts > 0.0) & (contrasts <= 0.4)
  squared_contrast = contrasts[fit_mask] ** 2
  quadratic_coefficient = float(
    squared_contrast @ ep_rates[fit_mask] / (squared_contrast @ squared_contrast)
  )
  return HiddenHousekeeping(
    n_spins=n_spins,
    J=J,
    beta_bar=beta_bar,
    model=model,
    metrics=metrics,
    times=times,
    matched_one_bath_ep=matched_one_bath_ep,
    hidden_ep=hidden_ep,
    steady_heat=steady_heat,
    steady_ep_rate=steady_ep_rate,
    information_budget=information_budget,
    stationary_error=stationary_error,
    contrasts=contrasts,
    currents=currents,
    ep_rates=ep_rates,
    quadratic_coefficient=quadratic_coefficient,
  )


@dataclass(frozen=True)
class SymmetryBitRun:
  r"""Relaxation of the state "equilibrium conditioned on ``m > 0``", exactly one bit from Gibbs."""
  n_spins: int
  J: float
  beta: float
  model: SectorModel
  metrics: RepeatedMetrics
  times: Array
  positive_mass: Array  # Pr(m > 0) at cycles 0..n_cycles
  initial_kl: float
  t50: float
  t90: float

  def summary(self) -> dict:
    return {
      "initial_kl": self.initial_kl,
      "final_pmmc": float(self.metrics.pmmc[-1]),
      "t50": self.t50,
      "t90": self.t90,
    }


def symmetry_bit_relaxation(
  n_spins: int,
  *,
  J: float = 1.0,
  beta: float = 1.25,
  dt: float = 0.1,
  n_cycles: int = 3000,
) -> SymmetryBitRun:
  r"""Track the loss of one positive-magnetization preparation bit; ``n_spins`` must be odd."""
  model = build_sector_model(n_spins, J, 0.0, [beta], [1.0])
  step = make_step_map(model, dt)
  equilibrium = gibbs_sector_distribution(model, beta)
  positive_equilibrium_mass = float(equilibrium[model.magnetization > 0.0].sum())
  if abs(positive_equilibrium_mass - 0.5) > 2e-12:
    raise AssertionError("the exact symmetry-bit preparation requires odd n_spins and h=0")
  p_plus = equilibrium * (model.magnetization > 0.0)
  p_plus /= p_plus.sum()
  bit_kl = relative_entropy(p_plus, equilibrium)
  if abs(bit_kl - np.log(2.0)) > 2e-12:
    raise AssertionError("positive-magnetization preparation is not one bit")
  metrics = repeat_map(step, p_plus, n_cycles, model.log_degeneracy, [beta])
  return SymmetryBitRun(
    n_spins=n_spins,
    J=J,
    beta=beta,
    model=model,
    metrics=metrics,
    times=dt * np.arange(1, n_cycles + 1),
    positive_mass=metrics.states[:, model.magnetization > 0.0].sum(axis=1),
    initial_kl=bit_kl,
    t50=first_crossing(metrics.pmmc, 0.5 * np.log(2.0), dt),
    t90=first_crossing(metrics.pmmc, 0.9 * np.log(2.0), dt),
  )


@dataclass(frozen=True)
class CriticalScan:
  r"""POP saturation, relaxation and heat conduction across the Curie-Weiss coupling ``J_c = 1/beta_bar``."""
  n_spins: int
  beta_bar: float
  critical_J: float
  J_values: Array
  horizons: tuple[float, ...]
  relaxation_times: Array
  abs_magnetization: Array
  susceptibility: Array
  steady_ep_rate: Array
  information_budget: Array
  fractions: Array  # (len(horizons), len(J_values)): M_T^* / D(p0 || pi)
  t50: Array

  def summary(self) -> dict:
    return {
      "critical_J": self.critical_J,
      "J_values": self.J_values,
      "relaxation_times": self.relaxation_times,
      "t50": self.t50,
    }


def critical_scan(
  *,
  n_spins: int = 51,
  beta_bar: float = 2.0,
  betas: Iterable[float] = (3.0, 1.0),
  J_values: Iterable[float] | None = None,
  dt: float = 0.1,
  max_time: float = 100.0,
  horizons: Iterable[float] = (1.0, 5.0, 20.0, 100.0),
) -> CriticalScan:
  r"""Scan the coupling at fixed two-bath temperatures, starting from all spins up."""
  betas = _as_1d(betas)
  gammas = np.ones(betas.size)
  critical_J = 1.0 / beta_bar
  J_values = np.linspace(0.25, 0.75, 21) if J_values is None else _as_1d(J_values)
  horizons = tuple(float(x) for x in horizons)
  n_cycles = int(round(max_time / dt))

  relaxation_times = np.zeros(J_values.size)
  abs_magnetization = np.zeros(J_values.size)
  susceptibility = np.zeros(J_values.size)
  steady_ep_rate = np.zeros(J_values.size)
  information_budget = np.zeros(J_values.size)
  fractions = np.zeros((len(horizons), J_values.size))
  t50 = np.full(J_values.size, np.nan)

  for index, J in enumerate(J_values):
    model = build_sector_model(n_spins, J, 0.0, betas, gammas)
    step = make_step_map(model, dt)
    metrics = repeat_map(step, all_up(n_spins), n_cycles, model.log_degeneracy, betas)
    stationary = stationary_distribution(model.generator, continuous=True)
    gap = spectral_gap(model.generator)
    relaxation_times[index] = 1.0 / gap if gap > 0.0 else np.inf
    abs_magnetization[index] = float(np.abs(model.magnetization) @ stationary)
    susceptibility[index] = float(beta_bar * n_spins * (model.magnetization**2 @ stationary))
    steady_ep_rate[index] = float(-betas @ (model.heat_rate @ stationary))
    plateau = relative_entropy(all_up(n_spins), stationary)
    information_budget[index] = plateau
    for horizon_index, horizon in enumerate(horizons):
      cycle_index = min(n_cycles, max(1, int(round(horizon / dt)))) - 1
      fractions[horizon_index, index] = metrics.pmmc[cycle_index] / plateau
    t50[index] = first_crossing(metrics.pmmc, 0.5 * plateau, dt)

  return CriticalScan(
    n_spins=n_spins,
    beta_bar=beta_bar,
    critical_J=critical_J,
    J_values=J_values,
    horizons=horizons,
    relaxation_times=relaxation_times,
    abs_magnetization=abs_magnetization,
    susceptibility=susceptibility,
    steady_ep_rate=steady_ep_rate,
    information_budget=information_budget,
    fractions=fractions,
    t50=t50,
  )


def hysteresis_loop(
  n_spins: int, J: float, beta: float, h0: float, period: float, *, n_points: int = 36
) -> tuple[Array, Array]:
  r"""Field and mean magnetization around the periodic steady state of the square-wave drive.

  The path runs through the ``+h0`` stage, the quench to ``-h0``, the ``-h0`` stage and the quench
  back, so plotting magnetization against field closes into a loop.
  """
  plus_model, cycle, _, _ = make_alternating_field_cycle(n_spins, J, beta, h0, period)
  minus_model = build_sector_model(n_spins, J, -h0, [beta], [1.0])
  periodic_state = stationary_distribution(cycle.G, continuous=False)
  stage_times = np.linspace(0.0, period / 2.0, n_points)
  plus_states = np.array([expm(plus_model.generator * t) @ periodic_state for t in stage_times])
  midpoint_state = plus_states[-1]
  minus_states = np.array([expm(minus_model.generator * t) @ midpoint_state for t in stage_times])
  plus_m = plus_states @ plus_model.magnetization
  minus_m = minus_states @ minus_model.magnetization
  field_path = np.concatenate(
    [np.full(stage_times.size, h0), [h0, -h0], np.full(stage_times.size, -h0), [-h0, h0]]
  )
  magnetization_path = np.concatenate(
    [plus_m, [plus_m[-1], plus_m[-1]], minus_m, [minus_m[-1], minus_m[-1]]]
  )
  return field_path, magnetization_path


@dataclass(frozen=True)
class PeriodicDriveScan:
  r"""Square-wave field of period ``T``: periodic-steady work and dissipation, and Floquet POP."""
  n_spins: int
  J: float
  beta: float
  h0: float
  periods: Array
  n_observation_cycles: int
  steady_work: Array
  steady_ep: Array
  final_pmmc: Array
  final_fraction: Array
  first_law_errors: Array
  steady_identity_errors: Array
  loops: dict = field(default_factory=dict)  # period -> (field path, magnetization path)

  def summary(self) -> dict:
    return {
      "periods": self.periods,
      "steady_work": self.steady_work,
      "steady_ep": self.steady_ep,
      "max_first_law_error": float(self.first_law_errors.max()),
      "max_steady_identity_error": float(self.steady_identity_errors.max()),
    }


def periodic_drive_scan(
  *,
  n_spins: int = 31,
  J: float = 1.0,
  beta: float = 1.25,
  h0: float = 0.25,
  periods: Iterable[float] | None = None,
  n_observation_cycles: int = 100,
) -> PeriodicDriveScan:
  r"""Scan the drive period with two sudden quenches per cycle, one bath, starting all up.

  Hysteresis loops are kept for the shortest, middle and longest period.
  """
  periods = np.logspace(-1.3, 2.0, 28) if periods is None else _as_1d(periods)
  steady_work = np.zeros(periods.size)
  steady_ep = np.zeros(periods.size)
  final_pmmc = np.zeros(periods.size)
  final_fraction = np.zeros(periods.size)
  first_law_errors = np.zeros(periods.size)
  steady_identity_errors = np.zeros(periods.size)

  p0 = all_up(n_spins)
  for index, period in enumerate(periods):
    model, cycle, work_map, error = make_alternating_field_cycle(n_spins, J, beta, h0, float(period))
    metrics = repeat_map(
      cycle, p0, n_observation_cycles, model.log_degeneracy, [beta], work_map=work_map
    )
    periodic_state = stationary_distribution(cycle.G, continuous=False)

    steady_work[index] = float(work_map @ periodic_state)
    steady_heat = float(cycle.heat[0] @ periodic_state)
    steady_ep[index] = -beta * steady_heat
    final_pmmc[index] = metrics.pmmc[-1]
    final_fraction[index] = metrics.pmmc[-1] / metrics.total_ep[-1]
    first_law_errors[index] = error
    steady_identity_errors[index] = abs(steady_ep[index] - beta * steady_work[index])
    if metrics.total_ep[-1] < -2e-9 or metrics.pmmc[-1] - metrics.total_ep[-1] > 2e-8:
      raise AssertionError("periodic mismatch exceeds physical entropy production")

  if float(steady_identity_errors.max()) > 2e-9:
    raise AssertionError("steady one-bath identity Sigma = beta W failed")

  loops = {
    float(periods[i]): hysteresis_loop(n_spins, J, beta, h0, float(periods[i]))
    for i in sorted({0, periods.size // 2, periods.size - 1})
  }
  return PeriodicDriveScan(
    n_spins=n_spins,
    J=J,
    beta=beta,
    h0=h0,
    periods=periods,
    n_observation_cycles=n_observation_cycles,
    steady_work=steady_work,
    steady_ep=steady_ep,
    final_pmmc=final_pmmc,
    final_fraction=final_fraction,
    first_law_errors=first_law_errors,
    steady_identity_errors=steady_identity_errors,
    loops=loops,
  )
