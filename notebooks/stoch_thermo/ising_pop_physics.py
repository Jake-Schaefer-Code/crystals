"""Physical analyses of periodic-optimal mismatch in the Curie-Weiss model.

This script extends the finite-state Ising calculations in ``ising.py`` with
an exact magnetization-sector reduction.  It generates four figure sets:

1. Identical state dynamics implemented by one effective bath or two physical
   baths, separating transient information loss from hidden housekeeping EP.
2. Forgetting a single symmetry-breaking bit, for which the asymptotic POP
   mismatch is exactly ``log(2)``.
3. Critical slowing of POP saturation across the Curie-Weiss transition.
4. A genuinely periodic square-wave field protocol and its Floquet POP bound.

The reduced state ``k = 0, ..., n_spins`` is the number of up spins.  For an
exchangeable distribution, this reduction preserves every KL divergence
exactly.  Microstate Shannon entropy is recovered by including the binomial
degeneracy of each magnetization sector.

Run a fast smoke calculation with::

    python ising_pop_physics.py --quick --output-dir /tmp/ising_pop_physics

Omit ``--quick`` for the denser publication-oriented sweeps.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigvals, expm
from scipy.special import gammaln, logsumexp

import ising
import utils


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
    and ``get_rates`` in ``ising.py``:

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
    """Check the reduced generator and heat vectors against ``ising.py``."""

    n_spins = 5
    J = 0.37
    h = -0.11
    betas = np.array([2.4, 0.8])
    gammas = np.array([1.0, 0.7])
    reduced = build_sector_model(n_spins, J, h, betas, gammas)

    graph = ising.make_mean_field_graph(n_spins)
    rows, spins = ising.make_state_space_graph(n_spins)
    rates_nu = np.asarray(ising.get_rates(graph, J, h, betas, gammas))
    micro_generator = np.asarray(
        ising.dense_generator_from_rates(rows, rates_nu.sum(axis=0))
    )
    d_energy = J * np.asarray(graph[1]) + h * np.asarray(graph[2])
    micro_heat = np.asarray(ising.heat_rate_vectors(rates_nu, d_energy))

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
    p0 = _all_up(n_spins)
    n_cycles = 7
    reduced_metrics = repeat_map(
        reduced_step,
        p0,
        n_cycles,
        reduced.log_degeneracy,
        betas,
    )
    reference_pmmc = float(utils.bound2(p0, reduced_step.G, n_cycles)[0])
    pmmc_error = abs(float(reduced_metrics.pmmc[-1]) - reference_pmmc)
    if generator_error > 2e-5 or heat_error > 2e-5 or pmmc_error > 2e-5:
        raise AssertionError(
            "magnetization reduction disagrees with ising.py: "
            f"generator={generator_error:.3e}, heat={heat_error:.3e}, "
            f"POP={pmmc_error:.3e}"
        )
    return {
        "generator_error": generator_error,
        "heat_error": heat_error,
        "pmmc_error": pmmc_error,
    }


def _all_up(n_spins: int) -> Array:
    p0 = np.zeros(n_spins + 1, dtype=float)
    p0[-1] = 1.0
    return p0


def _first_crossing(values: Array, threshold: float, dt: float) -> float:
    indices = np.flatnonzero(values >= threshold)
    return float((indices[0] + 1) * dt) if indices.size else float("nan")


def _finish_figure(fig: plt.Figure, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"figure was not written: {output_path}")
    return output_path


def plot_same_G_hidden_housekeeping(output_dir: Path, *, quick: bool) -> tuple[Path, dict]:
    """Compare one- and two-bath implementations of identical state dynamics."""

    n_spins = 9
    J = 0.2
    beta_bar = 2.0
    delta_beta = 2.0
    betas = np.array([beta_bar + delta_beta / 2, beta_bar - delta_beta / 2])
    gammas = np.ones(2)
    dt = 0.1
    n_cycles = 120 if quick else 500

    model = build_sector_model(n_spins, J, 0.0, betas, gammas)
    step = make_step_map(model, dt)
    metrics = repeat_map(
        step,
        _all_up(n_spins),
        n_cycles,
        model.log_degeneracy,
        betas,
    )
    times = dt * np.arange(1, n_cycles + 1)
    entropy_change = (
        microstate_entropy(metrics.states[1:], model.log_degeneracy)
        - microstate_entropy(metrics.states[0], model.log_degeneracy)
    )
    # The aggregate rates obey detailed balance at beta_bar.  Reinterpreting
    # those same rates as one channel preserves K, G, and POP exactly.
    matched_one_bath_ep = entropy_change - beta_bar * np.sum(
        metrics.cumulative_heat, axis=1
    )
    hidden_ep = metrics.total_ep - matched_one_bath_ep

    stationary = stationary_distribution(model.generator, continuous=True)
    gibbs = gibbs_sector_distribution(model, beta_bar)
    stationary_error = float(np.max(np.abs(stationary - gibbs)))
    steady_heat = model.heat_rate @ stationary
    steady_ep_rate = float(-betas @ steady_heat)
    information_budget = relative_entropy(_all_up(n_spins), stationary)
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

    contrasts = (
        np.array([0.0, 0.1, 0.2, 0.35, 0.7, 1.2, 2.0])
        if quick
        else np.linspace(0.0, 2.0, 21)
    )
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

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(times, metrics.pmmc, label=r"POP $M_N^*$", lw=2.2)
    ax.plot(times, matched_one_bath_ep, label="matched one-bath EP", lw=1.9)
    ax.plot(times, metrics.total_ep, label="physical two-bath EP", lw=1.9)
    ax.axhline(information_budget, color="0.35", ls=":", label=r"$D(p_0\Vert\pi)$")
    ax.set(xlabel="time", ylabel="entropy (nats)", title="Same state dynamics, different EP")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    ax.plot(times, hidden_ep, label="hidden channel EP", lw=2.2)
    ax.plot(times, steady_ep_rate * times, "--", label=r"$\dot\Sigma_{ss}t$")
    ax.set(xlabel="time", ylabel="entropy (nats)", title="Recurring housekeeping contribution")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    ax.plot(contrasts, currents[:, 0] / n_spins, marker="o", ms=3, label=r"$\dot Q_1$")
    ax.plot(contrasts, currents[:, 1] / n_spins, marker="o", ms=3, label=r"$\dot Q_2$")
    ax.axhline(0.0, color="0.3", lw=0.8)
    ax.set(
        xlabel=r"bath contrast $\Delta\beta$",
        ylabel="steady heat current per spin",
        title="Hidden heat conduction",
    )
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.plot(contrasts, ep_rates / n_spins, "o", ms=4, label="computed")
    ax.plot(
        contrasts,
        quadratic_coefficient * contrasts**2 / n_spins,
        "--",
        label=fr"${quadratic_coefficient / n_spins:.4f}(\Delta\beta)^2$",
    )
    ax.set(
        xlabel=r"bath contrast $\Delta\beta$",
        ylabel=r"$\dot\Sigma_{ss}/n_s$",
        title="Near-equilibrium quadratic scaling",
    )
    ax.legend(frameon=False)

    fig.suptitle(
        fr"Hidden housekeeping for $n_s={n_spins}$, $J={J}$, $\bar\beta={beta_bar}$",
        fontsize=13,
    )
    path = _finish_figure(fig, output_dir / "same_G_hidden_housekeeping.png")
    return path, {
        "steady_heat": steady_heat,
        "steady_ep_rate": steady_ep_rate,
        "information_budget": information_budget,
        "crossover_time": information_budget / steady_ep_rate,
        "quadratic_coefficient": quadratic_coefficient,
        "stationary_error": stationary_error,
    }


def plot_symmetry_bit(output_dir: Path, *, quick: bool) -> tuple[Path, dict]:
    """Track loss of a single positive-magnetization preparation bit."""

    sizes = [9, 15] if quick else [9, 15, 21, 31]
    J = 1.0
    beta = 1.25
    dt = 0.1
    n_cycles = 600 if quick else 3000
    times = dt * np.arange(1, n_cycles + 1)
    summaries: dict[int, dict[str, float]] = {}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for size_index, n_spins in enumerate(sizes):
        color = colors[size_index % len(colors)]
        model = build_sector_model(n_spins, J, 0.0, [beta], [1.0])
        step = make_step_map(model, dt)
        equilibrium = gibbs_sector_distribution(model, beta)
        positive_equilibrium_mass = float(
            equilibrium[model.magnetization > 0.0].sum()
        )
        if abs(positive_equilibrium_mass - 0.5) > 2e-12:
            raise AssertionError(
                "the exact symmetry-bit preparation requires odd n_spins and h=0"
            )
        p_plus = equilibrium * (model.magnetization > 0.0)
        p_plus /= p_plus.sum()
        bit_kl = relative_entropy(p_plus, equilibrium)
        if abs(bit_kl - np.log(2.0)) > 2e-12:
            raise AssertionError("positive-magnetization preparation is not one bit")
        metrics = repeat_map(
            step,
            p_plus,
            n_cycles,
            model.log_degeneracy,
            [beta],
        )
        positive_mass = metrics.states[:, model.magnetization > 0.0].sum(axis=1)
        axes[0].semilogx(
            times,
            metrics.pmmc / np.log(2.0),
            color=color,
            lw=2,
            label=fr"$n_s={n_spins}$",
        )
        axes[0].semilogx(
            times,
            metrics.total_ep / np.log(2.0),
            color=color,
            lw=1.2,
            ls="--",
            alpha=0.8,
        )
        axes[1].semilogx(
            times,
            positive_mass[1:],
            color=color,
            lw=2,
            label=fr"$n_s={n_spins}$",
        )
        summaries[n_spins] = {
            "initial_kl": bit_kl,
            "final_pmmc": float(metrics.pmmc[-1]),
            "t50": _first_crossing(metrics.pmmc, 0.5 * np.log(2.0), dt),
            "t90": _first_crossing(metrics.pmmc, 0.9 * np.log(2.0), dt),
        }

    axes[0].axhline(1.0, color="0.3", ls=":", label=r"$\ln 2$ plateau")
    axes[0].set(
        xlabel="time",
        ylabel=r"entropy / $\ln 2$",
        title="POP cost of forgetting one bit",
        ylim=(-0.03, 1.08),
    )
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    axes[0].text(0.03, 0.08, "solid: POP\ndashed: total EP", transform=axes[0].transAxes, fontsize=8)

    axes[1].axhline(0.5, color="0.3", ls=":", label="equilibrium mass")
    axes[1].set(
        xlabel="time",
        ylabel=r"$\Pr(m>0)$",
        title="Loss of the magnetization-sign memory",
        ylim=(0.45, 1.02),
    )
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle(fr"Symmetry-bit relaxation at $\beta J={beta * J:.2f}$", fontsize=13)
    path = _finish_figure(fig, output_dir / "symmetry_bit_forgetting.png")
    return path, summaries


def plot_critical_scaling(output_dir: Path, *, quick: bool) -> tuple[Path, dict]:
    """Compare POP saturation with critical relaxation and heat conduction."""

    n_spins = 21 if quick else 51
    beta_bar = 2.0
    betas = np.array([3.0, 1.0])
    gammas = np.ones(2)
    critical_J = 1.0 / beta_bar
    J_values = np.linspace(0.25, 0.75, 9 if quick else 21)
    dt = 0.1
    max_time = 20.0 if quick else 100.0
    n_cycles = int(round(max_time / dt))
    horizons = [0.5, 2.0, 5.0, 20.0] if quick else [1.0, 5.0, 20.0, 100.0]

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
        metrics = repeat_map(
            step,
            _all_up(n_spins),
            n_cycles,
            model.log_degeneracy,
            betas,
        )
        stationary = stationary_distribution(model.generator, continuous=True)
        gap = spectral_gap(model.generator)
        relaxation_times[index] = 1.0 / gap if gap > 0.0 else np.inf
        abs_magnetization[index] = float(np.abs(model.magnetization) @ stationary)
        susceptibility[index] = float(
            beta_bar * n_spins * (model.magnetization**2 @ stationary)
        )
        steady_ep_rate[index] = float(-betas @ (model.heat_rate @ stationary))
        plateau = relative_entropy(_all_up(n_spins), stationary)
        information_budget[index] = plateau
        for horizon_index, horizon in enumerate(horizons):
            cycle_index = min(n_cycles, max(1, int(round(horizon / dt)))) - 1
            fractions[horizon_index, index] = metrics.pmmc[cycle_index] / plateau
        t50[index] = _first_crossing(metrics.pmmc, 0.5 * plateau, dt)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for ax in axes.flat:
        ax.axvline(critical_J, color="0.3", ls=":", lw=1)

    axes[0, 0].semilogy(J_values, relaxation_times, "o-", ms=4)
    axes[0, 0].set(xlabel=r"coupling $J$", ylabel="relaxation time", title="Critical slowing")

    for horizon, values in zip(horizons, fractions):
        axes[0, 1].plot(J_values, values, marker="o", ms=3, label=fr"$T={horizon:g}$")
    axes[0, 1].set(
        xlabel=r"coupling $J$",
        ylabel=r"$M_T^*/D(p_0\Vert\pi)$",
        title="Finite-horizon POP saturation",
        ylim=(-0.03, 1.03),
    )
    axes[0, 1].legend(frameon=False, fontsize=8)

    axes[1, 0].plot(J_values, abs_magnetization, "o-", ms=4, label=r"$\langle|m|\rangle$")
    susceptibility_scaled = susceptibility / max(1.0, susceptibility.max())
    axes[1, 0].plot(J_values, susceptibility_scaled, "s--", ms=3, label="susceptibility (scaled)")
    axes[1, 0].set(xlabel=r"coupling $J$", ylabel="order/response", title="Finite-size ordering crossover")
    axes[1, 0].legend(frameon=False, fontsize=8)

    finite = np.isfinite(t50)
    if np.any(finite):
        recurring_cost = (
            steady_ep_rate[finite] * t50[finite] / information_budget[finite]
        )
        axes[1, 1].plot(J_values[finite], recurring_cost, "o-", ms=4)
    axes[1, 1].set(
        xlabel=r"coupling $J$",
        ylabel=r"$\dot\Sigma_{ss}t_{50}/D(p_0\Vert\pi)$",
        title="Housekeeping accrued by half-saturation",
    )

    fig.suptitle(
        fr"Two-bath Curie-Weiss crossover, $n_s={n_spins}$, $J_c={critical_J:.2f}$",
        fontsize=13,
    )
    path = _finish_figure(fig, output_dir / "critical_POP_scaling.png")
    return path, {
        "critical_J": critical_J,
        "J_values": J_values,
        "relaxation_times": relaxation_times,
        "t50": t50,
    }


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


def plot_periodic_drive(output_dir: Path, *, quick: bool) -> tuple[Path, dict]:
    """Analyze a periodic square-wave field with two sudden quenches per cycle."""

    n_spins = 15 if quick else 31
    J = 1.0
    beta = 1.25
    h0 = 0.25
    periods = np.logspace(-1.0, 1.4, 11) if quick else np.logspace(-1.3, 2.0, 28)
    n_observation_cycles = 30 if quick else 100

    steady_work = np.zeros(periods.size)
    steady_ep = np.zeros(periods.size)
    final_pmmc = np.zeros(periods.size)
    final_fraction = np.zeros(periods.size)
    first_law_errors = np.zeros(periods.size)
    steady_identity_errors = np.zeros(periods.size)

    p0 = _all_up(n_spins)
    for index, period in enumerate(periods):
        model, cycle, work_map, error = make_alternating_field_cycle(
            n_spins, J, beta, h0, float(period)
        )
        metrics = repeat_map(
            cycle,
            p0,
            n_observation_cycles,
            model.log_degeneracy,
            [beta],
            work_map=work_map,
        )
        periodic_state = stationary_distribution(cycle.G, continuous=False)

        steady_work[index] = float(work_map @ periodic_state)
        steady_heat = float(cycle.heat[0] @ periodic_state)
        steady_ep[index] = -beta * steady_heat
        final_pmmc[index] = metrics.pmmc[-1]
        final_fraction[index] = metrics.pmmc[-1] / metrics.total_ep[-1]
        first_law_errors[index] = error
        steady_identity_errors[index] = abs(
            steady_ep[index] - beta * steady_work[index]
        )
        if (
            metrics.total_ep[-1] < -2e-9
            or metrics.pmmc[-1] - metrics.total_ep[-1] > 2e-8
        ):
            raise AssertionError("periodic mismatch exceeds physical entropy production")

    if float(steady_identity_errors.max()) > 2e-9:
        raise AssertionError("steady one-bath identity Sigma = beta W failed")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    selected_indices = sorted({0, periods.size // 2, periods.size - 1})
    for period_index in selected_indices:
        period = float(periods[period_index])
        plus_model, cycle, _, _ = make_alternating_field_cycle(
            n_spins, J, beta, h0, period
        )
        minus_model = build_sector_model(n_spins, J, -h0, [beta], [1.0])
        periodic_state = stationary_distribution(cycle.G, continuous=False)
        stage_times = np.linspace(0.0, period / 2.0, 36)
        plus_states = np.array(
            [expm(plus_model.generator * time) @ periodic_state for time in stage_times]
        )
        midpoint_state = plus_states[-1]
        minus_states = np.array(
            [expm(minus_model.generator * time) @ midpoint_state for time in stage_times]
        )
        plus_m = plus_states @ plus_model.magnetization
        minus_m = minus_states @ minus_model.magnetization
        field_path = np.concatenate(
            [
                np.full(stage_times.size, h0),
                [h0, -h0],
                np.full(stage_times.size, -h0),
                [-h0, h0],
            ]
        )
        magnetization_path = np.concatenate(
            [
                plus_m,
                [plus_m[-1], plus_m[-1]],
                minus_m,
                [minus_m[-1], minus_m[-1]],
            ]
        )
        axes[0, 0].plot(
            field_path,
            magnetization_path,
            lw=1.8,
            label=fr"$T={period:.3g}$",
        )
    axes[0, 0].set(
        xlabel="field $h$",
        ylabel=r"magnetization $\langle m\rangle$",
        title="Periodic-steady hysteresis loops",
    )
    axes[0, 0].legend(frameon=False, fontsize=8)

    axes[0, 1].semilogx(
        periods,
        steady_work / n_spins,
        "o-",
        ms=4,
        label="work per cycle",
    )
    axes[0, 1].semilogx(
        periods,
        steady_ep / (beta * n_spins),
        "--",
        label=r"$\Sigma_{ss}/\beta$",
    )
    axes[0, 1].set(ylabel="energy per spin", title="Hysteretic work")
    axes[0, 1].tick_params(labelbottom=False)
    axes[0, 1].legend(frameon=False, fontsize=8)

    axes[1, 0].semilogx(periods, final_pmmc, "o-", ms=4)
    axes[1, 0].set(
        ylabel=fr"$M_{{{n_observation_cycles}}}^*$ (nats)",
        title="Floquet POP after repeated cycles",
    )

    axes[1, 1].semilogx(periods, final_fraction, "o-", ms=4)
    axes[1, 1].set(
        ylabel=r"$M_N^*/\Sigma_N$",
        title="Prior-independent fraction of dissipation",
        ylim=(-0.03, 1.03),
    )

    fig.supxlabel("drive period")
    fig.suptitle(
        fr"Square-wave field cycle: $n_s={n_spins}$, $\beta J={beta * J:.2f}$, $h_0={h0}$",
        fontsize=13,
    )
    path = _finish_figure(fig, output_dir / "periodic_field_POP.png")
    return path, {
        "periods": periods,
        "steady_work": steady_work,
        "steady_ep": steady_ep,
        "max_first_law_error": float(first_law_errors.max()),
        "max_steady_identity_error": float(steady_identity_errors.max()),
    }


def generate_all(output_dir: Path, *, quick: bool, only: str) -> list[Path]:
    """Generate selected analyses and print compact numerical summaries."""

    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    if only in {"all", "same-g"}:
        path, summary = plot_same_G_hidden_housekeeping(output_dir, quick=quick)
        generated.append(path)
        print(
            "same-G:",
            f"steady EP rate={summary['steady_ep_rate']:.6f},",
            f"crossover={summary['crossover_time']:.6f}",
        )

    if only in {"all", "symmetry"}:
        path, summary = plot_symmetry_bit(output_dir, quick=quick)
        generated.append(path)
        smallest = min(summary)
        print(
            "symmetry bit:",
            f"D(p+||pi)={summary[smallest]['initial_kl']:.12f},",
            f"log(2)={np.log(2.0):.12f}",
        )

    if only in {"all", "critical"}:
        path, summary = plot_critical_scaling(output_dir, quick=quick)
        generated.append(path)
        print("critical scan:", f"J_c={summary['critical_J']:.6f}")

    if only in {"all", "periodic"}:
        path, summary = plot_periodic_drive(output_dir, quick=quick)
        generated.append(path)
        print(
            "periodic drive:",
            f"max first-law error={summary['max_first_law_error']:.3e}",
        )

    return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ising_pop_physics_figures"),
        help="directory for generated PNG and PDF figures",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use smaller parameter sweeps for a fast smoke run",
    )
    parser.add_argument(
        "--only",
        choices=("all", "same-g", "symmetry", "critical", "periodic"),
        default="all",
        help="generate only one analysis (default: all)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="skip the reduced-versus-microstate consistency check",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_validation:
        errors = validate_sector_reduction()
        print(
            "sector validation:",
            f"generator={errors['generator_error']:.3e},",
            f"heat={errors['heat_error']:.3e},",
            f"POP={errors['pmmc_error']:.3e}",
        )

    paths = generate_all(args.output_dir, quick=args.quick, only=args.only)
    for path in paths:
        print(path.resolve())


if __name__ == "__main__":
    main()
