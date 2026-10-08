# plotting/ising_pop.py
r"""Figures for ``physics.ising_sectors``: periodic-optimal-prior (POP) mismatch in the Curie-Weiss model.

One function per analysis, each taking the dataclass the matching ``physics.ising_sectors``
function returns and giving back a ``Figure``. The numerics, including every consistency check, are
computed there; this module draws them, and ``physics`` does not import it. Save with
``plotting_utils.save_path`` or ``fig.savefig``.
"""
from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from physics.ising_sectors import CriticalScan, HiddenHousekeeping, PeriodicDriveScan, SymmetryBitRun


def plot_same_G_hidden_housekeeping(result: HiddenHousekeeping) -> Figure:
  r"""One- versus two-bath EP for identical state dynamics, hidden heat conduction, and its scaling."""
  r = result
  metrics, times, n_spins = r.metrics, r.times, r.n_spins
  fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
  ax = axes[0, 0]
  ax.plot(times, metrics.pmmc, label=r"POP $M_N^*$", lw=2.2)
  ax.plot(times, r.matched_one_bath_ep, label="matched one-bath EP", lw=1.9)
  ax.plot(times, metrics.total_ep, label="physical two-bath EP", lw=1.9)
  ax.axhline(r.information_budget, color="0.35", ls=":", label=r"$D(p_0\Vert\pi)$")
  ax.set(xlabel="time", ylabel="entropy (nats)", title="Same state dynamics, different EP")
  ax.legend(frameon=False, fontsize=8)

  ax = axes[0, 1]
  ax.plot(times, r.hidden_ep, label="hidden channel EP", lw=2.2)
  ax.plot(times, r.steady_ep_rate * times, "--", label=r"$\dot\Sigma_{ss}t$")
  ax.set(xlabel="time", ylabel="entropy (nats)", title="Recurring housekeeping contribution")
  ax.legend(frameon=False)

  ax = axes[1, 0]
  ax.plot(r.contrasts, r.currents[:, 0] / n_spins, marker="o", ms=3, label=r"$\dot Q_1$")
  ax.plot(r.contrasts, r.currents[:, 1] / n_spins, marker="o", ms=3, label=r"$\dot Q_2$")
  ax.axhline(0.0, color="0.3", lw=0.8)
  ax.set(
    xlabel=r"bath contrast $\Delta\beta$",
    ylabel="steady heat current per spin",
    title="Hidden heat conduction",
  )
  ax.legend(frameon=False)

  ax = axes[1, 1]
  ax.plot(r.contrasts, r.ep_rates / n_spins, "o", ms=4, label="computed")
  ax.plot(
    r.contrasts,
    r.quadratic_coefficient * r.contrasts**2 / n_spins,
    "--",
    label=fr"${r.quadratic_coefficient / n_spins:.4f}(\Delta\beta)^2$",
  )
  ax.set(
    xlabel=r"bath contrast $\Delta\beta$",
    ylabel=r"$\dot\Sigma_{ss}/n_s$",
    title="Near-equilibrium quadratic scaling",
  )
  ax.legend(frameon=False)

  fig.suptitle(
    fr"Hidden housekeeping for $n_s={n_spins}$, $J={r.J}$, $\bar\beta={r.beta_bar}$",
    fontsize=13,
  )
  return fig


def plot_symmetry_bit(runs: Sequence[SymmetryBitRun]) -> Figure:
  r"""POP cost and magnetization-sign memory while one symmetry bit is forgotten, per system size."""
  fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
  colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
  for index, run in enumerate(runs):
    color = colors[index % len(colors)]
    axes[0].semilogx(
      run.times, run.metrics.pmmc / np.log(2.0), color=color, lw=2, label=fr"$n_s={run.n_spins}$"
    )
    axes[0].semilogx(
      run.times, run.metrics.total_ep / np.log(2.0), color=color, lw=1.2, ls="--", alpha=0.8
    )
    axes[1].semilogx(
      run.times, run.positive_mass[1:], color=color, lw=2, label=fr"$n_s={run.n_spins}$"
    )

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
  first = runs[0]
  fig.suptitle(fr"Symmetry-bit relaxation at $\beta J={first.beta * first.J:.2f}$", fontsize=13)
  return fig


def plot_critical_scaling(scan: CriticalScan) -> Figure:
  r"""Critical slowing, finite-horizon POP saturation, ordering crossover and housekeeping."""
  s = scan
  fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
  for ax in axes.flat:
    ax.axvline(s.critical_J, color="0.3", ls=":", lw=1)

  axes[0, 0].semilogy(s.J_values, s.relaxation_times, "o-", ms=4)
  axes[0, 0].set(xlabel=r"coupling $J$", ylabel="relaxation time", title="Critical slowing")

  for horizon, values in zip(s.horizons, s.fractions):
    axes[0, 1].plot(s.J_values, values, marker="o", ms=3, label=fr"$T={horizon:g}$")
  axes[0, 1].set(
    xlabel=r"coupling $J$",
    ylabel=r"$M_T^*/D(p_0\Vert\pi)$",
    title="Finite-horizon POP saturation",
    ylim=(-0.03, 1.03),
  )
  axes[0, 1].legend(frameon=False, fontsize=8)

  axes[1, 0].plot(s.J_values, s.abs_magnetization, "o-", ms=4, label=r"$\langle|m|\rangle$")
  susceptibility_scaled = s.susceptibility / max(1.0, s.susceptibility.max())
  axes[1, 0].plot(s.J_values, susceptibility_scaled, "s--", ms=3, label="susceptibility (scaled)")
  axes[1, 0].set(xlabel=r"coupling $J$", ylabel="order/response", title="Finite-size ordering crossover")
  axes[1, 0].legend(frameon=False, fontsize=8)

  finite = np.isfinite(s.t50)
  if np.any(finite):
    recurring_cost = s.steady_ep_rate[finite] * s.t50[finite] / s.information_budget[finite]
    axes[1, 1].plot(s.J_values[finite], recurring_cost, "o-", ms=4)
  axes[1, 1].set(
    xlabel=r"coupling $J$",
    ylabel=r"$\dot\Sigma_{ss}t_{50}/D(p_0\Vert\pi)$",
    title="Housekeeping accrued by half-saturation",
  )

  fig.suptitle(
    fr"Two-bath Curie-Weiss crossover, $n_s={s.n_spins}$, $J_c={s.critical_J:.2f}$",
    fontsize=13,
  )
  return fig


def plot_periodic_drive(scan: PeriodicDriveScan) -> Figure:
  r"""Hysteresis loops, work and dissipation, and Floquet POP against the drive period."""
  s = scan
  fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
  for period, (field_path, magnetization_path) in s.loops.items():
    axes[0, 0].plot(field_path, magnetization_path, lw=1.8, label=fr"$T={period:.3g}$")
  axes[0, 0].set(
    xlabel="field $h$",
    ylabel=r"magnetization $\langle m\rangle$",
    title="Periodic-steady hysteresis loops",
  )
  axes[0, 0].legend(frameon=False, fontsize=8)

  axes[0, 1].semilogx(s.periods, s.steady_work / s.n_spins, "o-", ms=4, label="work per cycle")
  axes[0, 1].semilogx(s.periods, s.steady_ep / (s.beta * s.n_spins), "--", label=r"$\Sigma_{ss}/\beta$")
  axes[0, 1].set(ylabel="energy per spin", title="Hysteretic work")
  axes[0, 1].tick_params(labelbottom=False)
  axes[0, 1].legend(frameon=False, fontsize=8)

  axes[1, 0].semilogx(s.periods, s.final_pmmc, "o-", ms=4)
  axes[1, 0].set(
    ylabel=fr"$M_{{{s.n_observation_cycles}}}^*$ (nats)",
    title="Floquet POP after repeated cycles",
  )

  axes[1, 1].semilogx(s.periods, s.final_fraction, "o-", ms=4)
  axes[1, 1].set(
    ylabel=r"$M_N^*/\Sigma_N$",
    title="Prior-independent fraction of dissipation",
    ylim=(-0.03, 1.03),
  )

  fig.supxlabel("drive period")
  fig.suptitle(
    fr"Square-wave field cycle: $n_s={s.n_spins}$, $\beta J={s.beta * s.J:.2f}$, $h_0={s.h0}$",
    fontsize=13,
  )
  return fig
