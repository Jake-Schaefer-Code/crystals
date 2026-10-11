# examples/clock_information_figures.py
r"""Clock information and the time-averaged mismatch cost of time-homogeneous relaxation.

Generates six figure sets from ``physics.clock_information`` (numerics and identity checks) drawn
by ``plotting.clock_information``:

1. ``spacetime``: the space-time density of an Ornstein-Uhlenbeck relaxation and its log-ratio to the
   time average, with the nodal set ``p_t = qbar``.
2. ``decay``: clock information ``I(tau; X_{tau+s})`` erased by the dynamics; its slope at ``s = 0``
   is minus the minimal cost rate.
3. ``window``: minimal total mismatch cost against window length, far from and near equilibrium, with
   the near-equilibrium prediction ``2 g(aT)``.
4. ``blur``: Gaussian clock blur (heat flow in time) of the trajectory and what survives of the bound.
   The weighted log-ratio (top) and the log-ratio (bottom) share their nodal set, drawn with its
   heavy-blur prediction: the lines ``t = T/2`` and ``x = x*`` opened into hyperbolas.
5. ``ring``: a driven 10-state ring against its reversible twin, ringing versus overdamped decay.
6. ``hamiltonian``: the dynamical Hamiltonian ``H_dyn = (d_t - K)^dagger (d_t - K)`` of four rings (uniform and in a
   potential, each reversible and driven): its levels against the separable formula, and the first eigenfunction
   with state nodes, product-form and crossing for reversible ``K``, tilted for a uniform driven ring, reconnected
   for a driven ring in a potential.

Usage::

  python examples/clock_information_figures.py --quick --output-dir /tmp/clock_information
  python examples/clock_information_figures.py --only ring

Run from the repository root.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from physics import clock_information as ci
from plotting import clock_information as plot_ci

CHOICES = ("all", "spacetime", "decay", "window", "blur", "ring", "hamiltonian")


def save_figure(fig: plt.Figure, output_path: Path) -> Path:
  r"""Write ``output_path`` (PNG) and a PDF beside it, and close the figure."""
  output_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(output_path, dpi=240, bbox_inches="tight")
  fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
  plt.close(fig)
  if not output_path.exists() or output_path.stat().st_size == 0:
    raise RuntimeError(f"figure was not written: {output_path}")
  return output_path


def generate_all(output_dir: Path, *, quick: bool, only: str) -> list[Path]:
  r"""Generate the selected figures and print compact numerical summaries."""
  xs = jnp.linspace(-7.0, 7.0, 701 if quick else 1401)
  ou = dict(a=1.0, m0=2.0, v0=0.25)
  generated: list[Path] = []

  if only in {"all", "spacetime"}:
    r = ci.ou_space_time(T=3.0, n_t=151 if quick else 301, xs=xs, **ou)
    generated.append(save_figure(plot_ci.plot_space_time(r), output_dir / "spacetime.png"))
    print(f"spacetime: I(tau;X) = {r.info:.4f}, MC*_T = {r.mmc:.4f}")

  if only in {"all", "decay"}:
    r = ci.ou_clock_decay(np.linspace(0.0, 3.0, 31), T=3.0, n_t=151 if quick else 301, xs=xs, **ou)
    generated.append(save_figure(plot_ci.plot_clock_decay(r), output_dir / "clock_decay.png"))
    print(f"decay: minimal cost rate {r.rate:.5f} (boundary) vs {r.fd_rate:.5f} (finite difference)")

  if only in {"all", "window"}:
    Ts = np.array([0.5, 1, 2, 3, 5, 8, 12, 20, 30]) if quick else np.concatenate(
      [np.linspace(0.1, 2.0, 20), np.linspace(2.5, 30.0, 40)])
    n_t = 601 if quick else 1201
    far = ci.ou_window_scan(Ts, xs=xs, n_t=n_t, **ou)
    near = ci.ou_window_scan(Ts, xs=xs, n_t=n_t, a=1.0, m0=0.3, v0=1.0)
    prediction = ("near-equilibrium prediction $2g(aT)$", Ts, 2.0 * np.asarray(ci.mode_filter_uniform(Ts)))
    fig = plot_ci.plot_window_scans(
      [(r"far: $p_0=N(2,\,0.25)$", far), (r"near: $p_0=N(0.3,\,1)$", near)], prediction=prediction)
    generated.append(save_figure(fig, output_dir / "window_scan.png"))
    print(f"window: at T = {Ts[-1]:g}, far {far.mmc[-1] / far.budget:.4f}, near {near.mmc[-1] / near.budget:.4f}")

  if only in {"all", "blur"}:
    sigmas = np.linspace(0.0, 2.5, 26 if quick else 51)
    r = ci.ou_blur_scan(sigmas, T=3.0, n_t=200 if quick else 400, xs=jnp.linspace(-7.0, 7.0, 701),
                        display_sigmas=(0.0, 0.3, 0.8, 1.5, 2.0), **ou)
    generated.append(save_figure(plot_ci.plot_blur(r), output_dir / "clock_blur.png"))
    print(f"blur: at sigma = {sigmas[-1]:g}, I retained {r.info[-1] / r.info[0]:.3f}, M retained {r.mmc[-1] / r.mmc[0]:.3f}")
    print(f"blur: heavy-blur cross at t = T/2 and x* = {r.cross.x_star[0]:.4f}, hyperbola constant c = {r.cross.c[0]:.4f}")

  if only in {"all", "ring"}:
    Ts = (1, 2, 4, 8, 15, 30, 60) if quick else (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 60)
    r = ci.ring_comparison(10, 1.0, 0.01, Ts=Ts)
    generated.append(save_figure(plot_ci.plot_ring(r), output_dir / "ring.png"))
    lam = r.slowest[0]
    print(f"ring: slowest mode {lam.real:.4f} +- {abs(lam.imag):.4f}i, |Im/Re| on the coherence bound {r.coherence:.4f}")

  if only in {"all", "hamiltonian"}:
    r = ci.ring_hamiltonian(10, 1.0, 0.01, T=8.0, n_t=81 if quick else 161)
    generated.append(save_figure(plot_ci.plot_ring_hamiltonian(r), output_dir / "dynamical_hamiltonian.png"))
    for s in r.systems:
      dev = np.max(np.abs(s.energies[10:] - s.normal[10:]) / s.normal[10:])
      print(f"hamiltonian: {s.label:22s} levels off the separable formula by {dev:.1e}, first state E = {s.level:.4f}, "
            f"s2/s1 = {s.rank_residual:.3f}, saddle gap {s.gap:.1e}, trajectory E = {s.trajectory:.1e}")

  return generated


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  parser.add_argument("--quick", action="store_true", help="coarser grids and fewer windows")
  parser.add_argument("--output-dir", type=Path, default=Path("examples/out/clock_information"))
  parser.add_argument("--only", choices=CHOICES, default="all")
  args = parser.parse_args()
  for path in generate_all(args.output_dir, quick=args.quick, only=args.only):
    print(f"wrote {path}")


if __name__ == "__main__":
  main()
