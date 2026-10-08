# examples/ising_pop_physics.py
r"""Physical analyses of periodic-optimal-prior (POP) mismatch in the Curie-Weiss model.

Generates four figure sets from ``physics.ising_sectors`` (numerics and consistency checks) drawn by
``plotting.ising_pop``:

1. Identical state dynamics implemented by one effective bath or two physical baths, separating
   transient information loss from hidden housekeeping EP.
2. Forgetting a single symmetry-breaking bit, for which the asymptotic POP mismatch is exactly
   ``log(2)``.
3. Critical slowing of POP saturation across the Curie-Weiss transition.
4. A genuinely periodic square-wave field protocol and its Floquet POP bound.

Usage::

  python examples/ising_pop_physics.py --quick --output-dir /tmp/ising_pop_physics
  python examples/ising_pop_physics.py --only critical

Omit ``--quick`` for the denser publication-oriented sweeps. Run from the repository root.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from physics import ising_sectors as sectors
from plotting import ising_pop


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
  r"""Generate the selected analyses and print compact numerical summaries."""
  output_dir.mkdir(parents=True, exist_ok=True)
  generated: list[Path] = []

  if only in {"all", "same-g"}:
    result = sectors.hidden_housekeeping(
      n_cycles=120 if quick else 500,
      contrasts=[0.0, 0.1, 0.2, 0.35, 0.7, 1.2, 2.0] if quick else np.linspace(0.0, 2.0, 21),
    )
    generated.append(
      save_figure(ising_pop.plot_same_G_hidden_housekeeping(result), output_dir / "same_G_hidden_housekeeping.png")
    )
    print(
      "same-G:",
      f"steady EP rate={result.steady_ep_rate:.6f},",
      f"crossover={result.crossover_time:.6f}",
    )

  if only in {"all", "symmetry"}:
    runs = [
      sectors.symmetry_bit_relaxation(n_spins, n_cycles=600 if quick else 3000)
      for n_spins in ([9, 15] if quick else [9, 15, 21, 31])
    ]
    generated.append(save_figure(ising_pop.plot_symmetry_bit(runs), output_dir / "symmetry_bit_forgetting.png"))
    print(
      "symmetry bit:",
      f"D(p+||pi)={runs[0].initial_kl:.12f},",
      f"log(2)={np.log(2.0):.12f}",
    )

  if only in {"all", "critical"}:
    scan = sectors.critical_scan(
      n_spins=21 if quick else 51,
      J_values=np.linspace(0.25, 0.75, 9 if quick else 21),
      max_time=20.0 if quick else 100.0,
      horizons=[0.5, 2.0, 5.0, 20.0] if quick else [1.0, 5.0, 20.0, 100.0],
    )
    generated.append(save_figure(ising_pop.plot_critical_scaling(scan), output_dir / "critical_POP_scaling.png"))
    print("critical scan:", f"J_c={scan.critical_J:.6f}")

  if only in {"all", "periodic"}:
    drive = sectors.periodic_drive_scan(
      n_spins=15 if quick else 31,
      periods=np.logspace(-1.0, 1.4, 11) if quick else np.logspace(-1.3, 2.0, 28),
      n_observation_cycles=30 if quick else 100,
    )
    generated.append(save_figure(ising_pop.plot_periodic_drive(drive), output_dir / "periodic_field_POP.png"))
    print("periodic drive:", f"max first-law error={drive.summary()['max_first_law_error']:.3e}")

  return generated


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("ising_pop_physics_figures"),
    help="directory for generated PNG and PDF figures",
  )
  parser.add_argument("--quick", action="store_true", help="use smaller parameter sweeps for a fast smoke run")
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
    errors = sectors.validate_sector_reduction()
    print(
      "sector validation:",
      f"generator={errors['generator_error']:.3e},",
      f"heat={errors['heat_error']:.3e},",
      f"POP={errors['pmmc_error']:.3e}",
    )

  for path in generate_all(args.output_dir, quick=args.quick, only=args.only):
    print(path.resolve())


if __name__ == "__main__":
  main()
