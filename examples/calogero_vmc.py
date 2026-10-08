# examples/calogero_vmc.py
r"""Determinant-free VMC on Calogero-Sutherland, the benchmark whose answer lies in the ansatz.

The exact ground state is a power of the Vandermonde times a Gaussian, which is exactly
"Vandermonde times symmetric" -- the form ``physics.ansatz1d`` parametrizes. So this is not an
approximation test: the optimizer should reach the analytic energy, and failure means a bug.

Usage::

  python examples/calogero_vmc.py
  python examples/calogero_vmc.py --n 5 --lam 1.5 --steps 2000

Run from the repository root.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from physics import ansatz1d, models1d, nodal, vmc


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--n", type=int, default=4, help="number of fermions")
  ap.add_argument("--omega", type=float, default=1.0, help="trap frequency")
  ap.add_argument("--lam", type=float, default=2.0, help="Calogero coupling")
  ap.add_argument("--walkers", type=int, default=512)
  ap.add_argument("--steps", type=int, default=1000)
  ap.add_argument("--sweeps", type=int, default=4)
  ap.add_argument("--step-size", type=float, default=0.25)
  ap.add_argument("--lr", type=float, default=5e-3)
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args()

  n, omega, lam = args.n, args.omega, args.lam
  e_exact = models1d.calogero_energy(n, omega, lam)
  potential = models1d.calogero_potential(omega, lam)

  # 1. The cusp condition fixes the Vandermonde exponent. It is not a hyperparameter.
  alpha = float(ansatz1d.cusp_alpha(lam * (lam - 1.0)))

  # 2. Check the exact state is zero-variance through this same local-energy code. If this
  #    fails, nothing downstream is trustworthy.
  exact_log_psi = models1d.calogero_log_psi(omega, lam)
  key = jax.random.PRNGKey(args.seed)
  key, sub = jax.random.split(key)
  probe = jnp.sort(jax.random.normal(sub, (64, n, 1)) * 1.2, axis=1)
  e_probe = vmc.batched_local_energy(exact_log_psi, None, probe, potential)
  print(f"exact state    E = {float(jnp.mean(e_probe)):.10f}   "
        f"sigma = {float(jnp.std(e_probe)):.2e}   (analytic {e_exact:.10f})")

  # 3. Confirm the trial function is exactly antisymmetric, via physics.nodal.
  key, sub = jax.random.split(key)
  params = ansatz1d.init_params(sub, n, alpha=alpha, envelope=0.6 * omega)
  psi = ansatz1d.wavefunction(params)
  n_err, max_err = nodal.antisymmetry_error(psi, probe, jnp.zeros(n, dtype=int))
  print(f"antisymmetry   sign errors = {int(n_err)}   max log error = {float(max_err):.2e}")

  # 4. Optimize. alpha is frozen; the envelope starts wrong on purpose.
  key, sub = jax.random.split(key)
  R = ansatz1d.init_walkers(sub, args.walkers, n, spread=1.2)

  print(f"\nn = {n}, omega = {omega}, lambda = {lam}, alpha = {alpha:.6f}")
  print(f"exact E0 = {e_exact:.6f}\n")
  print(f"{'step':>6} {'E':>13} {'err':>11} {'var':>11} {'acc':>6}")

  def report(row):
    i, energy, variance, acc = row
    if i % max(1, args.steps // 25) == 0:
      print(f"{i:6d} {energy:13.6f} {energy - e_exact:+11.2e} {variance:11.3e} {acc:6.3f}")

  t0 = time.time()
  key, sub = jax.random.split(key)
  params, R, history = vmc.optimize(
    ansatz1d.log_psi, params, R, sub, potential,
    n_steps=args.steps, n_sweeps=args.sweeps, step=args.step_size,
    lr=args.lr, frozen=("log_alpha",), callback=report,
  )

  tail = history[-50:]
  energy = sum(row[1] for row in tail) / len(tail)
  variance = sum(row[2] for row in tail) / len(tail)
  print(f"\nwall time    {time.time() - t0:.1f}s")
  print(f"final E      {energy:.6f}   error {energy - e_exact:+.3e}")
  print(f"final var    {variance:.3e}")
  print(f"alpha        {float(jnp.exp(params['log_alpha'])):.6f}   (frozen at the cusp value)")
  print(f"envelope     {float(jnp.exp(params['log_envelope'])):.6f}   (started at {0.6 * omega})")
  print("\nThe envelope need not converge to omega: the Jastrow and the symmetric network\n"
        "absorb part of the Gaussian, so the split between them is not unique. The energy\n"
        "and the variance are the things with physical content.")


if __name__ == "__main__":
  main()
