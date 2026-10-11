"""Nodes in a crystal: how lattice structure moves the nodal surface.

Spin-polarized electrons in a periodic 2D box with a triangular lattice of Gaussian wells of
depth ``V0``. The one-body Hamiltonian is diagonalized in a real plane-wave basis, the lowest
``N`` Bloch orbitals fill one Slater determinant, and ``physics.nodal`` draws the node seen by
electron 0 with the others frozen.

``V0 = 0`` is the homogeneous electron gas (plane-wave node). Turning the lattice on deforms the
same node continuously; deep wells localize the orbitals and the node follows the Wannier
functions. The GIF moves one frozen electron along a lattice direction and shows the node
seen by electron 0 moving with it: a single-particle cut of a joint object.

Run from the repo root::

  python examples/crystal_nodes.py --quick          # ~2 min on CPU
  python examples/crystal_nodes.py                  # finer grids, census, GIF

Outputs go to ``examples/out/crystal_nodes/``.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as onp

root = next(p for p in [pathlib.Path(__file__).resolve().parent, *pathlib.Path(__file__).resolve().parents]
            if (p / "physics" / "nodal.py").exists())
sys.path.insert(0, str(root))

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

jax.config.update("jax_enable_x64", True)

import physics.nodal as nodal
import physics.nodal_models as models
import plotting.nodal as plot_nodal
from src.lattice_geometry import Bravais2D, canonical_basis_2d


# --------------------------------------------------------------------------- #
# Crystal: triangular lattice of wells in a commensurate rectangular periodic box
# --------------------------------------------------------------------------- #

def triangular_supercell(n1: int, n2: int, a: float = 1.0):
  r"""Sites of a triangular lattice (``src.lattice_geometry`` basis) tiled into a rectangular box.

  The rectangular cell ``(a, sqrt(3) a)`` holds two sites, ``0`` and ``a1/2 + a2/2``... in the
  basis ``a1 = a(1, 0)``, ``a2 = a(1/2, sqrt(3)/2)`` the second site is ``a2 - a1/2 + a1/2``, i.e.
  ``(a/2, sqrt(3)a/2)``. Returns ``(sites, L)`` with ``L = (n1 a, n2 sqrt(3) a)``.
  """
  A = canonical_basis_2d(Bravais2D.TRIANGULAR, scale=a)
  a1, a2 = A[:, 0], A[:, 1]
  rect = onp.array([n1 * a1[0], n2 * (2 * a2[1])])        # (n1 a, n2 sqrt3 a)
  motif = onp.array([[0.0, 0.0], a2])                      # two sites per rectangular cell
  sites = onp.array([m + onp.array([i * a1[0], j * 2 * a2[1]])
                     for i in range(n1) for j in range(n2) for m in motif])
  return sites, rect


def well_potential(sites, L, V0: float, sigma: float):
  r"""Periodic ``V(r) = -V0 sum_sites sum_images exp(-|r - s|^2 / 2 sigma^2)`` as a JAX function."""
  images = onp.array([[i, j] for i in (-1, 0, 1) for j in (-1, 0, 1)]) * L
  centers = jnp.asarray((sites[:, None, :] + images[None]).reshape(-1, 2))

  def V(r):                                                 # (..., 2) -> (...)
    d2 = jnp.sum((r[..., None, :] - centers) ** 2, axis=-1)
    return -V0 * jnp.sum(jnp.exp(-0.5 * d2 / sigma ** 2), axis=-1)

  return V


# --------------------------------------------------------------------------- #
# Bloch orbitals of the supercell in a real plane-wave basis
# --------------------------------------------------------------------------- #

def real_plane_waves(L, e_cut: float):
  r"""Real orthonormal plane waves ``1, sqrt2 cos(G.r), sqrt2 sin(G.r)`` on the box, ``|G|^2/2 <= e_cut``.

  Returns ``(chi, G)`` with ``chi(r) -> (..., n_basis)`` and the wavevectors used (one per
  cos/sin pair).
  """
  n_max = int(onp.ceil(onp.sqrt(2 * e_cut) * L.max() / (2 * onp.pi))) + 1
  Gs, kinds = [], []
  for nx in range(-n_max, n_max + 1):
    for ny in range(-n_max, n_max + 1):
      G = 2 * onp.pi * onp.array([nx / L[0], ny / L[1]])
      if 0.5 * G @ G > e_cut:
        continue
      if (nx, ny) == (0, 0):
        Gs.append(G); kinds.append(0)
      elif nx > 0 or (nx == 0 and ny > 0):                 # one representative of each +-G pair
        Gs.append(G); kinds.append(1)
        Gs.append(G); kinds.append(2)
  G = jnp.asarray(onp.array(Gs))
  kinds = jnp.asarray(onp.array(kinds))
  area = float(L[0] * L[1])

  def chi(r):
    phase = r @ G.T
    out = jnp.where(kinds == 0, 1.0, jnp.where(kinds == 1, jnp.sqrt(2.0) * jnp.cos(phase),
                                              jnp.sqrt(2.0) * jnp.sin(phase)))
    return out / jnp.sqrt(area)

  return chi, onp.array(Gs), onp.array(kinds)


def bloch_orbitals(sites, L, V0, sigma, e_cut=60.0, n_grid=128):
  r"""Diagonalize ``-Laplacian/2 + V`` in the real plane-wave basis; return ``(orbitals, eps)``.

  ``orbitals(r) -> (m, n_basis)`` evaluates every eigenfunction, lowest energy first, so
  ``models.slater_determinant`` can take its first ``N`` columns.
  """
  chi, G, kinds = real_plane_waves(L, e_cut)
  V = well_potential(sites, L, V0, sigma)
  x = onp.linspace(0, L[0], n_grid, endpoint=False)
  y = onp.linspace(0, L[1], n_grid, endpoint=False)
  r = jnp.asarray(onp.stack(onp.meshgrid(x, y, indexing="ij"), -1).reshape(-1, 2))
  X = chi(r)                                                # (n_pts, n_basis)
  w = float(L[0] * L[1]) / r.shape[0]                       # periodic trapezoid weight
  Vmat = (X.T * V(r)) @ X * w                               # exact for trig polynomials under cutoff
  T = jnp.diag(jnp.asarray(0.5 * onp.sum(G * G, axis=1)))
  eps, C = jnp.linalg.eigh(T + Vmat)

  def orbitals(rr):
    return chi(rr) @ C

  return orbitals, onp.asarray(eps)


def pick_filling(eps_by_v0, candidates=(5, 7, 9)):
  """Choose N with the largest minimal one-body gap across the V0 sweep."""
  best = max(candidates, key=lambda N: min(float(e[N] - e[N - 1]) for e in eps_by_v0))
  return best


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def sweep_figure(psis, labels, R, spins, sites, L, extent, n):
  fig, axs = plt.subplots(1, len(psis), figsize=(4.6 * len(psis), 4.4), layout="constrained")
  flips = []
  prev = None
  for ax, psi, label in zip(onp.atleast_1d(axs), psis, labels):
    u, v, s, l = plot_nodal.plot_particle_slice(ax, psi, R, 0, spins, extent=extent, n=n, gamma=0.35)
    ax.scatter(sites[:, 0], sites[:, 1], s=14, marker="+", c="limegreen", zorder=2, label="wells")
    if prev is not None:
      flips.append(float(onp.mean(s != prev)))
    prev = s
    ax.set_title(label)
    ax.set_aspect("equal")
  onp.atleast_1d(axs)[0].legend(loc="upper right", fontsize=7)
  return fig, flips


def node_gif(psi, R, spins, moving, path, extent, n, outfile, fps=8):
  r"""Frames of the node seen by electron 0 while frozen electron ``moving`` walks along ``path``."""
  R = onp.asarray(R, dtype=float).copy()
  spins = onp.asarray(spins)
  fig, ax = plt.subplots(figsize=(5.2, 4.6))
  u0, u1, v0, v1 = extent
  same = (onp.arange(len(R)) != 0) & (spins == spins[0])

  def frame(k):
    Rk = R.copy()
    Rk[moving] = path[k]
    return Rk, nodal.slice_grid(psi, *nodal.coordinate_plane(Rk, (0, 0), (0, 1)), extent, n)

  Rk, (u, v, s, l) = frame(0)
  im = plot_nodal.plot_signed(ax, u, v, s, l, gamma=0.35)
  fig.colorbar(im, ax=ax, shrink=0.8, label=r"sign$(\psi)\,|\psi|^{0.35}$")
  pts = ax.scatter(Rk[same, 0], Rk[same, 1], s=40, c="k", zorder=3)
  star = ax.scatter(Rk[0, 0], Rk[0, 1], s=180, marker="*", c="gold", edgecolors="k", zorder=4)
  ax.plot(path[:, 0], path[:, 1], "w--", lw=0.8, alpha=0.7)
  ax.set(xlim=(u0, u1), ylim=(v0, v1), xlabel="$r_0 \\cdot$ x", ylabel="$r_0 \\cdot$ y", aspect="equal")

  def draw(k):
    Rk, (u, v, s, l) = frame(k)
    ax.cla()
    plot_nodal.plot_signed(ax, u, v, s, l, gamma=0.35)
    ax.scatter(Rk[same, 0], Rk[same, 1], s=40, c="k", zorder=3)
    ax.scatter(Rk[0, 0], Rk[0, 1], s=180, marker="*", c="gold", edgecolors="k", zorder=4)
    ax.plot(path[:, 0], path[:, 1], "w--", lw=0.8, alpha=0.7)
    ax.set(xlim=(u0, u1), ylim=(v0, v1), xlabel="$r_0 \\cdot$ x", ylabel="$r_0 \\cdot$ y", aspect="equal",
           title=f"frozen electron {moving} at ({path[k, 0]:.2f}, {path[k, 1]:.2f})")
    return []

  anim = FuncAnimation(fig, draw, frames=len(path), interval=1000 / fps, blit=False)
  anim.save(outfile, writer=PillowWriter(fps=fps))
  plt.close(fig)


# --------------------------------------------------------------------------- #

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--quick", action="store_true")
  ap.add_argument("--n1", type=int, default=3)
  ap.add_argument("--n2", type=int, default=2)
  ap.add_argument("--sigma", type=float, default=0.25)
  ap.add_argument("--v0", type=float, nargs="+", default=[0.0, 20.0, 60.0])
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args()
  n_slice = 121 if args.quick else 241
  out = root / "examples" / "out" / "crystal_nodes"
  out.mkdir(parents=True, exist_ok=True)

  sites, L = triangular_supercell(args.n1, args.n2)
  extent = (0.0, float(L[0]), 0.0, float(L[1]))
  print(f"{len(sites)} wells in a {L[0]:.2f} x {L[1]:.2f} periodic box")

  built = {v0: bloch_orbitals(sites, L, v0, args.sigma) for v0 in args.v0}
  N = pick_filling([eps for _, eps in built.values()])
  spins = [0] * N
  for v0, (_, eps) in built.items():
    gap = eps[N] - eps[N - 1]
    print(f"V0 = {v0:5.2f}: eps[0..{N}] = {onp.round(eps[:N + 1], 3)}  gap at N={N}: {gap:.3f}")
  psis = {v0: models.slater_determinant(orb, spins) for v0, (orb, _) in built.items()}

  # Frozen electrons: a |psi|^2 sample of the deepest lattice (electrons sit in wells), reused
  # for every V0 so the panels differ only through the orbitals.
  key = jax.random.PRNGKey(args.seed)
  k1, key = jax.random.split(key)
  R0 = jax.random.uniform(k1, (8, N, 2)) * jnp.asarray(L)
  samples, acc = nodal.metropolis(psis[args.v0[-1]], R0, k1, n_sweeps=200, step=0.2)
  R = onp.asarray(samples[-1, 0])
  print(f"Metropolis acceptance {acc:.2f}; frozen configuration sampled at V0={args.v0[-1]}")

  labels = [f"$V_0 = {v0:g}$" for v0 in args.v0]
  fig, flips = sweep_figure(list(psis.values()), labels, R, spins, sites, L, extent, n_slice)
  fig.suptitle(f"Node seen by electron 0 of {N} spin-polarized electrons, same frozen electrons")
  fig.savefig(out / "node_vs_V0.png", dpi=150)
  print("sign-flip fraction between successive V0 panels:", onp.round(flips, 3))

  # Quantify: node motion when one frozen electron moves by a lattice vector, per V0.
  print("\nnode motion when electron 1 moves by 0.2 a:")
  shift = onp.array([0.2, 0.0])
  for v0, psi in psis.items():
    R2 = R.copy(); R2[1] = (R2[1] + shift) % L
    _, _, s1, _ = nodal.slice_grid(psi, *nodal.coordinate_plane(R, (0, 0), (0, 1)), extent, n_slice)
    _, _, s2, _ = nodal.slice_grid(psi, *nodal.coordinate_plane(R2, (0, 0), (0, 1)), extent, n_slice)
    print(f"  V0 = {v0:5.2f}: fraction of the slice that changes sign = {onp.mean(s1 != s2):.3f}")

  if not args.quick:
    v0 = args.v0[-1]
    path = (R[1] + onp.linspace(0, 1, 24)[:, None] * onp.array([1.0, 0.0])) % L
    node_gif(psis[v0], R, spins, moving=1, path=path, extent=extent, n=121,
             outfile=out / f"node_motion_V0_{v0:g}.gif")
    print("GIF written")

    print("\ncell census (bound on the number of nodal cells; 2 = Mitas two-cell structure):")
    for v0, psi in psis.items():
      k1, key = jax.random.split(key)
      c = nodal.cell_census(psi, spins, R, k1, step=0.25, n_walkers=24, n_sweeps=120, n_paths=12)
      print(f"  V0 = {v0:5.2f}: |S|/|G| <= {c.group_order // c.stabilizer_order}  "
            f"(certified {len(c.certified)} exchanges, attempts {c.attempts})")

  print(f"\nfigures in {out}")


if __name__ == "__main__":
  main()
