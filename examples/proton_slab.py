"""Programming a spin Hamiltonian with protons: the continuum-hardness instance as a structure.

The October 2026 construction (OpenAI math repo, family 275) proves QMA-hardness of the
continuum Coulomb energy with unit-charge nuclei by *designing* a nuclear geometry:

1. a slab of uniform positive density -> harmonic confinement perpendicular to the slab
   (Poisson: Laplacian U = 4 pi rho, so U(z) = 2 pi rho z^2 + const inside);
2. localized extra density at chosen in-plane sites -> wells, one electron per well;
3. the continuous density replaced by point unit charges on a grid (a Gauss rule carried by a
   Moser transport), in coordinates dilated by a polynomial lambda so each proton couples as 1/lambda;
4. tunnelling between wells gives hopping t, double occupancy costs U, superexchange gives an
   antiferromagnetic Heisenberg model J = 4 t^2 / U on whatever graph the wells form.

This script builds that object at toy scale and shows each step as a figure:

* ``slab_structure.png``   the proton crystal (slab lattice + well clusters), the "molecule";
* ``slab_potential.png``   the one-body potential on the midplane, the harmonic profile U(z),
                           and the point-charge sum converging to the smooth design as the
                           grid spacing h shrinks (the Gauss-rule step);
* ``slab_modes.png``       the lowest midplane eigenstates localized on a triangle of wells, and
                           the hopping t falling exponentially with well spacing: J is a dial.

Run from the repo root (numpy/scipy only, ~1 min)::

  python examples/proton_slab.py

Everything is in atomic units at toy scale (no polynomial lambda); the point is the geometry.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as onp
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import eigsh

root = next(p for p in [pathlib.Path(__file__).resolve().parent, *pathlib.Path(__file__).resolve().parents]
            if (p / "src" / "lattice_geometry.py").exists())
sys.path.insert(0, str(root))
from src.lattice_geometry import Bravais2D, canonical_basis_2d, finite_triangular_patch


# --------------------------------------------------------------------------- #
# 1. Geometry: uniform slab + well clusters, discretized to point charges
# --------------------------------------------------------------------------- #

SLAB_BASIS = canonical_basis_2d(Bravais2D.TRIANGULAR)          # a1 = (1, 0), a2 = (1/2, sqrt3/2)


def slab_protons(H: float, S: float, h: float, center=(0.0, 0.0)):
  r"""AA-stacked triangular-lattice layers of spacing ``h``: the lattice points within a disk of
  radius ``H/2`` about ``center`` (a hexagonal patch from ``src.lattice_geometry`` trimmed to the
  disk), at ``z = +-h/2, +-3h/2, ...`` in ``[-S/2, S/2]``.

  The midplane ``z = 0`` carries no charge; the number density is ``rho = 2 / (sqrt3 h^3)``. The
  patch must be centred on the wells: its in-plane confinement is parabolic and far larger than
  any tunnelling splitting, so off-centre wells are simply inequivalent (a square patch also
  splits a triangle's doublet through its fourfold edge field).
  """
  center = onp.asarray(center, dtype=float)
  reach = (H / 2 + onp.linalg.norm(center)) / (h * onp.sqrt(3) / 2)   # hexagon inradius must cover the disk
  xy = finite_triangular_patch(SLAB_BASIS * h, int(onp.ceil(reach)) + 1)
  xy = xy[onp.linalg.norm(xy - center, axis=1) <= H / 2]
  zs = onp.arange(-S / 2 + h / 2, S / 2, h)
  return onp.concatenate([onp.column_stack([xy, onp.full(len(xy), z)]) for z in zs])


def slab_density(h: float) -> float:
  return 2.0 / (onp.sqrt(3.0) * h ** 3)


def well_cluster(center, n_shell: int = 2, spacing: float = 0.35):
  r"""A small ball of extra protons around ``center``: the discretized well density.

  Built from stacked triangular-lattice layers so the cluster shares the slab's symmetry
  (a cubic cluster would make the three bonds of a triangle of wells inequivalent).
  """
  idx = onp.arange(-2 * n_shell, 2 * n_shell + 1)
  N1, N2 = onp.meshgrid(idx, idx, indexing="ij")
  xy = (N1.ravel()[:, None] * SLAB_BASIS[:, 0] + N2.ravel()[:, None] * SLAB_BASIS[:, 1]) * spacing
  zs = onp.arange(-n_shell, n_shell + 1) * spacing
  pts = onp.concatenate([onp.column_stack([xy, onp.full(len(xy), z)]) for z in zs])
  pts = pts[onp.linalg.norm(pts, axis=1) <= n_shell * spacing + 1e-9]   # round it off
  return pts + onp.asarray(center)


def build_structure(well_sites_xy, H=10.0, S=2.0, h=0.5, q_slab=1.0, q_well=1.0, well_spacing=0.35):
  r"""Returns ``(positions, charges, is_well)`` of all protons: slab grid plus one cluster per well site.

  ``q_slab`` scales the slab charges (``q = rho h^3`` keeps the density fixed under refinement);
  ``q_well`` scales the cluster charges.
  """
  slab = slab_protons(H, S, h, center=onp.atleast_2d(well_sites_xy).mean(axis=0))
  pos, chg, is_well = [slab], [onp.full(len(slab), q_slab)], [onp.zeros(len(slab), bool)]
  for s in onp.atleast_2d(well_sites_xy):
    c = well_cluster([s[0], s[1], 0.0], spacing=well_spacing)
    # clusters sit on the midplane: shift them off z = 0 by a third of their spacing so no
    # charge lands on the plane (or the probe lines) where we evaluate the potential
    c[:, 2] += well_spacing / 3
    pos.append(c); chg.append(onp.full(len(c), q_well)); is_well.append(onp.ones(len(c), bool))
  return onp.concatenate(pos), onp.concatenate(chg), onp.concatenate(is_well)


def potential(points, pos, chg, chunk=4096):
  r"""``U(r) = -sum_b q_b / |r - b|`` for an electron, evaluated at ``points`` (m, 3)."""
  out = onp.empty(len(points))
  for i in range(0, len(points), chunk):
    d = onp.linalg.norm(points[i:i + chunk, None, :] - pos[None], axis=-1)
    out[i:i + chunk] = -(chg[None] / d).sum(axis=1)
  return out


# --------------------------------------------------------------------------- #
# 2. Midplane Schrödinger problem: localized modes and the hopping between wells
# --------------------------------------------------------------------------- #

def midplane_modes(U_grid, dx, n_modes=4):
  r"""Lowest eigenpairs of ``-Laplacian/2 + U`` on a square grid with Dirichlet edges (2D cartoon)."""
  n = U_grid.shape[0]
  lap1 = sparse.diags([onp.ones(n - 1), -2 * onp.ones(n), onp.ones(n - 1)], [-1, 0, 1]) / dx ** 2
  I = sparse.identity(n)
  Hmat = -0.5 * (sparse.kron(lap1, I) + sparse.kron(I, lap1)) + sparse.diags(U_grid.ravel())
  vals, vecs = eigsh(Hmat.tocsr(), k=n_modes, sigma=U_grid.min() - 1.0, which="LM")
  order = onp.argsort(vals)
  return vals[order], vecs[:, order].T.reshape(n_modes, n, n)


def hopping_from_pair(k: int, H, S, h, grid_n=161, box=5.0, **kw):
  r"""Two wells ``k`` lattice steps apart along ``a1`` (``d = k h``), both at triangle centres of the
  slab lattice so their environments match: ``t = (E_antibonding - E_bonding) / 2`` on the midplane."""
  c = (SLAB_BASIS[:, 0] + SLAB_BASIS[:, 1]) * h / 3                  # an up-triangle centre
  sites = onp.array([c, c + k * h * SLAB_BASIS[:, 0]])
  pos, chg, _ = build_structure(sites, H=H, S=S, h=h, **kw)
  x = onp.linspace(-box, box, grid_n) + sites.mean(axis=0)[0]
  y = onp.linspace(-box, box, grid_n) + sites.mean(axis=0)[1]
  X, Y = onp.meshgrid(x, y, indexing="ij")
  pts = onp.stack([X.ravel(), Y.ravel(), onp.zeros(X.size)], axis=1)
  U = potential(pts, pos, chg).reshape(grid_n, grid_n)
  vals, _ = midplane_modes(U, x[1] - x[0], n_modes=2)
  return 0.5 * (vals[1] - vals[0])


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def fig_structure(pos, is_well, sites, out):
  fig = plt.figure(figsize=(7, 5.5))
  ax = fig.add_subplot(projection="3d")
  slab = ~is_well
  ax.scatter(*pos[slab].T, s=3, c="#9aa5b1", alpha=0.35, label="slab protons (uniform density)")
  ax.scatter(*pos[~slab].T, s=14, c="#c0392b", label="well-cluster protons")
  ax.scatter(sites[:, 0], sites[:, 1], onp.zeros(len(sites)), s=120, marker="*", c="gold",
             edgecolors="k", label="one electron per well")
  ax.set(xlabel="x", ylabel="y", zlabel="z", title="The hardness instance as a crystal: "
         f"{len(pos)} protons, {len(sites)} electrons")
  ax.legend(loc="upper left", fontsize=8)
  ax.set_box_aspect((1, 1, 0.35))
  ax.view_init(elev=58, azim=-55)
  ax.set_zticks([-0.75, 0, 0.75])
  fig.savefig(out / "slab_structure.png", dpi=150, bbox_inches="tight")
  plt.close(fig)


def fig_potential(sites, H, S, out, box=4.5, grid_n=181):
  fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained")

  pos, chg, _ = build_structure(sites, H=H, S=S, h=0.5)
  x = onp.linspace(-box, box, grid_n)
  X, Y = onp.meshgrid(x, x, indexing="ij")
  pts = onp.stack([X.ravel(), Y.ravel(), onp.zeros(X.size)], axis=1)
  U = potential(pts, pos, chg).reshape(grid_n, grid_n)
  im = axs[0].imshow(U.T, origin="lower", extent=(-box, box, -box, box), cmap="viridis",
                     vmin=U.min(), vmax=U.min() + 40)
  axs[0].scatter(sites[:, 0], sites[:, 1], s=60, marker="*", c="gold", edgecolors="k")
  axs[0].set(title="midplane potential $U(x, y, 0)$", xlabel="x", ylabel="y")
  fig.colorbar(im, ax=axs[0], shrink=0.85)

  # harmonic confinement across the slab, at a well and away from wells
  z = onp.linspace(-S / 2 + 0.3, S / 2 - 0.3, 121)
  for (px, py), label in [((sites[0, 0] + 0.08, sites[0, 1] + 0.05), "through a well"),
                          ((-box + 1.0, -box + 1.0), "away from wells")]:
    line = onp.stack([onp.full_like(z, px), onp.full_like(z, py), z], axis=1)
    Uz = potential(line, pos, chg)
    axs[1].plot(z, Uz - Uz.min(), label=label)
  rho = slab_density(0.5)                                      # slab number density at h = 0.5, q = 1
  axs[1].plot(z, 2 * onp.pi * rho * z ** 2, "k--", lw=1, label=r"$2\pi\rho z^2$ (infinite slab)")
  axs[1].set(title="confinement across the slab", xlabel="z", ylabel="$U - U_{min}$")
  axs[1].legend(fontsize=8)

  # Gauss-rule step: point charges at spacing h with q = rho h^3 -> smooth design as h -> 0
  line_x = onp.linspace(-box, box, 241)
  line = onp.stack([line_x, onp.zeros_like(line_x), onp.zeros_like(line_x)], axis=1)
  for h in (1.0, 0.5, 0.25):
    pos_h, chg_h, _ = build_structure(sites, H=H, S=S, h=h, q_slab=rho * h ** 3)
    axs[2].plot(line_x, potential(line, pos_h, chg_h), label=f"h = {h:g}, q = {rho * h ** 3:g}")
  axs[2].set(title="point charges converging to the design\n(fixed density, finer grid)",
             xlabel="x along a row of wells", ylabel="$U(x, 0, 0)$")
  axs[2].legend(fontsize=8)
  fig.savefig(out / "slab_potential.png", dpi=150)
  plt.close(fig)


def fig_modes(sites, H, S, out, box=4.0, grid_n=161):
  pos, chg, _ = build_structure(sites, H=H, S=S, h=0.5)
  x = onp.linspace(-box, box, grid_n)
  X, Y = onp.meshgrid(x, x, indexing="ij")
  pts = onp.stack([X.ravel(), Y.ravel(), onp.zeros(X.size)], axis=1)
  U = potential(pts, pos, chg).reshape(grid_n, grid_n)
  vals, modes = midplane_modes(U, x[1] - x[0], n_modes=4)

  fig, axs = plt.subplots(1, 4, figsize=(16, 4.2), layout="constrained")
  for k in range(3):
    axs[k].imshow((modes[k] ** 2).T, origin="lower", extent=(-box, box, -box, box), cmap="magma")
    axs[k].scatter(sites[:, 0], sites[:, 1], s=30, marker="+", c="cyan")
    axs[k].set(title=f"mode {k}: E = {vals[k]:.3f}", xlabel="x", ylabel="y")
  t_tri = (vals[1] - vals[0]) / 3.0                           # triangle: E0 - 2t, E0 + t (x2)
  axs[2].text(0.03, 0.95, f"triangle: $E_1 - E_0 = 3t$,  $t \\approx {t_tri:.3f}$",
              transform=axs[2].transAxes, color="w", fontsize=9, va="top")

  ks = onp.array([4, 5, 6])
  ds = ks * 0.5
  ts = onp.array([hopping_from_pair(k, H, S, 0.5) for k in ks])
  axs[3].semilogy(ds, onp.maximum(ts, 1e-6), "o-", label="pair of wells, same slab")
  axs[3].axhline(t_tri, color="k", ls="--", lw=1, label="triangle (side 2.0)")
  axs[3].set(title="hopping $t$ vs well spacing ($J = 4t^2/U$)", xlabel="spacing d",
             ylabel="$t$")
  axs[3].legend(fontsize=8)
  axs[3].text(0.03, 0.05, "the slab's in-plane confinement\nraises or lowers barriers too:\n"
              "a finite-size effect at toy scale", transform=axs[3].transAxes, fontsize=7)
  fig.savefig(out / "slab_modes.png", dpi=150)
  plt.close(fig)
  return vals, t_tri, ds, ts


# --------------------------------------------------------------------------- #

def main():
  out = root / "examples" / "out" / "proton_slab"
  out.mkdir(parents=True, exist_ok=True)
  H, S = 10.0, 2.0

  # wells on an equilateral triangle commensurate with the slab lattice (equivalent
  # environments), side 4h = 2.0: the smallest frustrated graph
  A = SLAB_BASIS * 4 * 0.5
  tri = onp.array([[0.0, 0.0], A[:, 0], A[:, 1]])
  tri -= tri.mean(axis=0)                                     # vertices at triangle centres of the slab lattice

  pos, chg, is_well = build_structure(tri, H=H, S=S, h=0.5)
  print(f"structure: {len(pos)} protons ({int((~is_well).sum())} slab, "
        f"{int(is_well.sum())} in well clusters), {len(tri)} electrons")
  fig_structure(pos, is_well, tri, out)
  fig_potential(tri, H, S, out)
  vals, t_tri, ds, ts = fig_modes(tri, H, S, out)
  print("midplane levels:", onp.round(vals, 3))
  print(f"triangle hopping t ~ {t_tri:.4f}; pair hopping vs spacing:")
  for d, t in zip(ds, ts):
    print(f"  d = {d:.1f}: t = {t:.4f}")
  print("superexchange J = 4 t^2 / U > 0 on every bond of the triangle: a frustrated magnet,\n"
        "programmed by where the protons sit. The real construction adds the Piddock–Montanaro\n"
        "mediators to realize signed couplings and dilates everything by a polynomial lambda.")
  print(f"figures in {out}")


if __name__ == "__main__":
  main()
