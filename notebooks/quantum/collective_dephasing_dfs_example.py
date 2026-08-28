from __future__ import annotations

from functools import reduce
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as onp
from scipy.linalg import expm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from plotting_utils import save_path
import src.symmetry.permutations as perms
import src.quantum_symmetry as qsym

Q8 = qsym.quaternion_group_rep()
I2 = Q8.matrices["1"]
X = -1j * Q8.matrices["k"]
Y = -1j * Q8.matrices["j"]
Z = -1j * Q8.matrices["i"]


KET0 = onp.array([1, 0], dtype=complex)
KET1 = onp.array([0, 1], dtype=complex)

LABELS = ["00", "01", "10", "11"]


def kron(*ops: onp.ndarray) -> onp.ndarray:
  return reduce(onp.kron, ops)


def rank_one(psi: onp.ndarray, phi: onp.ndarray | None = None) -> onp.ndarray:
  phi = psi if phi is None else phi
  return psi[:, None] @ onp.conjugate(phi[None, :])


def computational_basis() -> dict[str, onp.ndarray]:
  return {
    "00": kron(KET0, KET0),
    "01": kron(KET0, KET1),
    "10": kron(KET1, KET0),
    "11": kron(KET1, KET1),
  }


def restricted(A: onp.ndarray, basis: onp.ndarray) -> onp.ndarray:
  return onp.conjugate(basis).T @ A @ basis


def evolve_liouville(Lsuper: onp.ndarray, rho0: onp.ndarray, t: float) -> onp.ndarray:
  rho_t_vec = expm(t * Lsuper) @ qsym.vec(rho0)
  return qsym.unvec(rho_t_vec, d=rho0.shape[0])


def operator_retention_heatmap(Lsuper: onp.ndarray, t: float) -> onp.ndarray:
  d = 4
  out = onp.zeros((d, d), dtype=float)
  for i in range(d):
    for j in range(d):
      Eij = onp.zeros((d, d), dtype=complex)
      Eij[i, j] = 1.0
      image = evolve_liouville(Lsuper, Eij, t)
      out[i, j] = float(onp.abs(image[i, j]))
  return out


def annotate_heatmap(ax: Axes, mat: onp.ndarray, *, title: str) -> None:
  im = ax.imshow(mat, cmap="viridis", vmin=0.0, vmax=1.0)
  ax.set(title=title, xlabel="ket index", ylabel="bra index")
  ax.set_xticks(range(len(LABELS)), labels=LABELS)
  ax.set_yticks(range(len(LABELS)), labels=LABELS)
  for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
      val = mat[i, j]
      ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val < 0.72 else "black")
  plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def run_example(output_dir: str | Path | None = None, *, gamma: float = 0.8, omega: float = 1.0) -> dict[str, object]:
  out_dir = Path(output_dir or "/tmp/crystals_collective_dephasing_dfs")
  out_dir.mkdir(parents=True, exist_ok=True)

  basis = computational_basis()
  ket00 = basis["00"]
  ket01 = basis["01"]
  ket10 = basis["10"]
  ket11 = basis["11"]

  tplus = ket00
  tminus = ket11
  tzero = (ket01 + ket10) / onp.sqrt(2.0)
  singlet = (ket01 - ket10) / onp.sqrt(2.0)

  B_symadapt = onp.column_stack([tplus, tzero, tminus, singlet])
  B_dfs = onp.column_stack([ket01, ket10])

  Jz = kron(Z, I2) + kron(I2, Z)
  L = onp.sqrt(gamma) * Jz

  # XY exchange stays inside the DFS and gives nontrivial protected dynamics.
  H = 0.5 * omega * (kron(X, X) + kron(Y, Y))

  L_noise = qsym.lindblad_superoperator(onp.zeros((4, 4), dtype=complex), [L])
  L_full = qsym.lindblad_superoperator(H, [L])

  swap = perms.perm_from_cycles(((1, 2),), degree=2)
  S2 = perms.PermutationGroup.generated((swap,), name="S2")
  U_rep = qsym.permutation_tensor_rep(S2, local_dim=2, name="S2 on two qubits")
  cov = qsym.check_covariance(H, [L], U_rep)

  identity = perms.identity_perm(2)
  chi_trivial = {g: 1 for g in U_rep.elements}
  chi_sign = {g: (-1 if g == swap else 1) for g in U_rep.elements}
  chi_sign[identity] = 1
  P_sym = U_rep.projector(chi_trivial, irrep_dim=1)
  P_asym = U_rep.projector(chi_sign, irrep_dim=1)

  sym_basis = qsym.projected_basis(P_sym)
  asym_basis = qsym.projected_basis(P_asym)

  P_dfs = rank_one(ket01) + rank_one(ket10)
  dfs_dim = int(onp.linalg.matrix_rank(P_dfs))
  sym_intersection_rank = int(onp.linalg.matrix_rank(P_sym @ P_dfs))
  asym_intersection_rank = int(onp.linalg.matrix_rank(P_asym @ P_dfs))

  L_dfs = restricted(L, B_dfs)
  H_dfs = restricted(H, B_dfs)
  L_symadapt = restricted(L, B_symadapt)
  H_symadapt = restricted(H, B_symadapt)

  psi_dfs = (ket01 + onp.exp(1j * onp.pi / 4.0) * ket10) / onp.sqrt(2.0)
  psi_ghz = (ket00 + ket11) / onp.sqrt(2.0)
  rho_dfs0 = rank_one(psi_dfs)
  rho_ghz0 = rank_one(psi_ghz)

  ts = onp.linspace(0.0, 6.0, 181)
  dfs_coh = []
  ghz_coh = []
  dfs_purity = []
  ghz_purity = []
  dfs_weight = []
  dfs_fidelity = []
  ghz_fidelity = []
  for t in ts:
    U_t = expm(-1j * H * float(t))
    rho_dfs_ideal = U_t @ rho_dfs0 @ onp.conjugate(U_t).T
    rho_ghz_ideal = U_t @ rho_ghz0 @ onp.conjugate(U_t).T
    rho_dfs_t = evolve_liouville(L_full, rho_dfs0, float(t))
    rho_ghz_t = evolve_liouville(L_full, rho_ghz0, float(t))
    dfs_coh.append(float(onp.abs(rho_dfs_t[1, 2])))
    ghz_coh.append(float(onp.abs(rho_ghz_t[0, 3])))
    dfs_purity.append(float(onp.real(onp.trace(rho_dfs_t @ rho_dfs_t))))
    ghz_purity.append(float(onp.real(onp.trace(rho_ghz_t @ rho_ghz_t))))
    dfs_weight.append(float(onp.real(onp.trace(P_dfs @ rho_dfs_t))))
    dfs_fidelity.append(float(onp.real(onp.trace(rho_dfs_ideal @ rho_dfs_t))))
    ghz_fidelity.append(float(onp.real(onp.trace(rho_ghz_ideal @ rho_ghz_t))))

  heatmap = operator_retention_heatmap(L_noise, t=1.0)

  fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
  annotate_heatmap(
    axes[0],
    heatmap,
    title="Collective Dephasing: retained computational coherences at t=1",
  )
  axes[0].text(
    0.02,
    -0.24,
    "Only the 01-10 coherence survives off-diagonal because both states have Jz charge 0.",
    transform=axes[0].transAxes,
    fontsize=9,
  )

  axes[1].plot(ts, dfs_fidelity, label="DFS fidelity to ideal unitary target", lw=2.0)
  axes[1].plot(ts, ghz_fidelity, label="GHZ fidelity to ideal unitary target", lw=2.0)
  axes[1].plot(ts, dfs_purity, label=r"${\rm Tr}\,\rho_{\rm DFS}(t)^2$", lw=1.6, ls="--")
  axes[1].plot(ts, ghz_purity, label=r"${\rm Tr}\,\rho_{\rm GHZ}(t)^2$", lw=1.6, ls="--")
  axes[1].set_title("Protected vs unprotected evolution")
  axes[1].set_xlabel("time")
  axes[1].set_ylabel("fidelity / purity")
  axes[1].set_ylim(-0.02, 1.05)
  axes[1].grid(alpha=0.25)
  axes[1].legend(fontsize=8, loc="upper right")

  figure_path = save_path(fig, out_dir / "collective_dephasing_dfs.png", tight_layout=False)

  return {
    "output_dir": out_dir,
    "figure_path": figure_path,
    "swap_covariance_max_deviation": cov["max_deviation"],
    "sym_dim": int(sym_basis.shape[1]),
    "asym_dim": int(asym_basis.shape[1]),
    "dfs_dim": dfs_dim,
    "dfs_sym_intersection_rank": sym_intersection_rank,
    "dfs_asym_intersection_rank": asym_intersection_rank,
    "restricted_jump_on_dfs": L_dfs,
    "restricted_hamiltonian_on_dfs": H_dfs,
    "restricted_jump_symadapted": L_symadapt,
    "restricted_hamiltonian_symadapted": H_symadapt,
    "final_dfs_fidelity": dfs_fidelity[-1],
    "final_ghz_fidelity": ghz_fidelity[-1],
    "final_dfs_coherence": dfs_coh[-1],
    "final_ghz_coherence": ghz_coh[-1],
    "final_dfs_purity": dfs_purity[-1],
    "final_ghz_purity": ghz_purity[-1],
    "final_dfs_weight": dfs_weight[-1],
  }


def main() -> None:
  results = run_example()
  print("Collective two-qubit dephasing with swap symmetry")
  print(f"  swap covariance max deviation: {results['swap_covariance_max_deviation']:.3e}")
  print(f"  symmetric / antisymmetric dimensions: {results['sym_dim']} / {results['asym_dim']}")
  print(f"  DFS dimension: {results['dfs_dim']}")
  print(
    "  DFS overlap ranks with S2 sectors: "
    f"sym={results['dfs_sym_intersection_rank']}, asym={results['dfs_asym_intersection_rank']}"
  )
  print()
  print("Restricted operators on DFS basis {|01>, |10>}:")
  print(results["restricted_jump_on_dfs"])
  print(results["restricted_hamiltonian_on_dfs"])
  print()
  print(f"  final DFS fidelity to ideal unitary target: {results['final_dfs_fidelity']:.6f}")
  print(f"  final GHZ fidelity to ideal unitary target: {results['final_ghz_fidelity']:.6f}")
  print(f"  final DFS purity: {results['final_dfs_purity']:.6f}")
  print(f"  final GHZ purity: {results['final_ghz_purity']:.6f}")
  print(f"  final DFS population weight: {results['final_dfs_weight']:.6f}")
  print()
  print(f"saved figure to {results['figure_path']}")


if __name__ == "__main__":
  main()
