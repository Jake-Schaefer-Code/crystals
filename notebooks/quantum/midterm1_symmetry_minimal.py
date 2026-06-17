from __future__ import annotations

import itertools as it
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as onp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

import src.quantum_symmetry as qsym
import src.symmetry as sym
from plotting_utils import save_path


def rank_one(psi: onp.ndarray) -> onp.ndarray:
  psi = onp.asarray(psi, dtype=complex)
  return psi[:, None] @ onp.conjugate(psi[None, :])


def apply_kraus(rho: onp.ndarray, kraus_ops: list[onp.ndarray]) -> onp.ndarray:
  rho = onp.asarray(rho, dtype=complex)
  return sum(K @ rho @ K.conjugate().T for K in kraus_ops)


def cyclic_unitary_rep(generator: onp.ndarray, order: int, *, name: str) -> sym.Representation:
  elements = tuple(range(order))
  mats = {g: onp.linalg.matrix_power(generator, g) for g in elements}
  return sym.Representation(group=elements, matrices=mats, name=name)


def trine_states() -> list[onp.ndarray]:
  root3 = onp.sqrt(3.0)
  return [
    onp.array([1.0, 0.0], dtype=complex),
    onp.array([-0.5, root3 / 2.0], dtype=complex),
    onp.array([-0.5, -root3 / 2.0], dtype=complex),
  ]


def trine_povm() -> list[onp.ndarray]:
  return [(2.0 / 3.0) * rank_one(psi) for psi in trine_states()]


def transition_matrix(states: list[onp.ndarray], effects: list[onp.ndarray]) -> onp.ndarray:
  probs = onp.zeros((len(states), len(effects)), dtype=float)
  for i, psi in enumerate(states):
    for j, effect in enumerate(effects):
      probs[i, j] = onp.real(onp.vdot(psi, effect @ psi))
  return probs


def projective_basis(alpha: float, beta: float) -> tuple[onp.ndarray, onp.ndarray]:
  c = onp.cos(beta / 2.0)
  s = onp.sin(beta / 2.0)
  phase = onp.exp(1j * alpha)
  return (
    onp.array([c, phase * s], dtype=complex),
    onp.array([-onp.conjugate(phase) * s, c], dtype=complex),
  )


def best_projective_strategy(
  states: list[onp.ndarray],
  *,
  n_alpha: int = 361,
  n_beta: int = 181,
) -> dict[str, object]:
  best: dict[str, object] | None = None
  for beta in onp.linspace(0.0, onp.pi, n_beta):
    for alpha in onp.linspace(0.0, 2.0 * onp.pi, n_alpha, endpoint=False):
      basis = projective_basis(alpha, beta)
      projectors = [rank_one(vec) for vec in basis]
      outcome_probs = transition_matrix(states, projectors)
      for guesses in it.product(range(3), repeat=2):
        guessed = onp.zeros((3, 3), dtype=float)
        for state_idx in range(3):
          for outcome_idx, guess_idx in enumerate(guesses):
            guessed[state_idx, guess_idx] += outcome_probs[state_idx, outcome_idx]
        success = float(onp.trace(guessed) / 3.0)
        if best is None or success > best["success"]:
          best = {
            "alpha": float(alpha),
            "beta": float(beta),
            "guesses": tuple(int(g) for g in guesses),
            "basis": basis,
            "projectors": projectors,
            "transition_matrix": guessed,
            "success": success,
          }
  assert best is not None
  return best


def qutrit_clock() -> onp.ndarray:
  omega = onp.exp(2j * onp.pi / 3.0)
  return onp.diag([1.0, omega, omega**2]).astype(complex)


def qutrit_dephasing_kraus(p: float) -> list[onp.ndarray]:
  if not 0.0 <= p <= 1.0:
    raise ValueError("p must lie in [0, 1]")
  I3 = onp.eye(3, dtype=complex)
  Z = qutrit_clock()
  return [
    onp.sqrt(1.0 - p) * I3,
    onp.sqrt(p / 3.0) * I3,
    onp.sqrt(p / 3.0) * Z,
    onp.sqrt(p / 3.0) * (Z @ Z),
  ]


def operator_damping_heatmap(p: float) -> onp.ndarray:
  heatmap = onp.zeros((3, 3), dtype=float)
  kraus_ops = qutrit_dephasing_kraus(p)
  for m in range(3):
    for n in range(3):
      E_mn = onp.zeros((3, 3), dtype=complex)
      E_mn[m, n] = 1.0
      image = apply_kraus(E_mn, kraus_ops)
      heatmap[m, n] = float(onp.real_if_close(image[m, n]))
  return heatmap


def charge_sector_labels(n_sites: int = 2, local_dim: int = 3) -> dict[int, list[str]]:
  sectors = {q: [] for q in range(local_dim)}
  for state in it.product(range(local_dim), repeat=n_sites):
    q = sum(state) % local_dim
    sectors[q].append("|" + "".join(map(str, state)) + ">")
  return sectors


def collective_charge_vectors() -> dict[int, onp.ndarray]:
  basis = onp.eye(9, dtype=complex)
  sector_vectors: dict[int, onp.ndarray] = {}
  for q, labels in charge_sector_labels().items():
    vec = onp.zeros(9, dtype=complex)
    for label in labels:
      digits = tuple(int(ch) for ch in label[1:-1])
      idx = 3 * digits[0] + digits[1]
      vec += basis[idx]
    sector_vectors[q] = vec / onp.linalg.norm(vec)
  return sector_vectors


def collective_dephasing_kraus(p: float) -> list[onp.ndarray]:
  Z = qutrit_clock()
  U = onp.kron(Z, Z)
  I9 = onp.eye(9, dtype=complex)
  return [
    onp.sqrt(1.0 - p) * I9,
    onp.sqrt(p / 3.0) * I9,
    onp.sqrt(p / 3.0) * U,
    onp.sqrt(p / 3.0) * (U @ U),
  ]


def collective_sector_retention(p: float) -> onp.ndarray:
  vectors = collective_charge_vectors()
  kraus_ops = collective_dephasing_kraus(p)
  retention = onp.zeros((3, 3), dtype=float)
  for q in range(3):
    for r in range(3):
      O_qr = vectors[q][:, None] @ onp.conjugate(vectors[r][None, :])
      image = apply_kraus(O_qr, kraus_ops)
      retention[q, r] = float(onp.real_if_close(onp.vdot(vectors[q], image @ vectors[r])))
  return retention


def symmetry_sector_dimensions() -> dict[int, int]:
  Z = qutrit_clock()
  U = onp.kron(Z, Z)
  rep = cyclic_unitary_rep(U, 3, name="collective C3 charge rep")
  dims: dict[int, int] = {}
  for q, chi in enumerate(qsym.cyclic_characters(3)):
    Pq = rep.projector(chi, irrep_dim=1)
    dims[q] = int(qsym.projected_basis(Pq).shape[1])
  return dims


def annotate_heatmap(ax, mat: onp.ndarray, *, title: str, xlabels: list[str], ylabels: list[str]) -> None:
  im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="viridis")
  ax.set_title(title)
  ax.set_xticks(range(len(xlabels)), labels=xlabels)
  ax.set_yticks(range(len(ylabels)), labels=ylabels)
  for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
      val = mat[i, j]
      ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val < 0.72 else "black")
  plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def make_trine_figure() -> tuple[plt.Figure, dict[str, object], dict[str, object]]:
  states = trine_states()
  povm = trine_povm()
  povm_matrix = transition_matrix(states, povm)
  best_proj = best_projective_strategy(states)

  fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
  labels = [r"$\psi_0$", r"$\psi_1$", r"$\psi_2$"]
  annotate_heatmap(
    axes[0],
    povm_matrix,
    title=fr"Trine POVM: $P(\mathrm{{guess}}=j\mid \psi_i)$, $P_{{succ}}={onp.trace(povm_matrix)/3:.3f}$",
    xlabels=labels,
    ylabels=labels,
  )
  annotate_heatmap(
    axes[1],
    best_proj["transition_matrix"],
    title=fr"Best Projective Rule: $P_{{succ}}={best_proj['success']:.3f}$",
    xlabels=labels,
    ylabels=labels,
  )
  return fig, {"transition_matrix": povm_matrix, "success": float(onp.trace(povm_matrix) / 3.0)}, best_proj


def make_dephasing_figure(p: float) -> tuple[plt.Figure, dict[str, object]]:
  damping = operator_damping_heatmap(p)
  retention = collective_sector_retention(p)
  dims = symmetry_sector_dimensions()
  sectors = charge_sector_labels()

  fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
  labels = [r"$|0\rangle$", r"$|1\rangle$", r"$|2\rangle$"]
  annotate_heatmap(
    axes[0],
    damping,
    title=fr"Single-Qutrit Dephasing Factors, $p={p:.2f}$",
    xlabels=labels,
    ylabels=labels,
  )
  sector_names = [fr"$q={q}$" for q in range(3)]
  annotate_heatmap(
    axes[1],
    retention,
    title="Collective Two-Qutrit Sector Retention",
    xlabels=sector_names,
    ylabels=sector_names,
  )
  return fig, {
    "single_qutrit_fixed_space_dim": 3,
    "sector_dimensions": dims,
    "sector_labels": sectors,
    "single_qutrit_damping": damping,
    "collective_retention": retention,
  }


def run_analysis(output_dir: str | Path | None = None, *, p: float = 0.6) -> dict[str, object]:
  out_dir = Path(output_dir or "/tmp/crystals_midterm1_symmetry")
  out_dir.mkdir(parents=True, exist_ok=True)

  fig1, trine, projective = make_trine_figure()
  trine_path = save_path(fig1, out_dir / "trine_transition_heatmaps.png", tight_layout=False)

  fig2, dephasing = make_dephasing_figure(p)
  dephasing_path = save_path(fig2, out_dir / "dephasing_sector_heatmaps.png", tight_layout=False)

  povm = trine_povm()
  povm_sum = sum(povm)
  kraus_sum = sum(K.conjugate().T @ K for K in qutrit_dephasing_kraus(p))
  Z = qutrit_clock()
  clock_rep = cyclic_unitary_rep(Z, 3, name="C3 clock rep")
  clock_commutant = qsym.commutant_basis(clock_rep)

  return {
    "output_dir": out_dir,
    "trine_figure": trine_path,
    "dephasing_figure": dephasing_path,
    "trine_povm_completeness_error": float(onp.linalg.norm(povm_sum - onp.eye(2))),
    "trine_success": trine["success"],
    "best_projective_success": projective["success"],
    "best_projective_rule": projective["guesses"],
    "best_projective_alpha": projective["alpha"],
    "best_projective_beta": projective["beta"],
    "qutrit_kraus_completeness_error": float(onp.linalg.norm(kraus_sum - onp.eye(3))),
    "qutrit_offdiag_factor": float(operator_damping_heatmap(p)[0, 1]),
    "qutrit_fixed_space_dim": int(clock_commutant.shape[0]),
    "two_qutrit_sector_dimensions": dephasing["sector_dimensions"],
    "two_qutrit_sector_labels": dephasing["sector_labels"],
  }


def main() -> None:
  results = run_analysis()
  print("Problem 1A: trine POVM")
  print(f"  completeness error: {results['trine_povm_completeness_error']:.3e}")
  print(f"  trine success probability: {results['trine_success']:.6f}")
  print(f"  best projective success: {results['best_projective_success']:.6f}")
  print(f"  best projective guess rule (two outcomes): {results['best_projective_rule']}")
  print()
  print("Problem 2B: qutrit dephasing")
  print(f"  Kraus completeness error: {results['qutrit_kraus_completeness_error']:.3e}")
  print(f"  off-diagonal damping factor: {results['qutrit_offdiag_factor']:.6f}")
  print(f"  fixed-point / commutant dimension: {results['qutrit_fixed_space_dim']}")
  print()
  print("DFS extension: collective C3 dephasing on two qutrits")
  print(f"  symmetry sector dimensions: {results['two_qutrit_sector_dimensions']}")
  for q, labels in results["two_qutrit_sector_labels"].items():
    print(f"  charge {q}: {labels}")
  print()
  print(f"saved figures to {results['output_dir']}")


if __name__ == "__main__":
  main()
