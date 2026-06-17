from __future__ import annotations

from collections.abc import Sequence
import itertools as it
import numpy as onp
import jax.numpy as jnp

from src.core import Array
import src.permutations as perms
import src.symmetry as sym


def _H(x: Array) -> Array:
  return jnp.conjugate(jnp.asarray(x)).T


def vec(A: Array) -> Array:
  r"""Column-major vectorization using NumPy/JAX-compatible primitives."""
  return jnp.asarray(A).T.reshape((-1,))


def unvec(v: Array, d: int) -> Array:
  return jnp.asarray(v).reshape((d, d)).T


def comm(A: Array, B: Array) -> Array:
  A = jnp.asarray(A)
  B = jnp.asarray(B)
  return A @ B - B @ A


def anticomm(A: Array, B: Array) -> Array:
  A = jnp.asarray(A)
  B = jnp.asarray(B)
  return A @ B + B @ A


def outer(psi: Array, phi: Array | None = None) -> Array:
  psi = jnp.asarray(psi)
  phi = psi if phi is None else jnp.asarray(phi)
  return psi[:, None] @ jnp.conjugate(phi[None, :])


def lindblad_rhs(H: Array, Ls: Sequence[Array], rho: Array) -> Array:
  r"""Dense Lindblad right-hand side."""

  H = jnp.asarray(H)
  rho = jnp.asarray(rho)
  rhs = -1j * comm(H, rho)
  for L in Ls:
    L = jnp.asarray(L)
    LdL = _H(L) @ L
    rhs = rhs + L @ rho @ _H(L) - 0.5 * anticomm(LdL, rho)
  return rhs


def lindblad_superoperator(H: Array, Ls: Sequence[Array], *, vec_convention: str = "col") -> Array:
  r"""Dense Liouville-space Lindblad matrix using column-major vectorization."""

  if vec_convention != "col":
    raise NotImplementedError("only column-major vectorization is implemented")

  H = jnp.asarray(H)
  d = H.shape[0]
  I = jnp.eye(d, dtype=H.dtype)

  Lsuper = -1j * (jnp.kron(I, H) - jnp.kron(H.T, I))
  for L in Ls:
    L = jnp.asarray(L)
    LdL = _H(L) @ L
    jump = jnp.kron(jnp.conjugate(L), L)
    left_decay = jnp.kron(I, LdL)
    right_decay = jnp.kron(LdL.T, I)
    Lsuper = Lsuper + jump - 0.5 * (left_decay + right_decay)
  return Lsuper


def liouville_rep(U_rep: sym.Representation) -> sym.Representation:
  r"""Return the operator-space representation ``R_g = U_g^* \otimes U_g``."""

  mats = {
    g: jnp.kron(jnp.conjugate(jnp.asarray(U)), jnp.asarray(U))
    for g, U in U_rep.matrices.items()
  }
  name = f"{U_rep.name} Liouville".strip()
  return sym.Representation(group=U_rep.group, matrices=mats, name=name)


def projected_basis(P: Array, tol: float = 1e-10) -> Array:
  r"""Return an orthonormal basis for the image of a numerical projector."""

  P = jnp.asarray(P, dtype=complex)
  herm = 0.5 * (P + _H(P))
  evals, evecs = jnp.linalg.eigh(herm)
  keep = evals > tol
  return evecs[:, keep]


def restrict_operator(A: Array, B: Array) -> Array:
  return _H(B) @ jnp.asarray(A) @ jnp.asarray(B)


def check_covariance(H: Array, Ls: Sequence[Array], U_rep: sym.Representation) -> dict[str, object]:
  r"""Measure how far a Lindbladian is from commuting with a symmetry action."""

  Lsuper = lindblad_superoperator(H, Ls)
  R_rep = liouville_rep(U_rep)
  deviations = {
    g: float(jnp.linalg.norm(Lsuper @ Rg - Rg @ Lsuper))
    for g, Rg in R_rep.matrices.items()
  }
  return {
    "per_element": deviations,
    "max_deviation": max(deviations.values(), default=0.0),
  }


def cyclic_characters(n: int) -> tuple[Array, ...]:
  if n <= 0:
    raise ValueError("n must be positive")
  omega = jnp.exp(2j * jnp.pi / n)
  return tuple(
    jnp.asarray([omega ** (k * g) for g in range(n)], dtype=complex)
    for k in range(n)
  )


def _nullspace_basis(A: Array, tol: float = 1e-10) -> Array:
  A = jnp.asarray(A, dtype=complex)
  _, s, vh = jnp.linalg.svd(A, full_matrices=True)
  rank = int(jnp.sum(s > tol))
  return jnp.conjugate(vh[rank:].T)


def commutant_basis(rep: sym.Representation, tol: float = 1e-10) -> Array:
  r"""Basis for matrices ``X`` satisfying ``X rho(g) = rho(g) X`` for all ``g``."""

  d = rep.dim
  I = jnp.eye(d, dtype=complex)
  constraints = [
    jnp.kron(I, jnp.asarray(A)) - jnp.kron(jnp.asarray(A).T, I)
    for A in rep.matrices.values()
  ]
  stacked = jnp.concatenate(constraints, axis=0)
  basis_vecs = _nullspace_basis(stacked, tol=tol)
  if basis_vecs.shape[1] == 0:
    return jnp.zeros((0, d, d), dtype=complex)
  return jnp.stack([unvec(basis_vecs[:, i], d) for i in range(basis_vecs.shape[1])], axis=0)


def permutation_tensor_rep(
  group: perms.PermutationGroup | Sequence[perms.Permutation],
  *,
  local_dim: int,
  name: str | None = None,
) -> sym.Representation:
  r"""Permutation action on ``(\mathbb{C}^{local_dim})^{\otimes n}``."""

  if local_dim <= 0:
    raise ValueError("local_dim must be positive")

  elements = tuple(group)
  if not elements:
    raise ValueError("group must contain at least one permutation")

  n_sites = len(elements[0])
  basis = tuple(it.product(range(local_dim), repeat=n_sites))
  index_of = {state: i for i, state in enumerate(basis)}
  dim = local_dim ** n_sites

  mats: dict[object, Array] = {}
  for sigma in elements:
    U = jnp.zeros((dim, dim), dtype=complex)
    for col, state in enumerate(basis):
      image = perms.act_on_indexed_data(sigma, state)
      row = index_of[image]
      U[row, col] = 1.0
    mats[sigma] = U

  group_name = getattr(group, "name", "")
  rep_name = name or f"{group_name} tensor rep".strip()
  return sym.Representation(group=group, matrices=mats, name=rep_name)


def quaternion_to_su2(a, b, c, d, *, dtype=jnp.complex128):
  return jnp.array([
      [+a + 1j * b, c + 1j * d],
      [-c + 1j * d, a - 1j * b],
  ], dtype=dtype)


def quaternion_group_rep(*, dtype=jnp.complex128):
  mats = {
    "1": quaternion_to_su2(1, 0, 0, 0, dtype=dtype),
    "-1": quaternion_to_su2(-1, 0, 0, 0, dtype=dtype),
    "i": quaternion_to_su2(0, 1, 0, 0, dtype=dtype),  # = i Z
    "-i": quaternion_to_su2(0, -1, 0, 0, dtype=dtype),
    "j": quaternion_to_su2(0, 0, 1, 0, dtype=dtype),  # = i Y
    "-j": quaternion_to_su2(0, 0, -1, 0, dtype=dtype),
    "k": quaternion_to_su2(0, 0, 0, 1, dtype=dtype),  # = i X
    "-k": quaternion_to_su2(0, 0, 0, -1, dtype=dtype),
  }
  return sym.Representation(group=tuple(mats), matrices=mats, name="Q8 spin rep")

