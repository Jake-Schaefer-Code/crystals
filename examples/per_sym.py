# per_sym.py
r"""

Periodic symmetry, lattice, Bloch-wave, and simple dispersion helpers.

Notation used below:
  L^*: reciprocal lattice with G . c in 2 pi Z for all direct lattice vectors c.
  c_l: inclusion or lattice-site centers.
  B_n(c_l): site quantity to transform, such as multipole coefficients or
    sampled fields.
"""

from __future__ import annotations

import dataclasses as dcls
import numpy as onp
from numpy import sin, exp, sqrt
from collections.abc import Callable
from src.named_groups import cyclic_group
from src.core import Hom, Endo
from src.lattice_geometry import (
  basis_vectors,
  first_bz_hexagon_vertices,
  finite_triangular_patch,
  canonical_basis_2d,
  Bravais2D
)
import src.symmetry as sym

NDArray = onp.ndarray


# ---- common defs
_2π = 2 * onp.pi

@dcls.dataclass
class DomainConfig:
  a: float = 1.0
  Nx_cell: int = 10
  Ny_cell: int = 10
  res_per_cell: int = 80
  # pick k0
  k0 = 2.2

  @property
  def Lx(self):
    return self.a * self.Nx_cell
  @property
  def Ly(self):
    return self.a * self.Ny_cell
  @property
  def Nx(self):
    return self.Nx_cell * self.res_per_cell
  @property
  def Ny(self):
    return self.Ny_cell * self.res_per_cell



def gaussian(x: NDArray, mu: float = 0.0, sigma: float = 1.0) -> NDArray:
  return exp(-0.5 * ((x - mu) / sigma) ** 2)

def bloch_wave(u, k, Ω):
  # envelope times k phase
  # 2D: kx * X + ky * Y
  return u * exp(1j * onp.dot(Ω, k))

def bloch_transform(sites: NDArray, Bn: Callable):
  r""" 
  Bn: quantity to be transformed. Should be vectorized over coefficients 
  
  Properties:
    B_n(c_l) should satisfy
      B_n(c_l) = \sum_{m=1}^{N_B} \hat{B}_n^m exp(ik_B^m\cdot c_l),
    where \hat{B}_n^m is the complex amplitude of the Bloch wave associated with
    Bloch vector k_B^m in the decomposition of B_n(c_l).

  Thus, since exponentials form a basis, FB_n(k) should peak when k=k_B^m
  """
  Bnc = Bn(sites) # B_n(c_l)
  def F_Bn(k):
    K = onp.atleast_2d(k)        # shape (N_k, 2)
    phases = exp(-1j * onp.matmul(K, sites.T))  # exp(ik_B^m\cdot c_l) # (N_sites, N_B)
    values = onp.dot(Bnc, phases) # (N_sites,)
    return values
  return F_Bn

def synthesize_mode_on_sites(sites: NDArray, k_star: NDArray, Bn: Callable) -> NDArray:
  """
  B(c_l) = sum_m A_m exp(i k_B^m . c_l)

  This is the discrete object whose Bloch transform we will plot.
  """

  F_Bn = bloch_transform(sites, Bn)
  return F_Bn(k_star)

def bloch_transform_map(sites, k_star, K, amplitudes=None) -> NDArray:
  """
  F(k) = sum_l B(c_l) exp(-i k . c_l)
  evaluated on a 2D grid in reciprocal space.
  """
  def Bn(sites):
    if amplitudes is None:
      return onp.ones(len(k_star), dtype=complex)
    return amplitudes
  # equal amplitudes -> six symmetric peaks
  F_Bnk = synthesize_mode_on_sites(sites, k_star, Bn)
  phases = exp(-1j * (K @ sites.T))            # (N_k, N_sites)
  FK = phases @ F_Bnk                             # (N_k,)
  # intensity 
  return onp.abs(FK) ** 2


def bloch_potential(ks: NDArray, v: Hom[NDArray, Endo[NDArray]]):
  r""" 
  If a mode consists of N_B Bloch waves with Bloch vectors k_B^m, so that its 
  field distribution satisfies
  V(r) = \sum_{m=1}^{N_B} exp(ik k_B^m \cdot r) v_{k_B^m}(r),
  where functions v_{k_B^m}(r) have periodicity of the lattice L

  Properties:
    v should satisfy periodicity of the lattice L

  Note:
    implicitly, since v_{k_B^m}(r) is indexed by k_B^m, we may write it as a 
    function with two arguments as v(k_B^m, r), or as a function v: R^3 -> C(R^3),
    so a function taking a k-vector to a function of a k-vector, i.e., v is a kind of inner product
    that defines a linear functional for each k_B^m.
  """

  vkBm = v(ks) 
  def V(r: NDArray):
    # ks should be a vector of (N_B, 3), (or should N_B be last?),
    # and r should be a vector of (B, 3)

    # matmul: (n,k),(k,m) -> (n,m)
    phases = exp(1j * onp.matmul(ks, r.T))
    return onp.sum(phases * vkBm(r), axis=0)
  return V

def monatomic_dispersion(k: NDArray, K: float = 1.0, m: float = 1.0, a: float = 1.0) -> NDArray:
  """
  Monatomic nearest-neighbor chain:
    omega^2(k) = (4K/m) sin^2(ka/2)
  """
  return 2.0 * sqrt(K / m) * onp.abs(sin(0.5 * k * a))

def diatomic_dispersion(k, K=1.0, m1=1.0, m2=2.0, a=1.0):
  """
  Diatomic nearest-neighbor chain with two masses per primitive cell.

  Using the standard Bloch reduction, the two branches satisfy
    omega^2 = A ± sqrt(A^2 - B sin^2(ka/2))

  where
    A = K/m1 + K/m2
    B = 4K^2/(m1 m2)

  This produces:
    - acoustic branch  (minus sign)
    - optical branch   (plus sign)
  """
  A = K * (1 / m1 + 1 / m2)
  disc = A**2 - 4.0 * K**2 / (m1 * m2) * sin(0.5 * k * a)**2
  disc = onp.clip(disc, 0.0, None)
  B = sqrt(disc)

  ω2_acoustic = A - B
  ω2_optical = A + B

  ω_acoustic = sqrt(onp.clip(ω2_acoustic, 0.0, None))
  ω_optical = sqrt(onp.clip(ω2_optical, 0.0, None))

  return ω_acoustic, ω_optical


def free_electron_energy_1d(k: NDArray, hbar: float = 1.0, m: float = 1.0) -> NDArray:
  """Free-electron dispersion E(k) = hbar^2 k^2 / (2m)."""
  k = onp.asarray(k, dtype=float)
  return (hbar ** 2 / (2.0 * m)) * k ** 2


def nearly_free_electron_1d_bands(
  k: NDArray,
  V_G: complex | float,
  a: float = 1.0,
  hbar: float = 1.0,
  m: float = 1.0,
  V0: float = 0.0,
  G: float | None = None,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
  r"""
  Two-band nearly-free-electron model near a 1D Brillouin-zone boundary.

  In the basis {|k>, |k-G>}, the Hamiltonian is
    [[E0(k) + V0, V_G],
     [V_G^*,      E0(k-G) + V0]]

  so the split branches satisfy
    E_\pm(k) = trace \pm sqrt(detuning^2 + |V_G|^2).

  At k = G/2 = pi/a, the direct gap is 2|V_G|.
  """
  k = onp.asarray(k, dtype=float)
  G = _2π / a if G is None else float(G)

  E_k = free_electron_energy_1d(k, hbar=hbar, m=m) + V0
  E_kmG = free_electron_energy_1d(k - G, hbar=hbar, m=m) + V0

  trace = 0.5 * (E_k + E_kmG)
  detuning = 0.5 * (E_k - E_kmG)
  split = onp.sqrt(detuning ** 2 + onp.abs(V_G)**2)
  return E_k, E_kmG, trace - split, trace + split


def folded_free_electron_branches_1d(
  k: NDArray,
  a: float = 1.0,
  hbar: float = 1.0,
  m: float = 1.0,
  V0: float = 0.0,
  n_shells: int = 2,
  G: float | None = None,
) -> tuple[NDArray, NDArray]:
  """
  Free-electron branches folded into the first Brillouin zone.

  Returns:
    basis_indices: reciprocal-lattice shifts n with momenta k + n G
    folded: pointwise energy-ordered branches with shape (2*n_shells+1, len(k))
  """
  k = onp.asarray(k, dtype=float)
  G = _2π / a if G is None else float(G)
  basis_indices = onp.arange(-n_shells, n_shells + 1)
  branches = onp.vstack([
    free_electron_energy_1d(k + n * G, hbar=hbar, m=m) + V0
    for n in basis_indices
  ])
  return basis_indices, onp.sort(branches, axis=0)


def nearly_free_electron_reduced_bands_1d(
  k: NDArray,
  V_G: complex | float,
  V_2G: complex | float = 0.0,
  a: float = 1.0,
  hbar: float = 1.0,
  m: float = 1.0,
  V0: float = 0.0,
  n_shells: int = 2,
  G: float | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
  r"""
  Reduced-zone nearly-free-electron bands from a truncated plane-wave basis.

  Basis states are |k + nG> for n in {-n_shells, ..., n_shells}. We keep only
  the first few Fourier harmonics of the periodic potential:
    - neighboring plane-wave sectors couple by V_G
    - next-nearest sectors couple by V_2G

  Returns:
    basis_indices: reciprocal-lattice shifts n
    eigvals: energies with shape (n_bands, len(k))
    eigvecs: eigenvectors with shape (len(k), n_bands, n_bands)
  """
  k = onp.asarray(k, dtype=float)
  G = _2π / a if G is None else float(G)
  basis_indices = onp.arange(-n_shells, n_shells + 1)
  n_bands = len(basis_indices)
  eigvals = onp.zeros((len(k), n_bands), dtype=float)
  eigvecs = onp.zeros((len(k), n_bands, n_bands), dtype=complex)
  coupling_1 = complex(V_G)
  coupling_2 = complex(V_2G)

  for i, q in enumerate(k):
    plane_wave_ks = q + basis_indices * G
    H = onp.diag(
      (free_electron_energy_1d(plane_wave_ks, hbar=hbar, m=m) + V0).astype(complex)
    )
    for j in range(n_bands - 1):
      H[j, j + 1] = onp.conjugate(coupling_1)
      H[j + 1, j] = coupling_1
    if coupling_2 != 0:
      for j in range(n_bands - 2):
        H[j, j + 2] = onp.conjugate(coupling_2)
        H[j + 2, j] = coupling_2
    vals, vecs = onp.linalg.eigh(H)
    eigvals[i] = vals.real
    eigvecs[i] = vecs

  return basis_indices, eigvals.T, eigvecs

def make_k_mesh(kmax: float, ngrid: int) -> tuple[NDArray, ...]:
  k = onp.linspace(-kmax, kmax, ngrid)
  return onp.meshgrid(k, k, indexing='xy')


def periodic_potential_from_sites(X, Y, pts, sigma=0.22, depth=1.0, cutoff=None):
  """
  Sum of identical Gaussian wells centered at lattice sites.
  """
  V = onp.zeros_like(X)
  for p in pts:
    dx = X - p[0]
    dy = Y - p[1]
    r2 = dx * dx + dy * dy
    if cutoff is None:
      V -= depth * exp(-r2 / (2 * sigma * sigma))
    else:
      mask = r2 <= cutoff * cutoff
      V[mask] -= depth * exp(-r2[mask] / (2 * sigma * sigma))
  return V


def nearest_neighbor_vectors(A: NDArray, a2: NDArray | None = None) -> NDArray:
  """
  Six nearest neighbors for triangular lattice.
  """
  if a2 is None:
    a1, a2 = basis_vectors(A)
  else:
    a1 = onp.asarray(A, dtype=float)
    a2 = onp.asarray(a2, dtype=float)

  return onp.array([
    a1,
    -a1,
    a2,
    -a2,
    a2 - a1,
    a1 - a2,
  ])


def find_origin_site(pts: NDArray, tol: float = 1e-10) -> int:
  d2 = onp.sum(pts**2, axis=1)
  idx = onp.argmin(d2)
  if d2[idx] > tol:
    raise ValueError("No origin site found.")
  return idx



def plot_example():
  """Show a synthetic six-wave Bloch transform with first-BZ overlay."""
  import matplotlib.pyplot as plt


  A = canonical_basis_2d(Bravais2D.TRIANGULAR, scale=1.0)
  B = sym.recip_lattice(A)
  b1, b2 = basis_vectors(B)
  bz = first_bz_hexagon_vertices(B)

  # finite lattice in real space
  sites = finite_triangular_patch(A, N=7)

  # choose one Bloch vector near a BZ corner, then generate its sixfold orbit
  k_corner = (2.0 * b1 + b2) / 3.0
  k0 = 0.92 * k_corner
  # sixfold_star: 
  # Orbit of a single Bloch vector under 60-degree rotations:
  #       {R(n pi/3) k0}_{n=0}^5
  k_star = cyclic_group(6, dim=2).orbit(k0)

  # reciprocal-space sampling window

  rmax = 1.15 * onp.max(onp.linalg.norm(bz, axis=1))
  KX, KY = make_k_mesh(rmax, 501)
  K = onp.column_stack([KX.ravel(), KY.ravel()])   # (N_k, 2)

  I = bloch_transform_map(sites, k_star, K)
  I = I.reshape(KX.shape) / I.max()

  fig, ax = plt.subplots(figsize=(7, 7))
  im = ax.imshow(
    I,
    origin='lower',
    extent=(KX.min(), KX.max(), KY.min(), KY.max()),
    aspect='equal',
    interpolation='bilinear'
  )

  # overlay first Brillouin-zone hexagon
  bz_closed = onp.vstack([bz, bz[0]])
  ax.plot(bz_closed[:, 0], bz_closed[:, 1], color='white', lw=2.0)

  # optionally mark the six Bloch vectors that generated the mode
  ax.scatter(k_star[:, 0], k_star[:, 1], s=24)

  ax.set(xlabel=r'$k_x$', ylabel=r'$k_y$', title='Synthetic six-wave Bloch transform with first BZ overlay')
  plt.colorbar(im, ax=ax, label='normalized intensity')
  return fig

def main():
  import matplotlib.pyplot as plt
  fig = plot_example()
  plt.show()

if __name__ == "__main__":
  main()
