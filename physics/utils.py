# physics/utils.py
from __future__ import annotations
import numpy as onp

NDArray = onp.ndarray

def coulomb_like_potential(r: NDArray, depth: float = 3.2, softening: float = 0.22) -> NDArray:
  r"""
  Softened Coulomb-like well used for schematic bound-state plots.
  """
  return -depth / onp.sqrt(r**2 + softening**2)

def free_particle_energy(KX: NDArray, KY: NDArray, prefactor: float = 0.5) -> NDArray:
  r"""
  Dimensionless free-particle dispersion prefactor * (kx^2 + ky^2).
  """
  return prefactor * (KX**2 + KY**2)
