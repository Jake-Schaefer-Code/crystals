# src/bases.py
from __future__ import annotations
import dataclasses as dcls
import numpy as onp

NDArray = onp.ndarray

stat_arr_field = lambda arr: dcls.field(default_factory=lambda: arr)

def static_col_field(*xs):
	return stat_arr_field(onp.array([*xs]))
  
@dcls.dataclass(frozen=True)
class CanonBasis2:
	e1: NDArray = static_col_field(1, 0)
	e2: NDArray = static_col_field(0, 1)

@dcls.dataclass(frozen=True)
class CanonBasis3:
	e1: NDArray = stat_arr_field(onp.array([1, 0, 0]))
	e2: NDArray = stat_arr_field(onp.array([0, 1, 0]))
	e3: NDArray = stat_arr_field(onp.array([0, 0, 1]))
	def __iter__(self):
		for e in (self.e1, self.e2, self.e3):
			yield e

@dcls.dataclass(frozen=True)
class PauliBasis:
	x: NDArray = stat_arr_field(onp.array([[0, 1], [1, 0]]))
	y: NDArray = stat_arr_field(onp.array([[0, -1j], [1j, 0]]))
	z: NDArray = stat_arr_field(onp.array([[1, 0], [0, -1]]))


@dcls.dataclass(frozen=True)
class CanonBasis4x4:
  E11: NDArray = stat_arr_field(onp.array([[1, 0], [0, 0]], dtype=complex))  # E11
  E12: NDArray = stat_arr_field(onp.array([[0, 1], [0, 0]], dtype=complex))  # E12
  E21: NDArray = stat_arr_field(onp.array([[0, 0], [1, 0]], dtype=complex))  # E21
  E22: NDArray = stat_arr_field(onp.array([[0, 0], [0, 1]], dtype=complex))  # E22
  def __iter__(self):
    for E in (self.E11, self.E12, self.E21, self.E22):
      yield E

  def names_latex(self):
    return ("E_{11}", "E_{12}", "E_{21}", "E_{22}")
