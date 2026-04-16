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

@dcls.dataclass(frozen=True)
class PauliBasis:
	x: NDArray = stat_arr_field(onp.array([[0, 1], [1, 0]]))
	y: NDArray = stat_arr_field(onp.array([[0, -1j], [1j, 0]]))
	z: NDArray = stat_arr_field(onp.array([[1, 0], [0, -1]]))

