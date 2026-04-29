# categorization.py
from __future__ import annotations
import dataclasses as dcls
import numpy as onp
# from src import crystal_funcs as cfuncs
import src.symmetry as sym
from IPython.display import display, Math, Markdown
onp.set_printoptions(precision=2, suppress=True)
NDArray = onp.ndarray

stat_arr_field = lambda arr: dcls.field(default_factory=lambda: arr)

def static_col_field(*xs):
	return stat_arr_field(onp.array([*xs]))


@dcls.dataclass(frozen=True)
class CanonBasis3:
  r""" Ordered basis for R3, in standard (x,y,z) order"""
  e1: NDArray = stat_arr_field(onp.array([1, 0, 0]))
  e2: NDArray = stat_arr_field(onp.array([0, 1, 0])) 
  e3: NDArray = stat_arr_field(onp.array([0, 0, 1])) 


def parse_rotation(symbol, i, axis):
  n = int(symbol[i])
  return [sym.rotation3d(j * 2 * onp.pi / n, axis) for j in range(n)]
  # return [(cfuncs.q_rot_mat(j * 2 * onp.pi / n, axis), onp.zeros(3)) for j in range(n)]

def parse_mirror_plane(axis):
  # return (sym.reflection(axis), onp.zeros(3))
  return sym.reflection(axis)

def parse_glide_plane(symbol, i):
  B = CanonBasis3()
  if symbol[i - 1] == 'a':
    axis = B.e1
  elif symbol[i - 1] == 'b':
    axis = B.e2
  elif symbol[i - 1] == 'c':
    axis = B.e3
  else:
    raise NameError(f"Unknown axis at position {i}: {symbol[i]}")
  return sym.glide_reflection(axis, 0.5*axis)

def parse_screw_axis(symbol, i, axis_z):
  n = int(symbol[i])
  return [sym.screw_rotation(2 * onp.pi / n, axis_z, onp.array([0, 0, 0.5]))]


def parse_space_group(symbol):
  sym_ops = [sym.identity(3)]
  B = CanonBasis3()
  i = 0
  while i < len(symbol):
    char = symbol[i]
    if char.isdigit() and (i > 0 and symbol[i-1].lower() == 'p'):
      sym_ops.extend(parse_rotation(symbol, i, B.e3))
    
    elif char == 'm': 
      if i > 0 and symbol[i - 1].isdigit():  
        sym_ops.append(parse_mirror_plane(B.e3))
        if i + 1 < len(symbol) and symbol[i + 1] == 'm':
          sym_ops.append(parse_mirror_plane(B.e1))  
          sym_ops.append(parse_mirror_plane(B.e2))  
      else:  
        sym_ops.append(parse_mirror_plane(B.e3))

    elif char == 'n': 
      sym_ops.append(parse_glide_plane(symbol, i))
    
    elif char == '/':  
      i += 1
      if i < len(symbol) and symbol[i].isdigit():
        sym_ops.extend(parse_screw_axis(symbol, i, B.e3))
      elif i < len(symbol) and symbol[i] == 'm':
        sym_ops.append(parse_mirror_plane(B.e3))  
      elif i < len(symbol) and symbol[i] == 'c':
        translation = 0.5 * (B.e2 + B.e3)
        sym_ops.append(sym.glide_reflection(B.e3, translation))  

    elif char == '1':  
      sym_ops.append(sym.identity(3))
    
    # TODO more
    
    i += 1
  return sym_ops

def cyclic_power_rep(A, order: int):
  G = tuple(range(order)) # elements 0,1,2,...,n of Z/nZ
  reps = {g: onp.linalg.matrix_power(A, g) for g in G} # powers of A (I, A, ..., A^{n-1})
  return sym.Representation(
    group=G,
    matrices=reps, 
    name=r"Hom(C_\sigma^2, C_\tau^2)",
)


def latex_bmatrix(A, fmt=".3g"):
  A = onp.asarray(A)
  rows = [" & ".join(format(x, fmt) for x in row) for row in A]
  body = r" \\ ".join(rows)
  return rf"\begin{{bmatrix}}{body}\end{{bmatrix}}"

def latex_bmatrix_vec(A, fmt=".3g"):
  A = onp.asarray(A)
  rows = [format(x, fmt) for x in A]
  body = r" \\ ".join(rows)
  return rf"\begin{{bmatrix}}{body}\end{{bmatrix}}"

def show_latex_bmatrix(A, name: str='A', fmt=".3g"):
  display(Math(fr"$${name} = {latex_bmatrix(A, fmt)}$$"))

def show_latex_eqn(eqn: str):
  display(Math(fr"$${eqn}$$"))

def print_symmetry_operations(sym_ops: list[sym.AffineOperation]):
  for op in sym_ops:
    display(Markdown(f"**{op.label}**"))
    display(Math(fr"$$\text{{Rotation/Reflection:}}{latex_bmatrix(op.matrix)}\qquad\text{{Translation:}}\qquad{latex_bmatrix_vec(op.translation)}$$"))
    display(Markdown("---"))
