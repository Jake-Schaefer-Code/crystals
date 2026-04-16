# categorization.py
from __future__ import annotations
import numpy as onp
from src import crystal_funcs as cfuncs
import src.symmetry as sym
from IPython.display import display, Math, Markdown
onp.set_printoptions(precision=2, suppress=True)

def parse_rotation(symbol, i, axis):
  n = int(symbol[i])
  return [(cfuncs.q_rot_mat(j * 2 * onp.pi / n, axis), onp.zeros(3)) for j in range(n)]

def parse_mirror_plane(axis):
  return (sym.reflection(axis), onp.zeros(3))

def parse_glide_plane(symbol, i):
  if symbol[i - 1] == 'a':
    axis = onp.array([1., 0, 0])
  elif symbol[i - 1] == 'b':
    axis = onp.array([0, 1., 0])
  elif symbol[i - 1] == 'c':
    axis = onp.array([0, 0, 1.])
  else:
      raise NameError(f"Unknown axis at position {i}: {symbol[i]}")
  return sym.glide_reflection(axis, 0.5*axis)

def parse_screw_axis(symbol, i, axis_z):
  """
  """
  n = int(symbol[i])
  return [sym.screw_rotation(2 * onp.pi / n, axis_z, onp.array([0, 0, 0.5]))]


def parse_space_group(symbol):
  """
  """
  sym_ops = [sym.identity(3)]
  axis_z = onp.array([0, 0, 1.])
  axis_x = onp.array([1., 0, 0])
  axis_y = onp.array([0, 1., 0])

  i = 0
  while i < len(symbol):
    char = symbol[i]
    if char.isdigit() and (i > 0 and symbol[i-1].lower() == 'p'):
      sym_ops.extend(parse_rotation(symbol, i, axis_z))
    
    elif char == 'm': 
      if i > 0 and symbol[i - 1].isdigit():  
        sym_ops.append(parse_mirror_plane(axis_z))
        if i + 1 < len(symbol) and symbol[i + 1] == 'm':
          sym_ops.append(parse_mirror_plane(axis_x))  
          sym_ops.append(parse_mirror_plane(axis_y))  
      else:  
        sym_ops.append(parse_mirror_plane(axis_z))

    elif char == 'n': 
      sym_ops.append(parse_glide_plane(symbol, i))
    
    elif char == '/':  
      i += 1
      if i < len(symbol) and symbol[i].isdigit():
        sym_ops.extend(parse_screw_axis(symbol, i, axis_z))
      elif i < len(symbol) and symbol[i] == 'm':
        sym_ops.append(parse_mirror_plane(axis_z))  
      elif i < len(symbol) and symbol[i] == 'c':
        sym_ops.append(sym.glide_reflection(axis_z, onp.array([0, 0.5, 0.5])))  

    elif char == '1':  
      sym_ops.append(sym.identity(3))
    
    # TODO more
    
    i += 1
  return sym_ops

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


def print_symmetry_operations(sym_ops):
  for op in sym_ops:
    display(Markdown(f"**{op.name}**"))
    display(Math(fr"$$\text{{Rotation/Reflection:}}{latex_bmatrix(op.matrix)}\qquad\text{{Translation:}}\qquad{latex_bmatrix_vec(op.translation)}$$"))
    display(Markdown("---"))
