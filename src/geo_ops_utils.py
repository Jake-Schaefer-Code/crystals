# geo_ops_utils.py
from __future__ import annotations
import dataclasses as dcls
import itertools
import numpy as onp
import numpy.linalg as onpla
from src.bases import CanonBasis3

NDArray = onp.ndarray

# TODO lazy until something that needs it is called
B = CanonBasis3()


def rotation_quaternions(axes, thetas) -> NDArray:
  thetas = onp.atleast_1d(thetas)
  axes = axes / onpla.norm(axes, axis=-1, keepdims=True)
  sin_theta = onp.sin(thetas / 2)
  quaternions = onp.zeros((len(thetas), 4))
  quaternions[:, 0] = onp.cos(thetas / 2)
  quaternions[:, 1:] = axes * sin_theta[:, None]
  return quaternions

def make_q_rot_mats(q) -> NDArray:
  w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
  Rs = onp.zeros((len(w), 3, 3))
  Rs[:, 0, 0] = 2 * (w * w + x * x) - 1
  Rs[:, 0, 1] = 2 * (x * y - w * z)
  Rs[:, 0, 2] = 2 * (x * z + w * y)
  Rs[:, 1, 0] = 2 * (x * y + w * z)
  Rs[:, 1, 1] = 2 * (w * w + y * y) - 1
  Rs[:, 1, 2] = 2 * (y * z - w * x)
  Rs[:, 2, 0] = 2 * (x * z - w * y)
  Rs[:, 2, 1] = 2 * (y * z + w * x)
  Rs[:, 2, 2] = 2 * (w * w + z * z) - 1
  if len(Rs) == 1:
    return Rs[0]
  return Rs

def q_rot_mat(theta=0, axis: NDArray|None=None):
  """
  Creates a rotation matrix for a given quaternion

  Parameters:
  ----------------
  q : list
      quaternion
  
  Returns:
  ----------------
  """
  if axis is None: axis = B.e3
  q = rotation_quaternions(axis, theta)
  return make_q_rot_mats(q)

def rotate(points: NDArray, theta=0, axis: NDArray|None=None) -> NDArray:
  """
  Helper method. Rotates a configuration (array of positions)

  Parameters:
  ----------------
  crystal : Atoms
      The atomic structure to apply this transformation to
  theta : float, default: 0
      Angle by which to rotate the configuration
  axis : onp.ndarray, default: onp.array([0,0,1])
      Axis around which to rotate the configuration

  Returns:
  ----------------
  Rotated configuration
  """
  if axis is None: axis = B.e3
  q = rotation_quaternions(axis, theta)
  rot_mat = make_q_rot_mats(q)
  return points @ rot_mat.T


def quaternion_multiply(q1: NDArray, q2: NDArray) -> NDArray:
  w1, x1, y1, z1 = q1.T
  w2, x2, y2, z2 = q2.T
  w = w2 * w1 - x2 * x1 - y2 * y1 - z2 * z1
  x = w2 * x1 + x2 * w1 + y2 * z1 - z2 * y1
  y = w2 * y1 - x2 * z1 + y2 * w1 + z2 * x1
  z = w2 * z1 + x2 * y1 - y2 * x1 + z2 * w1
  return onp.array([w, x, y, z]).T

def generate_rotation_matrices(num_rots: int) -> NDArray:
  t = onp.linspace(0, 1, num_rots, endpoint=False)
  phis, thetas = onp.meshgrid(
    onp.pi / 2 * (2 * t - 1), 
    2 * onp.pi * t,
  )
  coords = onp.vstack([thetas.ravel(), phis.ravel()]).T
  axes_z = onp.array([B.e3 for _ in range(len(coords[:, 0]))])
  qs_theta = rotation_quaternions(axes_z, coords[:, 0])
  matrices = make_q_rot_mats(qs_theta)
  axes_x_rot = matrices @ B.e1
  # axes_x_rot = onp.einsum('nij,j->ni', matrices, onp.array([1,0,0]))
  qs_phi = rotation_quaternions(axes_x_rot, coords[:, 1])
  qs_combined = quaternion_multiply(qs_theta, qs_phi)
  combined_rot_mats = make_q_rot_mats(qs_combined)
  return combined_rot_mats

def generate_rot_mats2d(num_rots):
  thetas = onp.linspace(0, 2 * onp.pi, num_rots, endpoint=False)
  

def rotate_polyhedron(polyhedron, rotation_matrices):
  return onp.einsum('nij,mj->nmi', rotation_matrices, polyhedron)

def rot_mat(theta: float) -> NDArray:
  c, s = onp.cos(theta), onp.sin(theta)
  return onp.array([[c, -s], [s, c]])




def rotate_about_pivot(poly, angle, pivot=(0,0)):
  """ Rotate triangle around a pivot point """
  pivot = onp.array(pivot)
  R = rot_mat(angle)
  return (poly - pivot) @ R.T + pivot


def rotate_around_axis(v, k, theta):
  k = k / onpla.norm(k) 
  c, s = onp.cos(theta), onp.sin(theta)
  v_rot = v * c + onp.cross(k, v) * s + k * onp.dot(k, v) * (1 - c)
  return v_rot


def reflect_axis_points2(point: NDArray, edge) -> NDArray:
  """Reflect a point across a line defined by two points."""
  ax_p1, ax_p2 = edge
  # Vector along the line
  v = ax_p2 - ax_p1

  dx, dy = v
  norm2 = onpla.norm(v) ** 2
  # 1 = (dx^2 - dy^2) + 2 dy^2 => dx^2 - dy^2 = 1 - 2 dy^2
  # 1 - 2 dx ^ 2 = (dy^2 - dx^2) = -a
  a = (1 - 2 * (dy * dy / norm2))
  b = 2 * dx * dy / norm2
  
  px, py = (point - ax_p1)
  rp = onp.array([
    a * px + b * py,
    b * px - a * py
  ])
  return rp + ax_p1


def reflect_points(points: NDArray, edge) -> NDArray:
  """Reflect a point or an array of points across a line through two axis points."""
  # axis points
  ax_p1, ax_p2 = edge
  points = onp.asarray(points, dtype=float)
  ax_p1 = onp.asarray(ax_p1, dtype=float)
  ax_p2 = onp.asarray(ax_p2, dtype=float)
  # Vector along the line
  v = ax_p2 - ax_p1
  # Normal to the line
  normal = onp.array([-v[1], v[0]])
  normal = normal / onpla.norm(normal)
  # Reflection matrix
  # 1 - 2 dy^2
  R = onp.eye(len(normal)) - 2.0 * onp.outer(normal, normal)
  return (points - ax_p1) @ R + ax_p1
