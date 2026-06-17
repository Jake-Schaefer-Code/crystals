
from __future__ import annotations
from src.symmetry.symmetry import AffineOperation, FiniteGroupAction
import numpy as onp
NDArray = onp.ndarray
from src import geo_ops_utils as gops


# to handle numeric error near zero in generated group elements
def _snap(x, eps: float=1e-14):
  return onp.where(onp.abs(x) < eps, 0.0, x)

def identity(dim: int = 2, *, inversion: bool = False, label: str = "e") -> AffineOperation:
  matrix = -onp.eye(dim) if inversion else onp.eye(dim)
  return AffineOperation(matrix, onp.zeros(dim), label)

def about(center: NDArray, op: AffineOperation) -> AffineOperation:
  M, t = op.matrix, op.translation
  return AffineOperation(M, center - M @ center + t)

def rotation2d(theta: float, *, label: str | None = None) -> AffineOperation:
  return AffineOperation(_snap(gops.rot_mat(theta)), label=label or f"r({theta:g})")

def rotation3d(theta: float, axis: NDArray, *, label: str | None = None) -> AffineOperation:
  return AffineOperation(_snap(gops.q_rot_mat(theta, axis)), label=label or f"r({theta:g})")


def reflection(
  normal: NDArray,
  translation: NDArray | None = None,
  *,
  label: str = "s",
) -> AffineOperation:
  """
  R = I - 2nn^T
  
  Parameters:
  ----------------
  normal : arraylike
      The normal vector to the plane.
      
  Returns:
  --------
  R : NDArray, shape = (ndim, ndim)
  """
  normal = onp.asarray(normal, dtype=float)
  normal = normal / onp.linalg.norm(normal)
  matrix = onp.eye(len(normal)) - 2.0 * onp.outer(normal, normal)
  return AffineOperation(_snap(matrix), translation, label)


def glide_reflection(normal: NDArray, translation: NDArray, *, label: str = "g") -> AffineOperation:
  return reflection(normal, translation, label=label)

def screw_rotation(theta: float, axis: NDArray, translation: NDArray) -> AffineOperation:
  matrix = gops.q_rot_mat(theta, axis)
  translation = translation * axis
  return AffineOperation(_snap(matrix), translation)

def conjugate_action(action: FiniteGroupAction, op: AffineOperation):
  op_inv = op.inverse()
  return FiniteGroupAction(
    [op.compose(g).compose(op_inv) for g in action],
    name=action.name,
  )