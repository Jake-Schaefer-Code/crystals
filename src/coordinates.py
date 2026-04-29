# coordinates.py
from __future__ import annotations
import dataclasses as dcls
import numpy as np
import numpy as onp

import numpy.linalg as npla

NDArray = np.ndarray

def barycentric_coordinates2D(p: NDArray, triangle: NDArray):
  r"""
  `p` is some point in the triangle (a -> b -> c)

  - first, calculate distance of the two vertices (b, c) from the vertex a
  - similarly, calculate the distance of point p from vertex a
  -
  """
  a, b, c = triangle[:3]
  # tri_m1 = onp.roll(triangle[:3], -1)

  v0 = b - a
  v1 = c - a
  v2 = p - a

  d00 = np.dot(v0, v0)
  d01 = np.dot(v0, v1)
  d11 = np.dot(v1, v1)
  d20 = np.dot(v2, v0.T)
  d21 = np.dot(v2, v1.T)

  denom = d00 * d11 - d01 * d01
  l1 = (d11 * d20 - d01 * d21) / denom
  l2 = (d00 * d21 - d01 * d20) / denom
  l3 = 1 - l1 - l2
  return np.vstack([l1, l2, l3]).T

def barycentric_to_cartesian_2D(p, vertices):
  """
  vertices must be 3 vertices in cartesian
  """
  return np.dot(p, vertices)



def cart_to_bary_tetra(points: NDArray, vertices: NDArray):
  a, b, c, d = vertices
  T = onp.vstack([b - a, c - a, d - a]).T
  T_inv = onp.linalg.inv(T)
  bary_coords = onp.dot(points - a, T_inv)
  # adding l4
  # TODO axis was originally set to axis=1, but changed to axis=-1. Confirm this
  return onp.hstack((1 - bary_coords.sum(axis=-1, keepdims=True), bary_coords))

def bary_to_cart(points: NDArray, vertices: NDArray) -> NDArray:
  return sum(points[:, i][:, None] * v for i, v in enumerate(vertices))

def barycentric_weights(point:NDArray, normalized=True) -> NDArray:
  """
  Weights based on product of barycentric coordinates: 
  if along edge, one of the coordinates == 0, 
  so distribution value will be 0
  """
  weights = np.prod(point, axis=-1)
  if normalized:
    return weights / np.sum(weights)
  return weights

def calculate_barycentric_coordinates(vertices: NDArray, points: NDArray):
  # A = np.vstack([vertices.T, np.ones(vertices.shape[0])])
  # b = np.hstack([points, np.ones((points.shape[0],1))]).T
  A = np.vstack([np.ones(vertices.shape[0]), vertices.T])
  b = np.hstack([np.ones((points.shape[0], 1)), points]).T
  barycentric_coords, r, _, _ = np.linalg.lstsq(A, b, rcond=None)
  if np.any(r):
    print(f"barycentric residual: {r}")
  return barycentric_coords.T


@dcls.dataclass
class Sphere:
  @staticmethod
  def to_sphere(points):
    points = np.hstack((points, -np.ones((points.shape[0], 1))))
    return points / np.linalg.norm(points, axis=1, keepdims=True)
  
  @staticmethod
  def to_plane(points):
    scale_factor = -1/points[:,-1]
    return points[:,:-1] * scale_factor[:,None]



def stereographic_projection(points):
  # denom = points[:,-1, None]+1
  denom = 1-points[:,-1, None]
  denom = np.where(denom==0, 1, denom)
  return 2*points[:,:-1] / denom

def inv_stereo_proj(*xi):
  denom = 4 + np.sum([x**2 for x in xi], axis=0)
  proj_coords = np.vstack([(4*x)/denom for x in xi]+[(denom-8)/denom]).T
  return proj_coords

def cart_to_sph(coords):
  x = coords[:, 0]
  y = coords[:, 1]
  z = coords[:, 2]
  r = np.sqrt(x**2 + y**2 + z**2)
  r = np.linalg.norm(coords, axis=1)
  theta = np.arccos(z / r)
  phi = np.arctan2(y, x)
  return np.vstack((theta, phi)).T

def sph_to_cart(coords):
  theta = coords[:, 0]
  phi = coords[:, 1]
  x = np.sin(theta) * np.cos(phi)
  y = np.sin(theta) * np.sin(phi)
  z = np.cos(theta)
  return np.vstack((x, y, z)).T
  # return np.column_stack((x, y, z))






def geodesic(p1, p2, num_points=100):
  p1 /= npla.norm(p1)
  p2 /= npla.norm(p2)
  t = np.linspace(0, 1, num_points)[:, None]
  theta = np.arccos(np.dot(p1, p2))
  sin_theta = np.sin(theta)
  if sin_theta==0: 
    print(f"geodesic(): divide by zero: sin(theta)=0, cos(theta)={np.dot(p1, p2)}")
    sin_theta=1
  
  return (np.sin((1 - t) * theta) / sin_theta) * p1 + (np.sin(t * theta) / sin_theta) * p2


def dist_to_geodesic(p,v1,v2):
  p /= npla.norm(p)
  n = np.cross((v1 / npla.norm(v1)), (v2 / npla.norm(v2)))
  n /= npla.norm(n)
  theta = np.arccos(np.dot(p,n))
  return np.pi / 2 - theta 
