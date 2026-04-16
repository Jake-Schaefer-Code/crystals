# coordinates.py
from __future__ import annotations

import numpy as np

NDArray = np.ndarray

def barycentric_coordinates2D(p: NDArray, triangle):    
  """
  Parameters
  ----------------
  p
  
  a
  
  b
  
  c
  """
  a,b,c = triangle[:3]
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
  l3 = 1- l1 - l2
  return np.vstack([l1, l2, l3]).T

def barycentric_to_cartesian_2D(p, vertices):
  """
  vertices must be 3 vertices in cartesian
  """
  return np.dot(p, vertices)



def cart_to_bary3D(points, vertices):
  """

  Parameters
  ----------------
  
  """
  a,b,c,d = vertices
  T = np.vstack([b - a, c - a, d - a]).T
  T_inv = np.linalg.inv(T)
  bary_coords = np.dot(points - a, T_inv)
  # adding l4
  # TODO axis was originally set to axis=1, but changed to axis=-1. Confirm this
  bary_coords = np.hstack((1 - bary_coords.sum(axis=-1, keepdims=True), bary_coords))
  return bary_coords

def bary_to_cart(points, vertices) -> NDArray:
  """

  Parameters
  ----------------
  points

  vertices
  """
  return sum([points[:, i][:, None]*v for i, v in enumerate(vertices)])

def barycentric_weights(point:NDArray, normalized=True) -> NDArray:
  """
  Weights based on product of barycentric coordinates: 
  if along edge, one of the coordinates == 0, 
  so distribution value will be 0

  Parameters
  ----------------
  point:NDArray
  """
  weights = np.prod(point, axis=-1)
  if normalized:
    return weights / np.sum(weights)
  else:
    return weights

def calculate_barycentric_coordinates(vertices: NDArray, points: NDArray):
  """
  Parameters
  ----------------
  vertices

  points
  """
  # A = np.vstack([vertices.T, np.ones(vertices.shape[0])])
  # b = np.hstack([points, np.ones((points.shape[0],1))]).T
  A = np.vstack([np.ones(vertices.shape[0]), vertices.T])
  b = np.hstack([np.ones((points.shape[0], 1)), points]).T
  barycentric_coords, r, _, _ = np.linalg.lstsq(A, b, rcond=None)
  if np.any(r):
      print(f"barycentric residual: {r}")
  return barycentric_coords.T




def to_sphere(points):
  points = np.hstack((points, -np.ones((points.shape[0], 1))))
  return points / np.linalg.norm(points, axis=1, keepdims=True)

def from_sphere(points):
  scale_factor = -1/points[:,-1]
  return points[:,:-1] * scale_factor[:,None]

def stereographic_projection(points):
  # denom = points[:,-1, None]+1
  denom = 1-points[:,-1, None]
  denom = np.where(denom==0, 1, denom)
  return 2*points[:,:-1] / denom

def inv_stereo_proj(*xi):
  """
  """
  denom = 4 + np.sum([x**2 for x in xi], axis=0)
  proj_coords = np.vstack([(4*x)/denom for x in xi]+[(denom-8)/denom]).T
  return proj_coords

def cart_to_sph(coords):
  """

  Parameters
  ----------------
  
  """
  x = coords[:, 0]
  y = coords[:, 1]
  z = coords[:, 2]
  r = np.sqrt(x**2 + y**2 + z**2)
  theta = np.arccos(z / r)
  phi = np.arctan2(y, x)
  return np.vstack((theta, phi)).T

def sph_to_cart(coords):
  """

  Parameters
  ----------------
  
  """
  theta = coords[:, 0]
  phi = coords[:, 1]
  x = np.sin(theta) * np.cos(phi)
  y = np.sin(theta) * np.sin(phi)
  z = np.cos(theta)
  return np.vstack((x, y, z)).T
  # return np.column_stack((x, y, z))
