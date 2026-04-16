# legacy/old.py
#
# Kept for reference while the research helpers are condensed. Do not add new
# code here; active barycentric utilities live in coordinates.py, and the useful
# plotting diagnostics from this file have been migrated to plotting_utils.py.
from __future__ import annotations

from os import environ

environ["OPENBLAS_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull, Delaunay
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture

# Local Imports
from TriMap import TriMap
from src.crystal_funcs import *
from plotting_utils import *


_cmap = plt.get_cmap('afmhot').copy()
_cmap.set_bad(color='whitesmoke')

# TODO for final implementation this will be faster
def barycentric_coordinates2D(p, triangle):    
    """
    Parameters
    ----------------
    p
    
    triangle
    """
    a,b,c = triangle[0], triangle[1], triangle[2]
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
    l3 = 1-l1-l2
    # NOTE for this to work with `barycentric_to_cartesian_2D`, it must be np.vstack([l1, l2, l3]).T
    return np.vstack([l3, l1, l2]).T


def barycentric_to_cartesian_2D(p, vertices):
    """
    vertices must be 3 vertices in cartesian
    """
    return np.dot(p, vertices)


def plot_polygon(vertices, ax, **kwargs):
    """
    Parameters
    ----------------
    """
    polygon = plt.Polygon(vertices[:,:2], closed=True, fill=True, edgecolor=kwargs.get('edgecolor', None), alpha=kwargs.get('alpha', 1))
    ax.add_patch(polygon)
    ax.plot(vertices[:, 0], vertices[:, 1], 'o', 
            color=kwargs.get('color', 'k'), 
            ms=4)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')

def plot_polyhedron(vertices, faces):
    """
    Parameters
    ----------------
    """
    fig = plt.figure()
    ax:Axes3D = fig.add_subplot(111, projection='3d')
    poly3d = [[vertices[vertex] for vertex in face] for face in faces]
    poly_collection = Poly3DCollection(poly3d, facecolors='cyan', linewidths=1, edgecolors='r', alpha=0.25)
    ax.add_collection3d(poly_collection)
    ax.scatter(*vertices.T, color='black')
    max_range = max(vertices.max(axis=0) - vertices.min(axis=0)) / 2.0
    mid = (vertices.max(axis=0) + vertices.min(axis=0)) * 0.5
    bounds = (min(mid-max_range), max(mid+max_range))
    setup_3D_axes(ax, bounds)
    plt.show()
    
    
def plot_transformation(polygon, transformed_polygon, ax, **kwargs):
    """
    Parameters
    ----------------
    """
    plot_polygon(polygon, ax, alpha=0.5, color='k')
    plot_polygon(transformed_polygon, ax, color='r')
    ax.quiver(polygon[:,0], polygon[:,1], 
              transformed_polygon[:,0]-polygon[:,0], transformed_polygon[:,1]-polygon[:,1], 
              angles='xy', scale_units='xy', scale=1, alpha=0.5)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.plot(0,0,'o', color='k', ms=1)


def plot_result(map:TriMap, idx:int):
    best_map = map.maps[idx]
    best_poly_oriented = map.rotated_polygons[idx]
    best_matrices = map.matrices[idx]
    inv_mats = np.array([np.linalg.inv(M) for M in best_matrices])
    dist = barycentric_to_cartesian_2D(map.distribution, np.roll(map.triangle,-1,axis=0))
    dst_idxs = get_triangles(best_map, dist)
    trans_dst = np.array([apply_affine_mat(pt, inv_mats[idx]) for pt, idx in zip(dist, dst_idxs)])
    fig, ax = plt.subplots(1, 2, figsize=(16, 16))
    plot_transformation(best_poly_oriented, best_map, ax[0])
    # plot_transformation(best_map, best_poly_oriented, ax[1])
    best_tri = map.triangulations[idx]
    for t in best_tri.simplices:
        plot_polygon(best_poly_oriented[t], ax[1], alpha=0.1, edgecolor='k')

    ax[0].scatter(dist[:, 0], 
                  dist[:, 1], 
                  s=10, 
                  c=map.distribution_weights, 
                  cmap=_cmap, 
                  alpha=0.5)
    ax[1].scatter(trans_dst[:, 0], 
                  trans_dst[:, 1], 
                  s=10, 
                  c=map.distribution_weights, 
                  cmap=_cmap, 
                  alpha=0.5)
    plt.show()

def plot_result2(best_map: np.ndarray,
                best_mats: np.ndarray,
                best_poly: np.ndarray,
                best_tri: Delaunay,
                dist: np.ndarray,
                dist_weights: np.ndarray,
                **kwargs):
    """
    Parameters
    ----------------
    best_map : np.ndarray,

    best_mats : np.ndarray,

    best_poly : np.ndarray,

    best_tri : np.ndarray,

    dist : np.ndarray,

    dist_weights : np.ndarray,

    **kwargs 
        Any remaining keyword arguments
    """ 


    inv_mats = np.array([np.linalg.inv(M) for M in best_mats])
    # dist = bary_to_cart(map.distribution, np.roll(map.triangle,-1,axis=0))
    dst_idxs = get_triangles(best_map, dist)
    trans_dst = np.array([apply_affine_mat(pt, inv_mats[idx]) for pt, idx in zip(dist, dst_idxs)])
    fig, ax = plt.subplots(1, 2, figsize=(16, 16))
    plot_transformation(best_poly, best_map, ax[0])
    # plot_transformation(best_map, best_poly_oriented, ax[1])

    for t in best_tri.simplices:
        plot_simplex(best_poly[t], ax[1], alpha=0.1, edgecolor='k')
        
    s = kwargs.get("s", 10)
    alpha = kwargs.get("alpha", 0.5)
    cmap = kwargs.get("cmap", _cmap)
    ax[0].scatter(*dist.T, 
                  s=s, 
                  c=dist_weights, 
                  cmap=cmap, 
                  alpha=alpha)
    ax[1].scatter(*trans_dst.T, 
                  s=s, 
                  c=dist_weights, 
                  cmap=cmap, 
                  alpha=alpha)
    
    ax[1].set_xlim(-1.1, 1.1)
    ax[1].set_ylim(-1.1, 1.1)
    ax[1].set_aspect('equal')
    plt.show()


def plot_contours(X, Y, center, data, fund_unit, ax):
    ax.contourf(X + center[0], Y + center[1], data, levels=100)
    patch = plt.Polygon(fund_unit, fill=False, edgecolor='k')
    ax.add_patch(patch)

def plot_density(coords, weights, fund_unit, ax):
    kde = gaussian_kde([*coords.T], weights=weights)
    X_grid, Y_grid = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    Z_grid = kde([X_grid.ravel(), Y_grid.ravel()]).reshape(X_grid.shape)
    cf = ax.contourf(X_grid, Y_grid, Z_grid, levels=100, cmap=_cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(cf, cax=cax, orientation='vertical')
    patch = plt.Polygon(fund_unit, fill=False, edgecolor='k')
    ax.add_patch(patch)

def plot_symmetry_op(X: np.ndarray, 
                     Y: np.ndarray, 
                     domain, 
                     group_data, 
                     fund_unit, 
                     dirichlet_pts, 
                     dirichlet_weights, 
                     n_samples, 
                     group_name, 
                     axes):
    """
    Parameters
    ----------------
    X: np.ndarray, 
    Y: np.ndarray, 
    domain, 
    group_data, 
    fund_unit, 
    dirichlet_pts, 
    dirichlet_weights, 
    n_samples, 
    group_name, 
    axes
    """
    center = 0.5 * (domain[:,1] - domain[:,0])
    coords = np.vstack([X.ravel(), Y.ravel()]).T + center
    cart_pts = barycentric_to_cartesian_2D(dirichlet_pts, fund_unit)

    
    inside_coords = coords[isinside(coords, fund_unit)]
    weights = distance_weights(inside_coords, cart_pts, w=dirichlet_weights)
    n_comp = 4 if group_name == 'p4gm' else 1
    gmm = GaussianMixture(n_components=n_comp, max_iter=1000)
    probabilities = group_data.ravel()**2
    probabilities /= np.sum(probabilities)
    dist = np.random.choice(len(coords), size=n_samples, p=probabilities)
    gmm.fit(coords[dist])
    samples, _ = gmm.sample(n_samples)
    inside_samples = samples[isinside(samples, fund_unit)]
    inside_weights = distance_weights(inside_coords, inside_samples)
    total_weights = weights * inside_weights
    total_weights /= np.sum(total_weights)    
    filtered_coords = inside_coords[total_weights > 0]
    filtered_weights = total_weights[total_weights > 0]
    plot_contours(X, Y, center, group_data, fund_unit, axes[0])
    plot_density(filtered_coords, filtered_weights, fund_unit, axes[1])
    for ax in axes:
        setup_2D_axes(ax, title=group_name, bounds=(0,0.99))
