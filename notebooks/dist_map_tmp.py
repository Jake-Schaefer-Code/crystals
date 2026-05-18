# dist_map.py
from os import environ
from pathlib import Path
import sys
environ["OPENBLAS_NUM_THREADS"] = "1"
environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
_project_root = Path.cwd().resolve()
sys.path.insert(0, str(_project_root.parents[0]))
import dataclasses as dcls
from collections.abc import Callable
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as onp
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import src.crystal_funcs as cfuncs
from plotting_utils import plot_func, plot_result, plot_simplex, plot_weighted_dist
from TriMap import TriMap, define_distribution
import src.coordinates as cconv
import src.symmetry as sym
from src.categorization import print_symmetry_operations
from src.bases import CanonBasis3, CanonBasis2

onp.set_printoptions(precision=2, suppress=True)
NDArray = onp.ndarray




@dcls.dataclass(frozen=True)
class Config:
	seed: int = 0
	res_2d: int = 200
	res_3d: int = 50

	n_samples_fast: int = 5_000
	n_samples_plot: int = 10_000



cmap = plt.get_cmap("afmhot").copy()
cmap.set_bad(color="whitesmoke")


cfg = Config()
B = CanonBasis3()
B2 = CanonBasis2()


def func6(*xi):
	x,y = xi
	return x**6 + x**5*y + x**4*y**2 + x**3*y**3 + x**2*y**4 + x*y**5 + y**6


def func5(*xi):
	x,y = xi
	return x**5 + x**4*y + x**3*y**2 + x**2*y**3 + x*y**4 + y**7

def func4(*xi):
	x,y = xi
	return x**4 + x**3*y + x**2*y**2 + x*y**4 + y**7

def func3(*xi):
	x,y = xi
	return x**3 + x**2*y + x*y**2 + y**5

def func2(*xi):
	x,y = xi
	return onp.cos(2 * onp.pi * x**2) + onp.sin(onp.pi*y**2)

def func_xy(*xi):
	x, y = xi
	return x * y

ndim = 2
_0 = onp.zeros(ndim)
domain = onp.array([(0, 1)] * ndim)
center = 0.5 * (domain[:, 1] - domain[:, 0])

# ---- build grid
xi = onp.array([
	onp.linspace(d[0], d[1], cfg.res_2d, endpoint=False) for d in domain
], dtype=onp.float32) 

# center
xi = xi - center[:, None]

grid = onp.meshgrid(*xi)
coords = onp.vstack([x.ravel() for x in grid]).T + center
coords.astype(onp.float32)


# ---- Group operations
# rot_order = 4
# p4 = [sym.rotation2d(i * 2 * onp.pi / rot_order) for i in range(rot_order)]
p4 = sym.cyclic_group(order=4, dim=ndim)
# Glide reflections
gl_ref = sym.glide_reflection
g = [
	gl_ref(B2.e1, center),
	gl_ref(B2.e2, center),
	gl_ref(B2.e1 - B2.e2, center),
	gl_ref(B2.e1 + B2.e2, center),
]

# Reflections
ref = sym.reflection
m = [
	ref(B2.e2),
	ref(B2.e1),
	ref(B2.e1 - B2.e2),
	ref(B2.e1 + B2.e2),
]

# TODO better way to combine groups
Gp4 = sym.FiniteGroupAction(tuple(p4), name="p4")
Gp4m = sym.FiniteGroupAction((*p4, *m), name="p4m")
Gp4gm = sym.FiniteGroupAction((*p4, *g, *m), name="p4gm")
print_symmetry_operations(Gp4gm)
plot_func(func_xy, *grid, center=center, sym_ops=[Gp4gm, Gp4m, Gp4])

# TODO compute these from the symmetry group
fundamental_units = [
	0.5 * onp.array([(1, 0), (1, 1), (0, 1)]),
	0.5 * onp.array([(1, 0), (1, 1), (0, 1)]),
	0.5 * onp.array([(1, 0), (1, 1), (0, 0)]),
	0.5 * onp.array([(1, 0), (1, 1), (0, 1)]),
]

test_polygon_verts: list[tuple[float, float]] = [
	(1, 0),
	(0.62, 0.78),
	(-0.22, 1),
	(-0.90, 0.2),
	(-0.90, -0.43),
	(-0.22, -0.97),
	(0.62, -0.78),
]

test_unit_verts = [
	(0.5, 0), 
	(0.5, 0.5), 
	(0, 0.5)
]

test_unit = onp.array(test_unit_verts)
test_unit_cent = cfuncs.centroid(test_unit)
polygon = onp.array(test_polygon_verts)

f_sym = sym.symmetrize(func_xy, *grid, sym_ops=[*p4, *g, *m])
plot_weighted_dist(f_sym, coords.T, test_unit, n_samples=cfg.n_samples_plot, rng=cfg.seed + 1)


# ---- 3D

def func3D(*xi):
	x,y,z = xi
	return onp.cos(2*onp.pi*x**2) + onp.sin(onp.pi*y**2) + onp.cos(2*onp.pi*z**2) 

def func3D2(*xi):
	x,y,z = xi
	return onp.sin(2 * onp.pi * x) * onp.sin(2 * onp.pi * y) * onp.sin(2 * onp.pi * z)

def func_activation(*xi, n, func):
	return onp.prod([func(x) for x in xi], axis=0)

def func_sin(*xi, n):
	return onp.prod([onp.sin(2 * onp.pi / n * x) for x in xi], axis=0)

def func_cos(*xi, n):
	return onp.prod([onp.cos(2 * onp.pi / n * x) for x in xi], axis=0)

def func_xyz(*xi):
	x,y,z = xi
	return x * y * z

ndim = 3
# ndim-unit cube
domain = onp.array([(0, 1)]*ndim)
center = 0.5 * (domain[:,1] - domain[:,0])
# zero element
_0 = onp.zeros(ndim)

N = cfg.res_3d
xi = onp.array([onp.linspace(d[0], d[1], N, endpoint=False) for d in domain]) - center[:, None]
grid = onp.meshgrid(*xi)
coords = onp.vstack([x.ravel() for x in grid]).T + center

# ---- Rotations ----
rot_order = 4
dθ = 2 * onp.pi / rot_order
qRM = sym.rotation3d
# p4 = [qRM(i * dθ, B.e3) for i in range(1, rot_order)]
# p4.append(sym.identity(ndim))
p4 = sym.cyclic_group(order=4, dim=ndim, axis=B.e3)


# Glide Reflections
gl_ref = sym.glide_reflection
g =[
	gl_ref(B.e1, center),
	gl_ref(B.e2, center),
	gl_ref(B.e1 - B.e2, center),
	gl_ref(B.e1 + B.e2, center),
]

# Reflections
ref = lambda n: sym.reflection(n)
m = [
	ref(B.e1),
	ref(B.e2),
	ref(B.e1 - B.e2),
	ref(B.e1 + B.e2),
]
Gp4 = sym.FiniteGroupAction(tuple(p4), name="p4")
Gp4m = sym.FiniteGroupAction((*p4, *m), name="p4m")
Gp4gm = sym.FiniteGroupAction((*p4, *g, *m), name="p4gm")

import src.named_groups as groups
A4 = groups.tetrahedral_group()

plot_func(func_xyz, *grid, center=center, sym_ops=[Gp4gm, Gp4m, Gp4, A4])

# Reflections
m = [
	ref(B.e1),
	ref(B.e2),
	ref(B.e3),
]

rot_order = 3
dtheta3 = 2 * onp.pi / rot_order
qRM = sym.rotation3d
rot_3fold_1 = [qRM(i * dtheta3, B.e1 + B.e2 + B.e3) for i in range(1, rot_order)]
rot_3fold_2 = [qRM(i * dtheta3, B.e1 - B.e2 + B.e3) for i in range(1, rot_order)]
rot_3fold_3 = [qRM(i * dtheta3, - B.e1 + B.e2 + B.e3) for i in range(1, rot_order)]
rot_3fold_4 = [qRM(i * dtheta3, - B.e1 - B.e2 + B.e3) for i in range(1, rot_order)]

rot_order = 2
dtheta2 = 2 * onp.pi / rot_order
rot_2fold_x = [qRM(i * dtheta2, B.e1) for i in range(1, rot_order)]
rot_2fold_y = [qRM(i * dtheta2, B.e2) for i in range(1, rot_order)]
rot_2fold_z = [qRM(i * dtheta2, B.e3) for i in range(1, rot_order)]


group = sym.FiniteGroupAction(
	(
		sym.identity(ndim),
		*m,
		*rot_2fold_x, *rot_2fold_y, *rot_2fold_z,
		*rot_3fold_1, *rot_3fold_2, *rot_3fold_3, *rot_3fold_4,
		sym.identity(ndim, inversion=True)
	),
	name="????"
)
plot_func(func_xyz, *grid, center=center, sym_ops=[group])