# physics/spectral_curves.py
r"""Spectral curves: the Stieltjes transform of an eigenvalue distribution as a branch of a curve.

Numerics for the dictionary

  eigenvalue distribution            spectral curve P(s, z) = 0
  -----------------------            --------------------------
  support                            branch cuts, where two sheets are glued
  square-root edges                  simple branch points, locally y^2 = x (Airy)
  k intervals                        genus k - 1 (two disks -> sphere, two annuli -> torus)
  filling fractions                  periods of y dx around A-cycles
  free convolution                   resultants of bivariate polynomials
  1/N^2 corrections                  topological recursion on the curve
  finite N                           poles of -p'/(n p), which condense into the cut

Conventions
-----------
Stieltjes transform s(z) = int rho(x) dx / (x - z) ~ -1/z, as in S(z) = tr (A - z)^{-1} / n.
Cauchy transform G(z) = -s(z) ~ 1/z (free probability, loop equations). Inversion:
rho(x) = Im s(x + i0) / pi = -Im G(x + i0) / pi. The semicircle has variance 1, support [-2, 2]
and s^2 + z s + 1 = 0. Matrix models have density ~ exp(-N tr V(M)) and resolvent W = G, and
y = V'(x) - 2 W(x) solves y^2 = V'(x)^2 - 4 P(x), so rho = Im y(x + i0) / (2 pi) on the cuts.
Random matrices are normalized so that E|M_ij|^2 = 1/N: their spectra fill [-2, 2].

The figures live in ``plotting/spectral_curves.py``; nothing here draws.
"""
from __future__ import annotations

import dataclasses as dcls
import functools
import itertools as it
import math
from fractions import Fraction

import numpy as onp
from scipy import integrate, linalg, optimize, special
import sympy as sp

P = onp.polynomial.polynomial   # coefficient arrays run from low to high degree




















# --------------------------------------------------------------------------- #
# Random matrices
# --------------------------------------------------------------------------- #

def goe(n, rng=None):
  r"""GOE with E M_ij^2 = 1/n off the diagonal (2/n on it): spectrum -> semicircle on [-2, 2]."""
  a = onp.random.default_rng(rng).standard_normal((n, n))
  return (a + a.T) / onp.sqrt(2 * n)


def gue(n, rng=None, size=None):
  r"""GUE with E|M_ij|^2 = 1/n; ``size`` stacks independent samples along a leading axis."""
  rng = onp.random.default_rng(rng)
  shape = (n, n) if size is None else (size, n, n)
  a = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
  return (a + onp.conj(onp.swapaxes(a, -1, -2))) / (2 * onp.sqrt(n))


def haar_orthogonal(n, rng=None):
  r"""Haar-distributed orthogonal matrix (QR of a Gaussian matrix with the sign fix)."""
  q, r = onp.linalg.qr(onp.random.default_rng(rng).standard_normal((n, n)))
  return q * onp.sign(onp.diag(r))


def beta_hermite_top(n, beta, size, *, k=1, rng=None):
  r"""Largest ``k`` eigenvalues of ``size`` beta-Hermite matrices, scaled to fill [-2, 2].

  Dumitriu–Edelman: the tridiagonal matrix with N(0, 2) diagonal and chi_{beta (n-1)}, ...,
  chi_beta off the diagonal, over sqrt(2), has eigenvalue density
  prod |l_i - l_j|^beta exp(-sum l_i^2 / 2), i.e. GOE / GUE / GSE for beta = 1 / 2 / 4. Each
  eigenvalue costs O(n), so edge statistics at n ~ 10^3 are cheap.
  """
  rng = onp.random.default_rng(rng)
  dfs = beta * onp.arange(n - 1, 0, -1)
  out = onp.empty((size, k))
  for j in range(size):
    d = rng.normal(0.0, onp.sqrt(2.0), n)
    e = onp.sqrt(rng.chisquare(dfs))
    out[j] = linalg.eigvalsh_tridiagonal(d, e, select="i", select_range=(n - k, n - 1))
  return out / onp.sqrt(beta * n)


# --------------------------------------------------------------------------- #
# 1. The semicircle as a curve
# --------------------------------------------------------------------------- #

def semicircle_density(x):
  x = onp.asarray(x, float)
  return onp.sqrt(onp.clip(4 - x**2, 0, None)) / (2 * onp.pi)


def semicircle_s(z, sheet=1):
  r"""Roots of s^2 + z s + 1 = 0. Sheet 1 is the Stieltjes transform (|s| < 1, s ~ -1/z) and
  sheet 2 is 1/s. The product sqrt(z - 2) sqrt(z + 2) has its cut exactly on [-2, 2]."""
  z = onp.asarray(z, complex)
  root = onp.sqrt(z - 2) * onp.sqrt(z + 2)
  return (-z + root) / 2 if sheet == 1 else (-z - root) / 2


def joukowski(s):
  r"""z = -(s + 1/s): the rational parametrization of s^2 + z s + 1 = 0."""
  return -(s + 1 / s)


def stieltjes(eigs, z):
  r"""Empirical Stieltjes transform (1/n) sum_i 1 / (lambda_i - z) = -(1/n) p'(z) / p(z)."""
  z = onp.asarray(z, complex)
  out = onp.zeros_like(z)
  for lam in onp.asarray(eigs):
    out += 1.0 / (lam - z)
  return out / len(eigs)


def critical_points(eigs):
  r"""Zeros of p' for p(x) = prod (x - lambda_i). Between consecutive roots p'/p = sum 1/(x - l_i)
  falls from +inf to -inf, so each gap holds exactly one (Rolle); repeated roots are kept."""
  lam = onp.sort(onp.asarray(eigs, float))
  f = lambda x: onp.sum(1.0 / (x - lam))
  out = []
  for lo, hi in zip(lam[:-1], lam[1:]):
    if hi - lo < 1e-12:
      out.append(lo)
    else:
      d = 1e-9 * (hi - lo)
      out.append(optimize.brentq(f, lo + d, hi - d, xtol=1e-14))
  return onp.array(out)










# --------------------------------------------------------------------------- #
# 2. Edges: square roots, Airy, Tracy–Widom
# --------------------------------------------------------------------------- #

def edge_density(n, edges, *, size, k=12, beta=2, rng=None):
  r"""Density of the top ``k`` eigenvalues of ``size`` beta-Hermite matrices in xi = n^{2/3} (x - 2),
  histogrammed on ``edges``: eigenvalues per unit xi, summed over the ``k`` kept, per matrix."""
  top = beta_hermite_top(n, beta, size, k=k, rng=rng)
  counts, _ = onp.histogram(n ** (2 / 3) * (top - 2), bins=edges)
  return counts / (size * onp.diff(edges))


def scaled_top_eigenvalue(n, beta, size, *, rng=None):
  r"""``size`` samples of n^{2/3} (lambda_max - 2), which converge to Tracy–Widom F_beta."""
  return n ** (2 / 3) * (beta_hermite_top(n, beta, size, rng=rng)[:, -1] - 2)


def airy_asymptotics(xi):
  r"""WKB forms of Ai: ``(xi_neg, oscillatory, xi_pos, decaying)`` on the parts of ``xi`` with
  |xi| > 0.4, sin(2/3 |xi|^{3/2} + pi/4) / (sqrt(pi) |xi|^{1/4}) and exp(-2/3 xi^{3/2}) / (2 sqrt(pi) xi^{1/4})."""
  xi = onp.asarray(xi, float)
  neg, pos = xi[xi < -0.4], xi[xi > 0.4]
  a = onp.abs(neg)
  return (neg, onp.sin(2 / 3 * a**1.5 + onp.pi / 4) / (onp.sqrt(onp.pi) * a**0.25),
          pos, onp.exp(-2 / 3 * pos**1.5) / (2 * onp.sqrt(onp.pi) * pos**0.25))


def airy_kernel_density(xi):
  r"""K_Ai(xi, xi) = Ai'(xi)^2 - xi Ai(xi)^2: eigenvalues per unit xi = N^{2/3} (x - 2)."""
  ai, aip, _, _ = special.airy(xi)
  return aip**2 - xi * ai**2


def tracy_widom(s, beta=2, *, m=64, h=1e-3):
  r"""(F_beta(s), F_beta'(s)) for beta = 1, 2 by Bornemann's quadrature of Fredholm determinants.

  F_2(s) = det(I - K_Ai) and F_1(s) = det(I - Ai((x + y)/2) / 2), both on L^2(s, oo), with
  m-point Gauss–Legendre on [s, s + 14 + max(0, -s)] (the kernels vanish beyond); the density is
  a centered difference.
  """
  nodes, weights = onp.polynomial.legendre.leggauss(m)

  def cdf(si):
    b = si + 14.0 + max(0.0, -si)
    x = 0.5 * (b - si) * nodes + 0.5 * (b + si)
    sw = onp.sqrt(0.5 * (b - si) * weights)
    if beta == 2:
      ai, aip, _, _ = special.airy(x)
      with onp.errstate(divide="ignore", invalid="ignore"):
        K = (onp.outer(ai, aip) - onp.outer(aip, ai)) / (x[:, None] - x[None, :])
      K[onp.diag_indices(m)] = aip**2 - x * ai**2
    elif beta == 1:
      K = 0.5 * special.airy(0.5 * (x[:, None] + x[None, :]))[0]
    else:
      raise ValueError("beta must be 1 or 2")
    return linalg.det(onp.eye(m) - sw[:, None] * K * sw[None, :])

  s = onp.atleast_1d(onp.asarray(s, float))
  F = onp.array([cdf(si) for si in s])
  f = onp.array([(cdf(si + h) - cdf(si - h)) / (2 * h) for si in s])
  return F, f






# --------------------------------------------------------------------------- #
# 3. Matrix models: hyperelliptic curves, genus, filling fractions
# --------------------------------------------------------------------------- #

@dcls.dataclass(frozen=True)
class QuarticModel:
  r"""Matrix model exp(-N tr V(M)) with V(x) = g x^4 / 4 + a x^2 / 2 + t x and g > 0."""
  g: float = 1.0
  a: float = 0.0
  t: float = 0.0

  def V(self, x):
    return self.g * x**4 / 4 + self.a * x**2 / 2 + self.t * x

  def Vp(self, x):
    return self.g * x**3 + self.a * x + self.t


@dcls.dataclass(frozen=True, eq=False)
class SpectralCurve:
  r"""Physical branch y(x) = M(x) prod_i sqrt(x - e_i) of y^2 = F(x) = V'(x)^2 - 4 P(x).

  With principal square roots the product has cuts exactly on [e_1, e_2], [e_3, e_4], ... and
  grows like x^k, so y ~ V'(x) - 2/x at infinity and y = V_eff'(x) on the real axis off the cuts.
  The zeros of M are double points (nodes) of y^2 = F; the smooth curve ytilde^2 = prod (x - e_i)
  left after removing them has genus k - 1.
  """
  model: QuarticModel
  edges: onp.ndarray   # e_1 < ... < e_{2k}
  M: onp.ndarray       # low -> high

  @property
  def cuts(self):
    return list(zip(self.edges[::2], self.edges[1::2]))

  @property
  def genus(self):
    return len(self.edges) // 2 - 1

  @property
  def nodes(self):
    return P.polyroots(self.M) if len(self.M) > 1 else onp.array([], complex)

  def sqrt_disc(self, x):
    r"""prod_i sqrt(x - e_i) with principal roots: analytic off the support, ~ x^k at infinity."""
    x = onp.asarray(x, complex)
    out = onp.ones_like(x)
    for e in self.edges:
      out = out * onp.sqrt(x - e)
    return out

  def y(self, x):
    x = onp.asarray(x, complex)
    return P.polyval(x, self.M) * self.sqrt_disc(x)

  def F(self, x):
    x = onp.asarray(x, float)
    return P.polyval(x, self.M) ** 2 * onp.prod([x - e for e in self.edges], axis=0)

  def density(self, x):
    return onp.clip(self.y(onp.asarray(x, float) + 0j).imag, 0, None) / (2 * onp.pi)

  def filling_fractions(self):
    r"""The mass of each interval, int_cut rho."""
    return onp.array([integrate.quad(self.density, lo, hi, limit=200)[0] for lo, hi in self.cuts])

  def a_cycle(self, i, n=2048):
    r"""Counterclockwise circle around cut ``i`` that stays clear of the other cuts."""
    lo, hi = self.cuts[i]
    gaps = [lo - b for _, b in self.cuts if b < lo] + [a - hi for a, _ in self.cuts if a > hi]
    margin = min([0.5 * gap for gap in gaps] + [0.35])
    return 0.5 * (lo + hi) + (0.5 * (hi - lo) + margin) * onp.exp(2j * onp.pi * onp.arange(n) / n)

  def a_period(self, i, n=2048):
    r"""-(1 / 4 pi i) oint_{A_i} y dx by the trapezoid rule: the filling fraction of cut ``i``."""
    x = self.a_cycle(i, n)
    dx = 2j * onp.pi / n * (x - x.mean())
    return -onp.sum(self.y(x) * dx) / (4j * onp.pi)

  @property
  def real_node(self):
    r"""The double root d of F nearest the real axis (two-cut curves: the node in the gap)."""
    return self.nodes.real[onp.argmin(onp.abs(self.nodes.imag))]

  def lobe_areas(self):
    r"""int 2 |Re y| dx over the two halves of the gap, split at the node d; equal at equilibrium
    (Maxwell construction), where the B-period vanishes."""
    (_, e2), (e3, _) = self.cuts[:2]
    d = self.real_node
    return [integrate.quad(lambda x: 2 * abs(self.y(x + 0j).real), lo, hi)[0] for lo, hi in ((e2, d), (d, e3))]

  def b_period(self):
    r"""oint_B y dx = 2 int y dx across the first gap; it vanishes iff V_eff is equal on both cuts."""
    (_, e2), (e3, _) = self.cuts[:2]
    return 2 * integrate.quad(lambda x: self.y(x + 0j).real, e2, e3, limit=200)[0]

  @functools.cached_property
  def fermi_level(self):
    r"""The constant value of V_eff(x) = V(x) - 2 int log|x - l| rho(l) dl on the support."""
    e1 = self.edges[0]
    log_pot = sum(integrate.quad(lambda l: onp.log(abs(e1 - l)) * self.density(l), lo, hi,
                                 limit=200)[0] for lo, hi in self.cuts)
    return self.model.V(e1) - 2 * log_pot

  def effective_potential(self, x):
    r"""V_eff(x): flat on the support (no force on an eigenvalue there) with V_eff' = y off it."""
    x = onp.asarray(x, float)
    t = onp.union1d(onp.linspace(min(x.min(), self.edges[0]), max(x.max(), self.edges[-1]), 8001),
                    self.edges)
    integral = integrate.cumulative_trapezoid(self.y(t + 0j).real, t, initial=0.0)
    return self.fermi_level + onp.interp(x, t, integral - onp.interp(self.edges[0], t, integral))


def _one_cut(model):
  r"""Support [c - r, c + r] from the endpoint conditions <V'> = 0, <x V'> = 2, where <f> averages
  f(c + r cos theta) over theta in [0, pi]; then M = polynomial part of V' / sqrt((x-a)(x-b))."""
  g, a, t = model.g, model.a, model.t

  def eqs(p):
    c, r = p
    return [g * (c**3 + 1.5 * c * r**2) + a * c + t,
            g * (c**4 + 3 * c**2 * r**2 + 0.375 * r**4) + a * (c**2 + 0.5 * r**2) + t * c - 2]

  u0 = (-a + onp.sqrt(a * a + 12 * g)) / (1.5 * g)
  (c, r), _, ier, _ = optimize.fsolve(eqs, [0.0, onp.sqrt(u0)], full_output=True, xtol=1e-13)
  if ier != 1:
    return None
  lo, hi = c - abs(r), c + abs(r)
  s, p = lo + hi, lo * hi
  return SpectralCurve(model, onp.array([lo, hi]), onp.array([g * (3 * s * s / 8 - p / 2) + a, g * s / 2, g]))


def _admissible(curve, tol=1e-9):
  r"""rho >= 0 on the support and V_eff >= its support value off it."""
  (lo, hi), = curve.cuts
  if onp.any(P.polyval(onp.linspace(lo, hi, 401), curve.M) < -tol):
    return False
  x_out = onp.concatenate([onp.linspace(lo - 3, lo, 300), onp.linspace(hi, hi + 3, 300)])
  return bool(onp.all(curve.effective_potential(x_out) - curve.fermi_level > -1e-7))


def _two_cut(model, n_scan=240):
  r"""Two-cut solutions form a one-parameter family: F = V'^2 - 4P with P = g x^2 + p1 x + p0 and a
  double root at d fixes (p1, p0) linearly in d. Equilibrium (the B-period vanishing) picks d."""
  g = model.g
  Vp = onp.array([model.t, model.a, 0.0, g])
  dVp = P.polyder(Vp)

  def curve_at(d):
    Vd, dVd = P.polyval(d, Vp), P.polyval(d, dVp)
    p1 = Vd * dVd / 2 - 2 * g * d
    p0 = Vd**2 / 4 - g * d * d - p1 * d
    F = P.polysub(P.polymul(Vp, Vp), 4 * onp.array([p0, p1, g]))
    e = P.polyroots(P.polydiv(F, [d * d, -2 * d, 1])[0] / g**2)
    if onp.max(onp.abs(e.imag)) > 1e-9:
      return None
    e = onp.sort(e.real)
    return SpectralCurve(model, e, onp.array([-g * d, g])) if e[1] < d < e[2] else None

  def gap(d):
    c = curve_at(d)
    return onp.nan if c is None else c.b_period()

  crit = P.polyroots(Vp)
  crit = onp.sort(crit.real[onp.abs(crit.imag) < 1e-9])
  if len(crit) < 3:
    raise ValueError("V has a single well: there is no two-cut solution")
  ds = onp.linspace(crit[0], crit[-1], n_scan)[1:-1]
  vals = onp.array([gap(d) for d in ds])
  flips = onp.flatnonzero(onp.isfinite(vals[:-1]) & onp.isfinite(vals[1:]) & (vals[:-1] * vals[1:] < 0))
  if len(flips) == 0:
    raise ValueError("no two-cut equilibrium found")
  i = flips[0]
  return curve_at(optimize.brentq(gap, ds[i], ds[i + 1], xtol=1e-14))


def spectral_curve(model):
  r"""The equilibrium curve of ``model``: one cut when admissible, otherwise two cuts."""
  curve = _one_cut(model)
  return curve if curve is not None and _admissible(curve) else _two_cut(model)


def abel_infinity(curve):
  r"""u(x = -oo) = -int_{-oo}^{e_1} dx / sqrt|prod (x - e_i)|, the image of infinity under ``abel_map``
  (up to the sign of the real axis it sits on: +-``u_inf``)."""
  e = curve.edges
  return -integrate.quad(lambda x: 1.0 / onp.sqrt(onp.abs(onp.prod([x - ei for ei in e], axis=0))),
                         -onp.inf, e[0])[0]


def sample_log_gas(V, n, *, beta=2.0, window=(-3.0, 3.0), n_chains=32, n_sweeps=400, n_burn=150,
                   step=None, p_global=0.2, rng=None):
  r"""Metropolis samples of the eigenvalues of exp(-N tr V(M)): the beta log-gas.

  The target is prod_{i<j} |l_i - l_j|^beta exp(-(beta/2) n sum_i V(l_i)), whose large-n
  equilibrium measure does not depend on beta (beta = 2: Hermitian matrices). A sweep updates
  every eigenvalue once, in all chains at once: a Gaussian step or, with probability
  ``p_global``, a uniform draw from ``window``. The uniform draws let eigenvalues move between
  wells; local steps essentially never cross a barrier of height ~ n.

  Returns sorted samples of shape (n_sweeps - n_burn, n_chains, n) and the acceptance rate.
  """
  rng = onp.random.default_rng(rng)
  lo, hi = window
  step = (hi - lo) / n if step is None else step
  lam = onp.sort(rng.uniform(lo, hi, (n_chains, n)), axis=1)
  out, accepted = [], 0.0
  for sweep in range(n_sweeps):
    for i in rng.permutation(n):
      old = lam[:, i]
      new = onp.where(rng.random(n_chains) < p_global, rng.uniform(lo, hi, n_chains),
                      old + step * rng.standard_normal(n_chains))
      with onp.errstate(divide="ignore"):
        d_new = onp.log(onp.abs(new[:, None] - lam))
        d_old = onp.log(onp.abs(old[:, None] - lam))
      d_new[:, i] = d_old[:, i] = 0.0
      dE = 0.5 * beta * n * (V(new) - V(old)) - beta * (d_new.sum(1) - d_old.sum(1))
      ok = onp.log(rng.random(n_chains)) < -dE
      lam[ok, i] = new[ok]
      accepted += ok.mean()
    if sweep >= n_burn:
      out.append(onp.sort(lam, axis=1))
  return onp.array(out), accepted / (n_sweeps * n)








def abel_map(curve, *, R=None, nx=801, nh=500, h_min=1e-5):
  r"""Abel map u(x) = int dx / prod sqrt(x - e_i) of a two-cut curve on the upper half-plane.

  u maps the upper half-plane onto the rectangle [-K_B, 0] x [0, K_A]: cut 1 -> Re u = 0,
  cut 2 -> Re u = -K_B, the gap -> Im u = K_A, the real axis outside the support -> Im u = 0.
  The lower half-plane is the mirror image, so sheet I is the cylinder [-K_B, 0] x R/(2 K_A Z)
  and w = exp(pi u / K_A) maps it onto the annulus exp(-pi K_B / K_A) <= |w| <= 1. The periods
  of dx / y are 2 i K_A (A-cycle) and 2 K_B (B-cycle).

  Integrates along the top row x + i R and then down every column; heights are log-spaced so the
  1/sqrt singularities at the branch points are resolved. Returns (X, H, u, K_A, K_B).
  """
  e = curve.edges
  if len(e) != 4:
    raise ValueError("abel_map needs a two-cut curve")
  R = 1.6 * onp.max(onp.abs(e)) if R is None else R
  X = onp.linspace(-R, R, nx)
  H = onp.geomspace(h_min, R, nh)
  f = lambda x: 1.0 / curve.sqrt_disc(x)
  u_top = integrate.cumulative_trapezoid(f(X + 1j * R), X, initial=0.0)
  down = integrate.cumulative_trapezoid(f(X[None, :] + 1j * H[:, None])[::-1], H[::-1], axis=0,
                                        initial=0.0)[::-1]
  u = u_top[None, :] + 1j * down
  j_cut = onp.argmin(onp.abs(X - 0.5 * (e[0] + e[1])))
  j_out = onp.argmin(onp.abs(X - 0.5 * (e[3] + R)))
  u = u - u[0, j_cut].real - 1j * u[0, j_out].imag

  def half_period(lo, hi):
    o1, o2 = [ei for ei in e if ei != lo and ei != hi]
    g = lambda x: 1.0 / onp.sqrt(abs((x - o1) * (x - o2)))
    return integrate.quad(g, lo, hi, weight="alg", wvar=(-0.5, -0.5))[0]

  return X, H, u, half_period(e[0], e[1]), half_period(e[1], e[2])












# --------------------------------------------------------------------------- #
# 4. Free probability: swapping coordinates, resultants
# --------------------------------------------------------------------------- #

G, Z, W, R = sp.symbols("G z w r")


def _exact(v):
  r"""Rational form of a parameter: resultants over floating-point coefficients break down."""
  return sp.nsimplify(v, rational=True)


def semicircle_poly(variance=1):
  r"""Semicircle: variance G^2 - z G + 1 = 0."""
  return _exact(variance) * G**2 - Z * G + 1


def bernoulli_poly(a=1):
  r"""(delta_a + delta_{-a}) / 2: G = z / (z^2 - a^2)."""
  return G * (Z**2 - _exact(a) ** 2) - Z


def _strip(expr, var):
  r"""Drop constants and powers of ``var``: the factors introduced by clearing denominators."""
  _, factors = sp.factor_list(sp.expand(expr))
  return sp.Mul(*[f**k for f, k in factors if not f.free_symbols <= {var}])


def r_curve(L):
  r"""Curve of the R-transform. G(K(w)) = w with K(w) = R(w) + 1/w: swap the coordinates
  (G -> w, z -> K) and shift, i.e. substitute G -> w, z -> r + 1/w."""
  return _strip(sp.numer(sp.together(L.subs({G: W, Z: R + 1 / W}, simultaneous=True))), W)


def g_curve(Lr):
  r"""Inverse of :func:`r_curve`: w -> G, r -> z - 1/G."""
  return _strip(sp.numer(sp.together(Lr.subs({W: G, R: Z - 1 / G}, simultaneous=True))), G)


def physical_factor(L, *, mean=0, z0=1e3j):
  r"""The irreducible factor of L(G, z) with a root G ~ 1/z + mean/z^2 at large z."""
  factors = [f for f, _ in sp.factor_list(sp.expand(L))[1] if f.has(G) and f.has(Z)]

  def miss(f):
    coeffs = [complex(sp.sympify(c).subs(Z, z0)) for c in sp.Poly(f, G).all_coeffs()]
    return onp.min(onp.abs(onp.roots(coeffs) - (1 / z0 + mean / z0**2))) * abs(z0) ** 2

  return sp.expand(min(factors, key=miss))


def free_sum(L1, L2, *, mean=0):
  r"""Curve of mu_1 [+] mu_2 (Rao–Edelman). R-transforms add, so eliminate u from
  L1r(r - u, w) = L2r(u, w) = 0 with a resultant, return to (G, z), and keep the physical factor."""
  u = sp.Dummy("u")
  res = sp.resultant(r_curve(L1).subs(R, R - u), r_curve(L2).subs(R, u), u)
  return physical_factor(g_curve(res), mean=mean)


def cauchy_branch(L, x, *, eps=1e-9, height=20.0, n_steps=90):
  r"""Physical root G(x + i eps) of L(G, z) = 0: start at x + i height, where G ~ 1/z singles it
  out, and follow it down each vertical line taking the nearest root at every step."""
  x = onp.asarray(x, float)
  coeffs = [sp.lambdify(Z, c, "numpy") for c in sp.Poly(L, G).all_coeffs()]

  def roots(zz):
    C = onp.stack([onp.broadcast_to(onp.asarray(c(zz), complex), zz.shape) for c in coeffs], axis=-1)
    C = C[..., 1:] / C[..., :1]
    deg = C.shape[-1]
    comp = onp.zeros(zz.shape + (deg, deg), complex)
    comp[..., 0, :] = -C
    comp[..., onp.arange(1, deg), onp.arange(deg - 1)] = 1.0
    return onp.linalg.eigvals(comp)

  hs = onp.geomspace(height, eps, n_steps)
  zz = x + 1j * hs[0]
  rts = roots(zz)
  g = rts[onp.arange(len(x)), onp.argmin(onp.abs(rts - 1 / zz[:, None]), axis=1)]
  for h in hs[1:]:
    rts = roots(x + 1j * h)
    g = rts[onp.arange(len(x)), onp.argmin(onp.abs(rts - g[:, None]), axis=1)]
  return g


def real_branch(L, x, *, eps=1e-11):
  r"""G(x) of ``cauchy_branch`` where it is real (off the support), nan where it has an imaginary part."""
  g = cauchy_branch(L, x, eps=eps)
  return onp.where(onp.abs(g.imag) < 1e-7, g.real, onp.nan)


def r_transform_branch(L, side, *, z_max=400.0, z_min=0.05, n=1200):
  r"""``(w, R(w))`` on the branch through w = 0: follow G in from z = side * oo until it stops being
  real (the support) or leaves |G| <= 50, with R = z - 1/G."""
  zo = side * onp.geomspace(z_max, z_min, n)
  g = cauchy_branch(L, zo, eps=1e-11)
  stop = onp.flatnonzero((onp.abs(g.imag) > 1e-7) | (onp.abs(g) > 50))
  k = stop[0] if len(stop) else len(zo)
  return g.real[:k], zo[:k] - 1 / g.real[:k]


def bernoulli_r(a, w):
  r"""R-transform of (delta_a + delta_{-a}) / 2: (sqrt(1 + 4 a^2 w^2) - 1) / (2 w)."""
  w = onp.asarray(w, float)
  return (onp.sqrt(1 + 4 * a * a * w * w) - 1) / (2 * w)


def density_from_curve(L, x, eps=1e-8):
  return onp.clip(-cauchy_branch(L, x, eps=eps).imag / onp.pi, 0, None)


def branch_points(L):
  r"""Roots of the discriminant of L in G: where two sheets meet (the support edges among them)."""
  disc = sp.Poly(sp.discriminant(L, G), Z)
  return onp.roots([complex(c) for c in disc.all_coeffs()])






def goe_plus_atoms(W, a):
  r"""Eigenvalues of W + diag(+-a), half the atoms at +a and half at -a: W + A with W rotation
  invariant, so the two are asymptotically free."""
  n = len(W)
  return onp.linalg.eigvalsh(W + onp.diag(onp.where(onp.arange(n) < n // 2, a, -a)))


# --------------------------------------------------------------------------- #
# 5. Maps and the genus expansion
# --------------------------------------------------------------------------- #

def pairings(k):
  r"""All (2k - 1)!! pairings of the 2k sides of a polygon, as involutions ``sigma``."""
  def rec(free):
    if not free:
      yield {}
      return
    i = free[0]
    for j in free[1:]:
      for m in rec([f for f in free if f not in (i, j)]):
        yield {**m, i: j, j: i}
  for m in rec(list(range(2 * k))):
    yield tuple(m[i] for i in range(2 * k))


def gluing_genus(sigma):
  r"""Genus of the surface made by gluing side i of a 2k-gon to side sigma(i), orientably.

  Euler: V - E + F = 2 - 2g with F = 1 face, E = k edges, and V = the number of cycles of
  i -> sigma(i) + 1 (corners that get identified).
  """
  n = len(sigma)
  seen, cycles = [False] * n, 0
  for i in range(n):
    if not seen[i]:
      cycles += 1
      j = i
      while not seen[j]:
        seen[j] = True
        j = (sigma[j] + 1) % n
  return (n // 2 + 1 - cycles) // 2


@functools.lru_cache(maxsize=None)
def harer_zagier(g, k):
  r"""epsilon_g(k), the number of genus-g gluings of a 2k-gon, from the Harer–Zagier recursion
  (k + 1) e_g(k) = 2 (2k - 1) e_g(k - 1) + (k - 1)(2k - 1)(2k - 3) e_{g-1}(k - 2)."""
  if g < 0 or k < 0:
    return 0
  if k == 0:
    return int(g == 0)
  total = 2 * (2 * k - 1) * harer_zagier(g, k - 1) + (k - 1) * (2 * k - 1) * (2 * k - 3) * harer_zagier(g - 1, k - 2)
  return total // (k + 1)


def gue_moments(n, kmax, size, *, rng=None, chunk=20000):
  r"""Monte Carlo estimates of E tr M^{2k} / n for k = 1..kmax, with standard errors."""
  rng = onp.random.default_rng(rng)
  total, total_sq, done = onp.zeros(kmax), onp.zeros(kmax), 0
  while done < size:
    m = min(chunk, size - done)
    lam = onp.linalg.eigvalsh(gue(n, rng, size=m))
    mom = onp.stack([onp.mean(lam ** (2 * k), axis=1) for k in range(1, kmax + 1)], axis=1)
    total += mom.sum(0)
    total_sq += (mom**2).sum(0)
    done += m
  mean = total / size
  return mean, onp.sqrt((total_sq / size - mean**2) / size)








t_ = sp.Symbol("t")


def _series_coeffs(num, den, shift, m, t=t_):
  r"""First m Taylor coefficients of num / (den / t^shift), by inverting the power series of den."""
  Pn, Pd = sp.Poly(num, t), sp.Poly(den, t)
  d = [Pd.coeff_monomial(t ** (shift + k)) for k in range(m)]
  nn = [Pn.coeff_monomial(t**k) for k in range(m)]
  inv = [1 / d[0]]
  for k in range(1, m):
    inv.append(sp.cancel(-sum(d[j] * inv[k - j] for j in range(1, k + 1)) / d[0]))
  return [sp.cancel(sum(nn[i] * inv[j - i] for i in range(j + 1))) for j in range(m)]


def _principal_part(expr, t=t_):
  r"""[c_{-m}, ..., c_{-1}]: the principal part at t = 0 of a rational function of t."""
  num, den = sp.fraction(sp.cancel(sp.together(expr)))
  m = min(k for (k,) in sp.Poly(den, t).monoms())
  return _series_coeffs(num, den, m, m, t) if m > 0 else []


def _taylor(expr, m, t=t_):
  r"""[c_0, ..., c_{m-1}]: Taylor coefficients at t = 0 of a rational function regular there."""
  num, den = sp.fraction(sp.cancel(sp.together(expr)))
  return _series_coeffs(num, den, 0, m, t)


class TopologicalRecursion:
  r"""Eynard-Orantin topological recursion on a rational spectral curve.

  Input: x(z), y(z) rational in a global coordinate z, the ramification points a (dx(a) = 0) and
  an involution sigma with x(sigma z) = x(z) and sigma(a) = a (global and rational here). With
  omega_{0,1} = y dx and omega_{0,2} = dz_1 dz_2 / (z_1 - z_2)^2,

    omega_{g,n+1}(z_0, J) = sum_a Res_{z=a} K(z_0, z) [ omega_{g-1,n+2}(z, sigma z, J)
                           + sum' omega_{h,1+|I|}(z, I) omega_{g-h,1+|J-I|}(sigma z, J - I) ],
    K(z_0, z) = (1/2) int_{sigma z}^{z} omega_{0,2}(z_0, .) / (omega_{0,1}(sigma z) - omega_{0,1}(z)),

  where sum' leaves out omega_{0,1}. ``omega(g, n)`` is the coefficient of dz_0 ... dz_{n-1}, a
  sympy expression in ``self.zs[:n]``. In this convention the Airy curve x = z^2/2, y = z gives
  omega_{g,n} = sum <tau_d1 ... tau_dn>_g prod (2 d_i + 1)!! / z_i^(2 d_i + 2), and the semicircle
  curve x = z + 1/z, y = (z - 1/z)/2 gives omega_{g,1} = W_g(x) dx with W_g the generating
  function of genus-g gluings.
  """

  def __init__(self, x, y, sigma, ramification, z):
    self.x, self.y, self.sigma, self.ramification, self.z = x, y, sigma, list(ramification), z
    self.zs = sp.symbols("z0:12")
    self.dsigma = sp.diff(sigma, z)
    dx = sp.diff(x, z)
    self._den = sp.cancel(y.subs(z, sigma) * dx.subs(z, sigma) * self.dsigma - y * dx)

  @functools.lru_cache(maxsize=None)
  def _kernel_taylor(self, a, m):
    r"""Taylor coefficients in t = z - a of (1/2) (1/(z0 - z) - 1/(z0 - sigma z)); z0 = zs[0]."""
    z0, z = self.zs[0], self.z
    kernel = sp.Rational(1, 2) * (1 / (z0 - z) - 1 / (z0 - self.sigma))
    return _taylor(kernel.subs(z, a + t_), m)

  def _at(self, g, args):
    r"""omega_{g,len(args)} evaluated at ``args`` (each z or sigma(z)), pulled back to dz."""
    f = self.omega(g, len(args))
    val = f.subs({self.zs[i]: a for i, a in enumerate(args)}, simultaneous=True)
    return val * self.dsigma ** sum(a is self.sigma for a in args)

  @functools.lru_cache(maxsize=None)
  def omega(self, g, n):
    zs = self.zs[:n]
    if (g, n) == (0, 2):
      return 1 / (zs[0] - zs[1]) ** 2
    if 2 * g - 2 + n <= 0:
      raise ValueError("omega_{0,1} is y dx; the recursion starts at 2g - 2 + n > 0")
    z, s = self.z, self.sigma
    z0, J = zs[0], zs[1:]
    bracket = self._at(g - 1, (z, s) + J) if g >= 1 else 0
    idx = range(len(J))
    for h in range(g + 1):
      for size in range(len(J) + 1):
        for I in it.combinations(idx, size):
          if (h, size) == (0, 0) or (g - h, len(J) - size) == (0, 0):
            continue
          rest = tuple(J[i] for i in idx if i not in I)
          bracket += self._at(h, (z,) + tuple(J[i] for i in I)) * self._at(g - h, (s,) + rest)
    total = 0
    for a in self.ramification:   # Res = sum_k [t^k] kernel * [t^(-1-k)] (bracket / den)
      c = _principal_part((bracket / self._den).subs(z, a + t_))
      k_ser = self._kernel_taylor(a, len(c))
      total += sum(k_ser[k] * c[len(c) - 1 - k] for k in range(len(c)))
    # a common denominator in several variables costs far more than everything else, and the
    # terms are already simplified products of one-variable functions
    return sp.cancel(total) if n == 1 else total


def airy_curve():
  r"""x = z^2 / 2, y = z: the local model of every simple branch point (y^2 = 2x)."""
  z = sp.Symbol("z")
  return TopologicalRecursion(z**2 / 2, z, -z, [0], z)


def semicircle_curve():
  r"""x = z + 1/z, y = (z - 1/z) / 2 = V'/2 - W: the GUE spectral curve, branch points x = +-2."""
  z = sp.Symbol("z")
  return TopologicalRecursion(z + 1 / z, (z - 1 / z) / 2, 1 / z, [1, -1], z)


def intersection_numbers(tr, g, n):
  r"""<tau_d1 ... tau_dn>_g on the moduli space M_{g,n}, read off omega_{g,n} of the Airy curve."""
  zs = tr.zs[:n]
  ts = sp.symbols(f"t0:{n}")
  poly = sp.Poly(sp.expand(tr.omega(g, n).subs({zi: 1 / ti for zi, ti in zip(zs, ts)}, simultaneous=True)), *ts)
  out = {}
  for powers, coeff in poly.terms():
    ds = tuple((p - 2) // 2 for p in powers)
    if tuple(sorted(ds, reverse=True)) == ds:
      out[ds] = coeff / math.prod(math.prod(range(1, 2 * d + 2, 2)) for d in ds)
  return dict(sorted(out.items()))


def one_point_function(tr, g):
  r"""W_g(x) = omega_{g,1} / dx as a numpy function of real x > 2 (semicircle curve)."""
  z = tr.z
  f = sp.lambdify(z, sp.factor(tr.omega(g, 1).subs(tr.zs[0], z) / sp.diff(tr.x, z)), "numpy")
  return lambda x: f((x + onp.sqrt(x * x - 4)) / 2)


def gluing_counts_from_curve(tr, g, kmax=6):
  r"""The coefficients epsilon_g(k) of W_g(x) = sum_k epsilon_g(k) x^{-2k-1}, from the curve."""
  z, u = tr.z, sp.Symbol("u")
  Wg = sp.cancel(tr.omega(g, 1).subs(tr.zs[0], z) / sp.diff(tr.x, z))
  zu = (1 + sp.sqrt(1 - 4 * u**2)) / (2 * u)   # z(x) on sheet I, u = 1/x
  ser = sp.series(Wg.subs(z, zu), u, 0, 2 * kmax + 2).removeO()
  return [sp.nsimplify(ser.coeff(u, 2 * k + 1)) for k in range(kmax + 1)]




# --------------------------------------------------------------------------- #
# 6. Finite N: log-derivatives and finite free probability
# --------------------------------------------------------------------------- #

def elementary_symmetric(roots):
  r"""e_0, ..., e_n (exact for ints / Fractions), so that p(x) = sum_k (-1)^k e_k x^(n-k)."""
  e = [Fraction(1)] + [Fraction(0)] * len(roots)
  for r in roots:
    for k in range(len(e) - 1, 0, -1):
      e[k] += e[k - 1] * r
  return e


def finite_free_sum(a, b):
  r"""Marcus–Spielman–Srivastava finite free convolution, in the e_k basis:
  c_k = sum_{i+j=k} (n-i)! (n-j)! / (n! (n-k)!) a_i b_j. It equals E_Q det(x - A - Q B Q^T) for
  Haar orthogonal (or unitary) Q, and preserves real-rootedness."""
  n = len(a) - 1
  return [sum(Fraction(math.factorial(n - i) * math.factorial(n - k + i),
                       math.factorial(n) * math.factorial(n - k)) * a[i] * b[k - i] for i in range(k + 1))
          for k in range(n + 1)]


def rotated_char_polys(spectrum, count, rng=None):
  r"""Coefficients (high -> low) of det(x - A - Q A Q^T) for ``count`` Haar orthogonal Q, A = diag(spectrum)."""
  rng = onp.random.default_rng(rng)
  A = onp.diag(onp.asarray(spectrum, float))
  return onp.array([onp.poly(A + Q @ A @ Q.T) for Q in (haar_orthogonal(len(A), rng) for _ in range(count))])


def bernoulli_square_roots(m, dps=None):
  r"""Roots of (x^2 - 1)^{m/2} [+]_m (x^2 - 1)^{m/2}: they fill out the arcsine law as m grows."""
  e = elementary_symmetric([1] * (m // 2) + [-1] * (m // 2))
  return polynomial_roots(finite_free_sum(e, e), dps=max(60, 3 * m) if dps is None else dps)


def random_rotation_spectrum(m, rng=None):
  r"""Eigenvalues of one sample A + Q A Q^T, A = diag(1, ..., 1, -1, ..., -1) of size ``m``, Q Haar."""
  Q = haar_orthogonal(m, rng)
  A = onp.diag(onp.where(onp.arange(m) < m // 2, 1.0, -1.0))
  return onp.linalg.eigvalsh(A + Q @ A @ Q.T)


def rank_one_update_roots(eigs, m, *, scale=0.55, rng=None):
  r"""Roots of det(x - A - v_i v_i^T) for ``m`` Gaussian vectors v_i, A = diag(eigs): one list of
  roots per update, each interlacing ``eigs``."""
  rng = onp.random.default_rng(rng)
  eigs = onp.sort(onp.asarray(eigs, float))
  A = onp.diag(eigs)
  vs = rng.standard_normal((m, len(eigs))) * scale
  return [onp.linalg.eigvalsh(A + onp.outer(v, v)) for v in vs]


def polynomial_roots(e, dps=60):
  r"""Roots of sum_k (-1)^k e_k x^(n-k) in extended precision (monomial coefficients are badly
  conditioned beyond n ~ 20)."""
  import mpmath
  with mpmath.workdps(dps):
    coeffs = [mpmath.mpf(c.numerator) / c.denominator * (-1) ** k for k, c in enumerate(map(Fraction, e))]
    roots = mpmath.polyroots(coeffs, maxsteps=500, extraprec=4 * dps)
    return onp.array([complex(r) for r in roots])






