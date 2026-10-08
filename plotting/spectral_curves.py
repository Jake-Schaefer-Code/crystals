# plotting/spectral_curves.py
r"""Figures for ``physics.spectral_curves``: the Stieltjes transform as a branch of a curve.

One function per figure, in the order of the sections of ``physics.spectral_curves``. The numerics
(random matrices, curves, resultants, topological recursion) are computed there; this module draws
them, and ``physics.spectral_curves`` does not import it.

Figure colors carry meaning: blue = real points of a curve (off the support), orange = imaginary
points (the cuts, where the eigenvalues live), aqua / violet tints = sheet I / sheet II.
"""
from __future__ import annotations

import collections
import math

import numpy as onp
import matplotlib.pyplot as plt
import sympy as sp
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize, hsv_to_rgb, to_rgb
from scipy import special

from physics.spectral_curves import (
  G, QuarticModel, Z, abel_infinity, abel_map,
  airy_asymptotics, airy_kernel_density, bernoulli_poly, bernoulli_r, bernoulli_square_roots,
  branch_points, critical_points, density_from_curve, edge_density, elementary_symmetric,
  finite_free_sum, free_sum, gluing_genus, goe, goe_plus_atoms,
  gue_moments, harer_zagier, joukowski, pairings, r_transform_branch,
  random_rotation_spectrum, rank_one_update_roots, real_branch, rotated_char_polys, scaled_top_eigenvalue,
  semicircle_density, semicircle_poly, semicircle_s, spectral_curve, stieltjes,
  tracy_widom,
)


# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #



BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
  "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE_RAMP = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
REAL, IMAG = BLUE, ORANGE          # real points of a curve / imaginary points (the cuts)
SHEET_I, SHEET_II = AQUA, VIOLET   # physical sheet / second sheet


def tint(color, amount):
  r"""Blend ``color`` toward the chart surface: amount 0 is the color, 1 the surface."""
  c, s = onp.array(to_rgb(color)), onp.array(to_rgb(SURFACE))
  return tuple((1 - amount) * c + amount * s)


def ramp(n, colors=BLUE_RAMP[2:]):
  r"""``n`` ordinal colors, light to dark, for an ordered parameter (N, eta, a, ...)."""
  cmap = LinearSegmentedColormap.from_list("ramp", colors)
  return [cmap(v) for v in onp.linspace(0, 1, n)]


def use_style():
  r"""Recessive axes, thin marks and the categorical slot order for every figure below."""
  plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK_2,
    "axes.titlecolor": INK, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.prop_cycle": plt.cycler(color=[BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]),
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.8, "lines.markersize": 5,
    "legend.frameon": False, "legend.fontsize": 9, "font.size": 10, "text.color": INK,
    "image.origin": "lower", "savefig.dpi": 150,
  })


# --------------------------------------------------------------------------- #
# Complex-plane pictures
# --------------------------------------------------------------------------- #



def complex_grid(xlim, ylim, nx=600):
  r"""Grid ``x + i y`` of shape (ny, nx), rows bottom to top; an even ny keeps y = 0 off the grid."""
  ny = max(2, 2 * round(nx * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0]) / 2))
  return onp.linspace(*xlim, nx)[None, :] + 1j * onp.linspace(*ylim, ny)[:, None]


def phase_portrait(w, *, saturation=0.6):
  r"""RGB image of complex values: hue = arg w (red on the positive reals) and a brightness step
  at every doubling of |w|, so zeros and poles are the points where all hues meet."""
  w = onp.asarray(w, complex)
  hue = (onp.angle(w) / (2 * onp.pi)) % 1.0
  with onp.errstate(divide="ignore", invalid="ignore"):
    level = onp.log2(onp.abs(w))
  steps = onp.where(onp.isfinite(level), level - onp.floor(level), 0.0)
  value = 0.74 + 0.26 * steps
  return hsv_to_rgb(onp.stack([hue, onp.full_like(hue, saturation), value], axis=-1))


def show_complex(ax, zz, w, *, saturation=0.6):
  r"""Phase portrait of ``w`` on the grid ``zz`` from :func:`complex_grid`."""
  extent = (zz.real.min(), zz.real.max(), zz.imag.min(), zz.imag.max())
  ax.imshow(phase_portrait(w, saturation=saturation), extent=extent, origin="lower",
            interpolation="bilinear", aspect="equal")


def phase_colorbar(fig, axs, *, saturation=0.6, **kw):
  r"""Colorbar for :func:`phase_portrait`: hue against arg in (-pi, pi]."""
  t = onp.linspace(0, 1, 256)
  hsv = onp.stack([(t - 0.5) % 1.0, onp.full_like(t, saturation), onp.ones_like(t)], axis=-1)
  sm = ScalarMappable(Normalize(-onp.pi, onp.pi), ListedColormap(hsv_to_rgb(hsv)))
  cb = fig.colorbar(sm, ax=axs, ticks=[-onp.pi, 0, onp.pi], **kw)
  cb.set_ticklabels([r"$-\pi$", "0", r"$\pi$"])
  cb.outline.set_visible(False)
  return cb


def _cosine_grid(lo, hi, n=400):
  r"""Points on [lo, hi] clustered at the ends, where curves meet branch points vertically."""
  return 0.5 * (lo + hi) - 0.5 * (hi - lo) * onp.cos(onp.linspace(0, onp.pi, n))


def plot_joukowski(*, s0=0.6 * onp.exp(0.3j * onp.pi), r0=0.6):
  r"""z = -(s + 1/s) folds the s-plane 2:1 onto the z-plane: s and 1/s land on the same z.

  |s| < 1 (sheet I, the Stieltjes transform) and |s| > 1 (sheet II) each cover the z-plane once.
  The circles |s| = r and |s| = 1/r map onto one ellipse with foci -+2, the unit circle is
  squashed onto the cut [-2, 2], and the critical points s = +-1 of the map become the branch
  points z = -+2.
  """
  fig, (ax_s, ax_z) = plt.subplots(1, 2, figsize=(11.5, 4.9), layout="constrained")
  th = onp.linspace(0, 2 * onp.pi, 721)
  radii = onp.array([0.3, 0.42, 0.55, 0.7, 0.85])
  ax_s.set_facecolor(tint(SHEET_II, 0.86))
  ax_s.add_patch(plt.Circle((0, 0), 1, color=tint(SHEET_I, 0.8), lw=0, zorder=0))
  for r in onp.concatenate([radii, 1 / radii]):
    ax_s.plot(r * onp.cos(th), r * onp.sin(th), color=MUTED, lw=0.6)
  for r in radii:
    zc = joukowski(r * onp.exp(1j * th))
    ax_z.plot(zc.real, zc.imag, color=MUTED, lw=0.6)
  rr = onp.geomspace(radii[0], 1 / radii[0], 400)
  for phi in onp.arange(12) * onp.pi / 6:
    s = rr * onp.exp(1j * phi)
    ax_s.plot(s.real, s.imag, color=MUTED, lw=0.6)
    zc = joukowski(s[rr <= 1])
    ax_z.plot(zc.real, zc.imag, color=MUTED, lw=0.6)
  for r in (r0, 1 / r0):
    ax_s.plot(r * onp.cos(th), r * onp.sin(th), color=INK, lw=1.4)
  zc = joukowski(r0 * onp.exp(1j * th))
  ax_z.plot(zc.real, zc.imag, color=INK, lw=1.4)
  ax_s.plot(onp.cos(th), onp.sin(th), color=IMAG, lw=2.6)
  ax_z.plot([-2, 2], [0, 0], color=IMAG, lw=3.2, solid_capstyle="butt")

  z0 = joukowski(s0)
  ax_s.plot(s0.real, s0.imag, "o", ms=8, mfc=INK, mec=SURFACE, mew=1.5)
  ax_s.plot((1 / s0).real, (1 / s0).imag, "s", ms=7.5, mfc=INK, mec=SURFACE, mew=1.5)
  ax_s.annotate(r"$s_0$", (s0.real, s0.imag), xytext=(7, 5), textcoords="offset points")
  ax_s.annotate(r"$1/s_0$", ((1 / s0).real, (1 / s0).imag), xytext=(7, -12), textcoords="offset points")
  ax_z.plot(z0.real, z0.imag, "D", ms=7, mfc=INK, mec=SURFACE, mew=1.5)
  ax_z.annotate(r"$z_0$, with preimages $s_0$ and $1/s_0$", (z0.real, z0.imag), xytext=(-1.2, 2.45),
                arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8),
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
  for sc, zb in ((1, -2), (-1, 2)):
    ax_s.plot(sc, 0, "o", ms=6, color=INK)
    ax_z.plot(zb, 0, "o", ms=6, color=INK)
  ax_s.annotate(r"$s = \pm 1$: $dz/ds = 0$", (1, 0), xytext=(1.12, -0.35))
  ax_z.annotate(r"branch points $z = \mp 2$", (2, 0), xytext=(1.2, -0.55))
  ax_s.text(0, 0.2, "sheet I\n$|s| < 1$", ha="center", va="center", color=INK_2,
            bbox=dict(facecolor=tint(SHEET_I, 0.8), edgecolor="none", pad=1))
  ax_s.text(-2.35, 2.55, "sheet II\n$|s| > 1$", ha="center", va="center", color=INK_2)
  ax_z.text(0, 0.28, "cut = image of $|s| = 1$", ha="center", color=INK_2)
  ax_s.set(xlim=(-3.3, 3.3), ylim=(-3.3, 3.3), aspect="equal", xlabel=r"$\mathrm{Re}\,s$",
           ylabel=r"$\mathrm{Im}\,s$", title=r"$s$-plane: the curve itself (a sphere)")
  ax_z.set(xlim=(-4.1, 4.1), ylim=(-3.3, 3.3), aspect="equal", xlabel=r"$\mathrm{Re}\,z$",
           ylabel=r"$\mathrm{Im}\,z$", title=r"$z$-plane: $z = -(s + 1/s)$, two-to-one")
  return fig


def semicircle_sheets_3d(*, r_min=0.4, collar=1.6, n_r=48, n_phi=181):
  r"""Interactive two-sheeted surface of s^2 + z s + 1 = 0: height Im s over the z-plane.

  It is drawn through the uniformizing coordinate s = r e^{i phi}, z = -(s + 1/s), so there are
  no branch-cut artifacts: r < 1 is sheet I, r > 1 is sheet II. The sheets are glued along the
  image of |s| = 1, which stands over the cut as the curve Im s = +-sqrt(4 - x^2)/2 = +-pi rho(x).
  Over |x| > 2 both sheets have Im s = 0, so they seem to cross there; they are separated in
  Re s, which this picture does not show. Sheet II is drawn only for 1 < |s| < ``collar``, the
  part next to the seam; it is a steep plane (s ~ -z) further out. Drag to rotate.
  """
  import plotly.graph_objects as go
  phi = onp.linspace(0, 2 * onp.pi, n_phi)
  sheets = [(onp.geomspace(r_min, 1, n_r), SHEET_I, 0.95, "sheet I, |s| < 1 (the Stieltjes transform)"),
            (onp.geomspace(1, collar, n_r // 2), SHEET_II, 0.75, f"sheet II, 1 < |s| < {collar:g}")]
  traces = []
  for r, color, opacity, name in sheets:
    S = r[:, None] * onp.exp(1j * phi[None, :])
    Zs = joukowski(S)
    traces.append(go.Surface(x=Zs.real, y=Zs.imag, z=S.imag, surfacecolor=onp.zeros(S.shape),
                             colorscale=[[0, color], [1, color]], showscale=False, opacity=opacity,
                             name=name, showlegend=True, hoverinfo="skip",
                             lighting=dict(ambient=0.55, diffuse=0.7, specular=0.1)))
  for r in (0.5, 0.65, 0.82, 1.25):
    s = r * onp.exp(1j * phi)
    zc = joukowski(s)
    traces.append(go.Scatter3d(x=zc.real, y=zc.imag, z=s.imag, mode="lines", hoverinfo="skip",
                               line=dict(color=INK_2, width=2), showlegend=False))
  th = onp.linspace(0, 2 * onp.pi, 361)
  traces.append(go.Scatter3d(x=-2 * onp.cos(th), y=0 * th, z=onp.sin(th), mode="lines",
                             line=dict(color=IMAG, width=8),
                             name="seam over the cut: Im s = ±π ρ(x)"))
  traces.append(go.Scatter3d(x=[-2, 2], y=[0, 0], z=[0, 0], mode="markers",
                             marker=dict(size=5, color=INK), name="branch points z = ±2"))
  fig = go.Figure(traces)
  fig.update_layout(
    height=560, margin=dict(l=0, r=0, t=40, b=0), paper_bgcolor=SURFACE,
    title=dict(text="Two sheets of s² + zs + 1 = 0 glued along the cut [−2, 2]", x=0.02),
    legend=dict(x=0.01, y=0.95, bgcolor="rgba(0,0,0,0)"),
    scene=dict(xaxis_title="Re z", yaxis_title="Im z", zaxis_title="Im s",
               aspectmode="manual", aspectratio=dict(x=1, y=0.8, z=0.7),
               camera=dict(eye=dict(x=0.9, y=-1.8, z=0.5))))
  return fig


def plot_poles_to_cut(ns=(8, 40, 400), *, rng=None, xlim=(-3, 3), ylim=(-1.4, 1.4)):
  r"""Phase portraits of S_N(z) = tr (M - z)^{-1} / N for GOE matrices next to s(z).

  S_N has a pole at every eigenvalue and a zero between each pair (the critical points of the
  characteristic polynomial). As N grows the poles fill [-2, 2] densely and, away from the real
  axis, S_N converges to s(z): the branch cut is a condensate of poles.
  """
  rng = onp.random.default_rng(rng)
  zz = complex_grid(xlim, ylim, 520)
  fig, axs = plt.subplots(1, len(ns) + 1, figsize=(3.6 * (len(ns) + 1) + 0.8, 2.5),
                          layout="constrained", sharey=True)
  for ax, n in zip(axs, ns):
    eigs = onp.linalg.eigvalsh(goe(n, rng))
    show_complex(ax, zz, stieltjes(eigs, zz))
    if n <= 40:
      cp = critical_points(eigs)
      ax.plot(eigs, 0 * eigs, "o", ms=3.5, mfc=INK, mec=SURFACE, mew=0.6, label="poles: eigenvalues")
      ax.plot(cp, 0 * cp, "o", ms=3.5, mfc=SURFACE, mec=INK, mew=0.8, label=r"zeros: roots of $p'$")
    ax.set_title(rf"$S_N(z)$, $N = {n}$")
  axs[0].legend(loc="upper left", fontsize=7.5, handletextpad=0.2, borderaxespad=0.2)
  show_complex(axs[-1], zz, semicircle_s(zz))
  axs[-1].plot([-2, 2], [0, 0], color=IMAG, lw=2.2, solid_capstyle="butt")
  axs[-1].set_title(r"$s(z)$, $N \to \infty$: a cut")
  for ax in axs:
    ax.set_xlabel(r"$\mathrm{Re}\,z$")
  axs[0].set_ylabel(r"$\mathrm{Im}\,z$")
  phase_colorbar(fig, axs, label=r"$\arg$", shrink=0.85, pad=0.01)
  return fig


def plot_density_jump(n=1000, etas=(0.3, 0.03, 0.003), *, rng=None):
  r"""The density is the jump of s across the cut, and finite matrices see it at resolution eta.

  Left: s(x + i0) and s(x - i0) differ by i sqrt(4 - x^2) = 2 pi i rho(x) on the cut while Re s
  is continuous. Right: Im S_N(x + i eta) / pi is rho smoothed by a Cauchy kernel of width eta;
  once eta drops below the eigenvalue spacing ~ 1/N it resolves the individual poles instead.
  """
  fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.5, 3.9), layout="constrained")
  x = onp.linspace(-3, 3, 1201)
  up, down = semicircle_s(x + 1e-13j), semicircle_s(x - 1e-13j)
  ax0.fill_between(x, down.imag, up.imag, color=tint(IMAG, 0.82), lw=0,
                   label=r"jump $s(x+i0) - s(x-i0) = 2\pi i\,\rho(x)$")
  ax0.plot(x, up.imag, color=IMAG, label=r"$\mathrm{Im}\,s(x + i0) = +\pi\rho(x)$")
  ax0.plot(x, down.imag, color=IMAG, lw=1.0, label=r"$\mathrm{Im}\,s(x - i0) = -\pi\rho(x)$")
  ax0.plot(x, up.real, color=INK_2, lw=1.2, label=r"$\mathrm{Re}\,s(x \pm i0)$: continuous")
  ax0.set(xlabel="$x$", ylim=(-1.25, 1.6), title="Crossing the cut")
  ax0.legend(loc="upper left", ncols=2, fontsize=8)

  eigs = onp.linalg.eigvalsh(goe(n, rng))
  ax1.hist(eigs, bins=90, range=(-2.5, 2.5), density=True, color=tint(INK, 0.82),
           label=f"GOE eigenvalues, $N = {n}$")
  for eta, c in sorted(zip(etas, ramp(len(etas))), key=lambda ec: ec[0]):
    ax1.plot(x, stieltjes(eigs, x + 1j * eta).imag / onp.pi, color=c, lw=0.7 if eta < 3 / n else 1.4,
             label=rf"$\mathrm{{Im}}\,S_N(x + {eta:g}\,i)/\pi$")
  ax1.plot(x, semicircle_density(x), color=IMAG, lw=1.6, label=r"$\rho(x)$")
  ax1.set(xlabel="$x$", xlim=(-2.8, 2.8), ylim=(0, 0.62),
          title="Finite $N$: the jump seen at height $\\eta$")
  ax1.legend(loc="upper left", fontsize=8)
  return fig


def plot_edge(ns=(50, 200, 800), *, k=12, size=1200, rng=None):
  r"""Why edges are square roots and why Airy appears.

  Left: y = z + 2 s solves y^2 = z^2 - 4. Its real points are a hyperbola (|z| >= 2) and its
  imaginary points a circle, z^2 + (Im y)^2 = 4, whose upper half is 2 pi rho: the semicircle is a
  circle. At z = 2 both are the parabola y^2 = 4 (z - 2): the local model at every simple branch
  point. Middle: zoomed to xi = N^{2/3} (x - 2) the GUE edge converges to Ai'^2 - xi Ai^2, which
  continues the square root sqrt(-xi)/pi past the edge. Right: Ai is the wave whose WKB phase is
  int y dx on y^2 = xi: it oscillates where y is imaginary and decays where y is real.
  """
  rng = onp.random.default_rng(rng)
  fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), layout="constrained")
  ax = axs[0]
  for side in (-1, 1):
    zr = side * onp.cosh(onp.linspace(0, 1.25, 200)) * 2
    for sgn in (-1, 1):
      ax.plot(zr, sgn * onp.sqrt(zr**2 - 4), color=REAL, lw=1.8,
              label="real points: $y^2 = z^2 - 4$" if (side, sgn) == (1, 1) else None)
  th = onp.linspace(0, 2 * onp.pi, 400)
  ax.plot(2 * onp.cos(th), 2 * onp.sin(th), color=IMAG, lw=1.8,
          label=r"imaginary points: $z^2 + (\mathrm{Im}\,y)^2 = 4$")
  ax.fill_between(_cosine_grid(-2, 2), 0, onp.sqrt(4 - _cosine_grid(-2, 2) ** 2),
                  color=tint(IMAG, 0.85), lw=0)
  ax.text(0, 0.9, r"$2\pi\rho(x)$", ha="center", color=INK_2)
  zeta = onp.linspace(-0.6, 0.6, 200)
  ax.plot(2 + zeta, 2 * onp.sqrt(onp.abs(zeta)), color=INK, lw=1.0)
  ax.plot(2 + zeta, -2 * onp.sqrt(onp.abs(zeta)), color=INK, lw=1.0,
          label=r"local model $y^2 = 4\,(z - 2)$")
  ax.plot([-2, 2], [0, 0], "o", color=INK, ms=5)
  ax.set(xlim=(-3.9, 3.9), ylim=(-3.1, 3.1), aspect="equal", xlabel="$z$",
         ylabel=r"$y$  (or $\mathrm{Im}\,y$)", title=r"The curve $y = z + 2s$, $y^2 = z^2 - 4$")
  ax.legend(loc="upper center", fontsize=7.5, ncols=2, bbox_to_anchor=(0.5, -0.14))

  ax = axs[1]
  xi = onp.linspace(-10, 3, 400)
  edges = onp.linspace(-10, 3, 66)
  for n, c in zip(ns, ramp(len(ns))):
    ax.stairs(edge_density(n, edges, size=size, k=k, rng=rng), edges, color=c, lw=1.3,
              label=f"GUE, $N = {n}$")
  ax.plot(xi, airy_kernel_density(xi), color=INK, lw=1.8, label=r"$\mathrm{Ai}'^2 - \xi\,\mathrm{Ai}^2$")
  ax.plot(xi[xi < 0], onp.sqrt(-xi[xi < 0]) / onp.pi, color=IMAG, lw=1.2,
          label=r"$\sqrt{-\xi}/\pi$ (the semicircle edge)")
  ax.set(xlabel=r"$\xi = N^{2/3}(x - 2)$", ylabel="eigenvalues per unit $\\xi$",
         title=rf"Edge zoom (top {k} eigenvalues sampled)", xlim=(-10, 3), ylim=(0, 1.15))
  ax.legend(loc="upper right", fontsize=8)

  ax = axs[2]
  xi = onp.linspace(-12, 4, 800)
  ai = special.airy(xi)[0]
  xi_neg, osc, xi_pos, decay = airy_asymptotics(xi)
  ax.plot(xi, ai, color=INK, lw=1.8, label=r"$\mathrm{Ai}(\xi)$")
  ax.plot(xi_neg, osc,
          color=IMAG, lw=1.1, label=r"$\sin(\frac{2}{3}|\xi|^{3/2} + \frac{\pi}{4})\,/\,\sqrt{\pi}|\xi|^{1/4}$")
  ax.plot(xi_pos, decay,
          color=REAL, lw=1.1, label=r"$e^{-\frac{2}{3}\xi^{3/2}}\,/\,2\sqrt{\pi}\xi^{1/4}$")
  ax.axvline(0, color=AXIS, lw=0.8)
  ax.text(-11.5, 0.62, "$y$ imaginary: oscillation\n(eigenvalues)", color=INK_2, fontsize=8.5)
  ax.text(0.6, 0.62, "$y$ real:\ndecay", color=INK_2, fontsize=8.5)
  ax.set(xlabel=r"$\xi$", ylim=(-0.6, 0.85),
         title=r"Airy: WKB phase $\int y\,d\xi = \frac{2}{3}|\xi|^{3/2}$ on $y^2 = \xi$")
  ax.legend(loc="lower right", fontsize=7.5)
  return fig


def plot_tracy_widom(n=400, size=6000, *, rng=None):
  r"""The largest eigenvalue fluctuates on the N^{-2/3} scale set by the square-root edge,
  with Tracy–Widom laws F_1 (GOE) and F_2 (GUE)."""
  rng = onp.random.default_rng(rng)
  fig, ax = plt.subplots(figsize=(6.6, 3.9), layout="constrained")
  s = onp.linspace(-6, 4, 181)
  for beta, color, name in ((1, BLUE, "GOE"), (2, ORANGE, "GUE")):
    ax.hist(scaled_top_eigenvalue(n, beta, size, rng=rng), bins=70, range=(-6, 4), density=True,
            color=tint(color, 0.7), label=rf"{name} ($\beta = {beta}$), $N = {n}$")
    ax.plot(s, tracy_widom(s, beta)[1], color=color, lw=2.0, label=rf"Tracy–Widom $F_{beta}'$")
  ax.set(xlabel=r"$N^{2/3}(\lambda_{\max} - 2)$", ylabel="density",
         title="Largest eigenvalue: fluctuations of size $N^{-2/3}$")
  ax.legend(fontsize=8.5)
  return fig


def plot_cut_transition(curves, samples, *, xlim=(-2.9, 2.9)):
  r"""V_eff is flat where eigenvalues sit (the forces balance) and the support follows the wells.

  Top: V and V_eff = V - 2 int log|x - l| rho(l) dl. Bottom: rho from the curve against log-gas
  samples. Deepening the double well pushes the density at 0 to zero (a = -2 for g = 1) and then
  the support splits: genus 0 -> 1.
  """
  n = len(curves)
  fig, axs = plt.subplots(2, n, figsize=(3.2 * n, 5.4), layout="constrained", sharex=True, sharey="row")
  x = onp.linspace(*xlim, 1201)
  for j, (curve, lam) in enumerate(zip(curves, samples)):
    ax_v, ax_r = axs[0, j], axs[1, j]
    for lo, hi in curve.cuts:
      for ax in (ax_v, ax_r):
        ax.axvspan(lo, hi, color=tint(IMAG, 0.88), lw=0)
    ax_v.plot(x, curve.model.V(x), color=MUTED, lw=1.2, label="$V$")
    ax_v.plot(x, curve.effective_potential(x), color=INK, lw=1.8, label=r"$V_{\rm eff}$")
    ax_r.hist(onp.ravel(lam), bins=90, range=xlim, density=True, color=tint(INK, 0.8),
              label="log-gas samples")
    ax_r.plot(x, curve.density(x), color=IMAG, lw=1.8, label=r"$\rho$ from the curve")
    kind = {0: "one cut, genus 0", 1: "two cuts, genus 1"}[curve.genus]
    ax_v.set_title(rf"$a = {curve.model.a:g}$: {kind}", fontsize=10)
    ax_r.set_xlabel("$x$")
  axs[0, 0].set(ylabel="potential", ylim=(-2.6, 3.2))
  axs[1, 0].set(ylabel="density")
  axs[0, 0].legend(loc="upper center", fontsize=8)
  axs[1, 0].legend(loc="upper left", fontsize=7.5)
  return fig


def plot_branch_points(a_values, *, g=1.0):
  r"""Roots of F(x) = V'^2 - 4P in the x-plane as the symmetric double well deepens.

  Simple roots (filled) are branch points; double roots (open) are nodes, the singular points of
  y^2 = F. Two complex nodes slide down the imaginary axis, meet on the support at a = -2 sqrt(g)
  and open into one real node plus two new branch points: genus 0 -> 1.
  """
  a_values = onp.asarray(a_values, float)
  cmap = LinearSegmentedColormap.from_list("a", BLUE_RAMP[1:][::-1])
  norm = Normalize(a_values.min(), a_values.max())
  fig, ax = plt.subplots(figsize=(6.2, 4.3), layout="constrained")
  ax.axhline(0, color=AXIS, lw=0.8)
  ax.axvline(0, color=AXIS, lw=0.8)
  for a in a_values:
    curve = spectral_curve(QuarticModel(g, a))
    color = cmap(norm(a))
    ax.plot(curve.edges, 0 * curve.edges, "o", ms=6, color=color)
    nodes = curve.nodes
    ax.plot(nodes.real, nodes.imag, "o", ms=7, mfc="none", mec=color, mew=1.4)
  ax.plot([], [], "o", ms=6, color=INK_2, label="branch points (simple roots)")
  ax.plot([], [], "o", ms=7, mfc="none", mec=INK_2, mew=1.4, label="nodes (double roots)")
  cb = fig.colorbar(ScalarMappable(norm, cmap), ax=ax, label="$a$", pad=0.02)
  cb.outline.set_visible(False)
  cb.ax.axhline(-2 * onp.sqrt(g), color=INK, lw=1.5)
  ax.set(xlabel=r"$\mathrm{Re}\,x$", ylabel=r"$\mathrm{Im}\,x$", aspect="equal",
         title=rf"Roots of $V'^2 - 4P$ for $V = x^4/4 + a x^2/2$ (critical $a = -2$)")
  ax.legend(loc="upper left", fontsize=8.5)
  return fig


def plot_real_section(curve, samples=None):
  r"""A two-cut curve seen in the real plane, and its cycles in the complex x-plane.

  Left: the real points of y^2 = F (blue) and the imaginary points y = +-2 pi i rho (orange). The
  orange ovals are the real shadows of the A-cycles and enclose area 4 pi eps_i. In the gap the
  blue curve has a node at the double root d, and equilibrium (B-period = 0) says its two lobes
  have equal area: a Maxwell construction. Right: y on sheet I; A-cycles around each cut give
  the filling fractions and the B-cycle runs from cut to cut on sheet I and back on sheet II.
  """
  (e1, e2), (e3, e4) = curve.cuts[:2]
  d = curve.real_node
  eps_a = onp.array([curve.a_period(i).real for i in range(2)])
  fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained",
                                 gridspec_kw={"width_ratios": [1.0, 1.15]})
  span = e4 - e1
  pieces = [(e1 - 0.2 * span, e1), (e2, e3), (e4, e4 + 0.2 * span)]
  for k, (lo, hi) in enumerate(pieces):
    x = _cosine_grid(lo, hi)
    yr = onp.sqrt(onp.clip(curve.F(x), 0, None))
    if k == 1:
      ax0.fill_between(x, -yr, yr, color=tint(REAL, 0.82), lw=0)
    for sgn in (-1, 1):
      ax0.plot(x, sgn * yr, color=REAL, lw=1.8,
               label="real points of $y^2 = F(x)$" if (k, sgn) == (0, 1) else None)
  for i, (lo, hi) in enumerate(curve.cuts):
    x = _cosine_grid(lo, hi)
    yi = onp.sqrt(onp.clip(-curve.F(x), 0, None))
    ax0.fill_between(x, -yi, yi, color=tint(IMAG, 0.8), lw=0)
    for sgn in (-1, 1):
      ax0.plot(x, sgn * yi, color=IMAG, lw=1.8,
               label=r"imaginary points $y = \pm 2\pi i\rho$" if (i, sgn) == (0, 1) else None)
    ax0.text(0.5 * (lo + hi), 0, rf"area $= 4\pi\epsilon_{i + 1}$" + "\n" + rf"$\epsilon_{i + 1} = {eps_a[i]:.3f}$",
             ha="center", va="center", fontsize=8.5)
  lobe = curve.lobe_areas()
  ax0.annotate(f"node $d$: lobes {lobe[0]:.3f} and {lobe[1]:.3f}\n(equal areas = equilibrium)", (d, 0),
               xytext=(0.5, 0.93), textcoords="axes fraction", ha="center", va="top", fontsize=8.5,
               arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8))
  ymax = 1.9 * onp.max(onp.sqrt(onp.clip(-curve.F(onp.linspace(e1, e4, 2001)), 0, None)))
  ax0.set(xlabel="$x$", ylabel=r"$y$  (or $\mathrm{Im}\,y$)", ylim=(-ymax, ymax),
          title="Real section: ovals over the cuts, a node in the gap")
  ax0.legend(loc="lower left", fontsize=8)

  pad = 0.45 * span
  zz = complex_grid((e1 - pad, e4 + pad), (-0.55 * span, 0.55 * span), 560)
  show_complex(ax1, zz, curve.y(zz))
  for i, (lo, hi) in enumerate(curve.cuts):
    ax1.plot([lo, hi], [0, 0], color=IMAG, lw=3, solid_capstyle="butt")
    circ = curve.a_cycle(i, 400)
    ax1.plot(onp.append(circ.real, circ.real[0]), onp.append(circ.imag, circ.imag[0]), color=INK, lw=1.4)
    top = circ[100]
    ax1.annotate("", xy=(top.real - 0.02, top.imag), xytext=(top.real + 0.02, top.imag),
                 arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
    ax1.text(top.real, top.imag + 0.1, f"$A_{i + 1}$", ha="center", va="bottom")
  mid, half = 0.5 * (e2 + e3), 0.5 * (e3 - e2) + 0.5 * (e2 - e1)
  th = onp.linspace(0, onp.pi, 200)
  b_up = mid - half * onp.cos(th) + 0.35 * span * 1j * onp.sin(th)
  ax1.plot(b_up.real, b_up.imag, color=INK, lw=1.4)
  ax1.plot(b_up.real, -b_up.imag, color=INK, lw=1.4, ls=(0, (4, 3)))
  ax1.text(mid, 0.35 * span + 0.08, "$B$ (sheet I)", ha="center", va="bottom")
  ax1.text(mid, -0.35 * span - 0.1, "back on sheet II", ha="center", va="top", color=INK_2)
  ax1.plot(d, 0, "o", ms=5, mfc=SURFACE, mec=INK)
  lines = [rf"$-\frac{{1}}{{4\pi i}}\oint_{{A_i}} y\,dx$ = {eps_a[0]:.4f}, {eps_a[1]:.4f}",
           rf"$\int_{{\rm cut}}\rho$ = " + ", ".join(f"{v:.4f}" for v in curve.filling_fractions()),
           rf"$\oint_B y\,dx$ = {curve.b_period():.1e}"]
  if samples is not None:
    left = onp.mean(onp.ravel(samples) < d)
    lines.insert(2, f"eigenvalue counts = {left:.4f}, {1 - left:.4f}")
  ax1.text(0.02, 0.03, "\n".join(lines), transform=ax1.transAxes, fontsize=8.5, va="bottom",
           bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.85))
  ax1.set(xlabel=r"$\mathrm{Re}\,x$", ylabel=r"$\mathrm{Im}\,x$", title="Phase of $y$ on sheet I, with cycles")
  return fig


def _surface_3d(ax, pts, colors):
  ax.plot_surface(*pts, facecolors=colors, rstride=1, cstride=1, linewidth=0, antialiased=False,
                  shade=True, zorder=1)


def _line_3d(ax, pts, normals, **kw):
  r"""Draw a curve on a surface, hiding the points whose normal faces away from the viewer."""
  el, az = onp.radians(ax.elev), onp.radians(ax.azim)
  view = onp.array([onp.cos(el) * onp.cos(az), onp.cos(el) * onp.sin(az), onp.sin(el)])
  pts = onp.array(pts, float)
  pts[:, onp.einsum("i...,i->...", normals, view) < 0] = onp.nan
  ax.plot(*pts, zorder=kw.pop("zorder", 3), **kw)


def _torus(theta, phi, R0=1.0, r0=0.42):
  pts = onp.array([(R0 + r0 * onp.cos(phi)) * onp.cos(theta), (R0 + r0 * onp.cos(phi)) * onp.sin(theta),
                   r0 * onp.sin(phi)])
  normals = onp.array([onp.cos(phi) * onp.cos(theta), onp.cos(phi) * onp.sin(theta), onp.sin(phi)])
  return pts, normals


def _sphere(s):
  r"""Stereographic image of s on the unit sphere: |s| < 1 -> southern hemisphere."""
  s = onp.asarray(s, complex)
  pts = onp.array([2 * s.real, 2 * s.imag, onp.abs(s) ** 2 - 1]) / (onp.abs(s) ** 2 + 1)
  return pts, pts


def plot_uniformization(curve, *, levels=5):
  r"""Sheet I is a disk for one cut and a cylinder (an annulus) for two; the curve is its double.

  Top: the Joukowski coordinate s maps the z-plane minus [-2, 2] onto |s| < 1; gluing two disks
  along their boundary circle (the cut) gives a sphere, genus 0. Bottom: the Abel map
  u = int dx / ytilde sends the x-plane minus two cuts onto the rectangle [-K_B, 0] x [-K_A, K_A]
  with top and bottom glued (both are the gap): a cylinder, whose boundary circles are the cuts
  (exp(pi u / K_A) makes it an annulus). Sheet II is the mirror rectangle, and gluing the two
  cylinders along both circles gives the period rectangle of a torus, genus 1. Grey lines are
  the same coordinate lines in every panel of a row: in the second row the vertical ones
  (closed curves around a cut) are A-cycles, meridians of the torus, and the horizontal ones
  run from cut to cut: halves of B-cycles, longitudes. Blue: the real axis off the cuts (the
  real ovals of the curve); orange: the cuts.
  """
  fig = plt.figure(figsize=(14, 9.2), layout="constrained")
  axs = [[fig.add_subplot(2, 3, 3 * row + col + 1, **(dict(projection="3d", computed_zorder=False) if col == 2 else {}))
          for col in range(3)] for row in range(2)]
  th = onp.linspace(0, 2 * onp.pi, 721)
  radii = onp.linspace(0.25, 0.85, levels)
  angles = onp.arange(12) * onp.pi / 6

  # one cut: z-plane minus [-2, 2] -> |s| < 1 -> sphere
  ax = axs[0][0]
  ax.set_facecolor(tint(SHEET_I, 0.85))
  for r in radii:
    zc = joukowski(r * onp.exp(1j * th))
    ax.plot(zc.real, zc.imag, color=MUTED, lw=0.7)
  rr = onp.linspace(radii[0], 1, 200)
  for phi in angles:
    zc = joukowski(rr * onp.exp(1j * phi))
    ax.plot(zc.real, zc.imag, color=MUTED, lw=0.7)
  ax.plot([-4.5, -2], [0, 0], [2, 4.5], [0, 0], color=REAL, lw=2.2)
  ax.plot([-2, 2], [0, 0], color=IMAG, lw=3.2, solid_capstyle="butt")
  ax.set(xlim=(-4.5, 4.5), ylim=(-3.4, 3.4), aspect="equal", xlabel=r"$\mathrm{Re}\,z$",
         ylabel=r"$\mathrm{Im}\,z$", title="one cut: sheet I = $z$-plane minus $[-2, 2]$")
  ax = axs[0][1]
  ax.add_patch(plt.Circle((0, 0), 1, color=tint(SHEET_I, 0.85), lw=0, zorder=0))
  for r in radii:
    ax.plot(r * onp.cos(th), r * onp.sin(th), color=MUTED, lw=0.7)
  for phi in angles:
    ax.plot(rr * onp.cos(phi), rr * onp.sin(phi), color=MUTED, lw=0.7)
  ax.plot([-1, 1], [0, 0], color=REAL, lw=2.2)
  ax.plot(onp.cos(th), onp.sin(th), color=IMAG, lw=3)
  ax.plot(0, 0, "o", ms=5, color=INK)
  ax.annotate(r"$z = \infty$", (0, 0), xytext=(4, 6), textcoords="offset points", fontsize=9)
  ax.set(xlim=(-1.25, 1.25), ylim=(-1.25, 1.25), aspect="equal", xlabel=r"$\mathrm{Re}\,s$",
         ylabel=r"$\mathrm{Im}\,s$", title=r"$s(z)$: a disk, boundary circle = the cut")
  ax = axs[0][2]
  ax.view_init(elev=24, azim=-60)
  S = onp.geomspace(0.02, 50, 90)[:, None] * onp.exp(1j * onp.linspace(0, 2 * onp.pi, 121))[None, :]
  pts, _ = _sphere(S)
  colors = onp.where((onp.abs(S) < 1)[..., None], onp.array(to_rgb(SHEET_I) + (1,)),
                     onp.array(to_rgb(SHEET_II) + (1,)))
  _surface_3d(ax, pts, onp.array([[tint(c[:3], 0.45) + (1,) for c in row] for row in colors]))
  for r in radii:
    _line_3d(ax, *_sphere(r * onp.exp(1j * th)), color=INK_2, lw=0.6)
  for phi in angles:
    _line_3d(ax, *_sphere(rr * onp.exp(1j * phi)), color=INK_2, lw=0.6)
  _line_3d(ax, *_sphere(onp.exp(1j * th)), color=IMAG, lw=2.6)
  real = onp.concatenate([-onp.geomspace(1e3, 1e-3, 300), onp.geomspace(1e-3, 1e3, 300)])
  _line_3d(ax, *_sphere(real), color=REAL, lw=2.0)
  ax.set_title("double of the disk: a sphere (genus 0)")

  # two cuts: x-plane minus two cuts -> annulus -> torus
  X, H, u, K_A, K_B = abel_map(curve)
  e = curve.edges
  re_levels = -K_B * onp.arange(5, 0, -1) / 6
  im_levels = K_A * onp.arange(1, 6) / 6
  ax = axs[1][0]
  ax.set_facecolor(tint(SHEET_I, 0.85))
  for Y, U in ((H, u), (-H[::-1], onp.conj(u[::-1]))):   # u(conj x) = conj u(x)
    ax.contour(X, Y, U.real, levels=re_levels, colors=[MUTED], linewidths=0.7, linestyles="solid")
    ax.contour(X, Y, U.imag, levels=onp.sort(onp.sign(Y[0]) * im_levels), colors=[MUTED], linewidths=0.7,
               linestyles="solid")
  ax.plot([X[0], e[0]], [0, 0], [e[1], e[2]], [0, 0], [e[3], X[-1]], [0, 0], color=REAL, lw=2.2)
  for lo, hi in curve.cuts:
    ax.plot([lo, hi], [0, 0], color=IMAG, lw=3.2, solid_capstyle="butt")
  ax.set(xlim=(X[0], X[-1]), ylim=(-0.75 * X[-1], 0.75 * X[-1]), aspect="equal", xlabel=r"$\mathrm{Re}\,x$",
         ylabel=r"$\mathrm{Im}\,x$", title="two cuts: sheet I = $x$-plane minus the support")
  ax = axs[1][1]
  u_inf = abel_infinity(curve)
  ax.add_patch(plt.Rectangle((-K_B, -K_A), K_B, 2 * K_A, color=tint(SHEET_I, 0.85), lw=0, zorder=0))
  ax.add_patch(plt.Rectangle((0, -K_A), K_B, 2 * K_A, color=tint(SHEET_II, 0.85), lw=0, zorder=0))
  for c in re_levels:
    ax.plot([c, c], [-K_A, K_A], [-c, -c], [-K_A, K_A], color=MUTED, lw=0.7)
  for c in onp.concatenate([im_levels, -im_levels]):
    ax.plot([-K_B, K_B], [c, c], color=MUTED, lw=0.7)
  for c in (-K_A, 0.0, K_A):
    ax.plot([-K_B, K_B], [c, c], color=REAL, lw=2.2)
  for c in (-K_B, 0.0, K_B):
    ax.plot([c, c], [-K_A, K_A], color=IMAG, lw=3, solid_capstyle="butt")
  for c in (-0.5 * K_B, 0.5 * K_B):
    for yy in (-K_A, K_A):
      ax.annotate("", xy=(c + 0.06 * K_B, yy), xytext=(c - 0.06 * K_B, yy),
                  arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
  ax.plot([u_inf, -u_inf], [0, 0], "o", ms=5, color=INK)
  ax.annotate(r"$x = \infty$", (u_inf, 0), xytext=(4, 5), textcoords="offset points", fontsize=9)
  for c, label in ((-K_B, "cut 2"), (0.0, "cut 1"), (K_B, "cut 2")):
    ax.text(c, K_A * 1.04, label, ha="center", va="bottom", fontsize=9, color=INK_2)
  ax.text(-0.5 * K_B, -K_A * 1.18, "sheet I", ha="center", va="top", fontsize=9.5)
  ax.text(0.5 * K_B, -K_A * 1.18, "sheet II", ha="center", va="top", fontsize=9.5)
  ax.text(-0.5 * K_B, K_A * 0.9, "gap (top = bottom)", ha="center", va="top", fontsize=8.5, color=INK_2)
  ax.set(xlim=(-1.08 * K_B, 1.08 * K_B), ylim=(-1.35 * K_A, 1.25 * K_A), aspect="equal",
         xlabel=r"$\mathrm{Re}\,u$", ylabel=r"$\mathrm{Im}\,u$",
         title=r"$u = \int dx/\tilde y$: two cylinders, one period rectangle")
  ax = axs[1][2]
  ax.view_init(elev=38, azim=-62)
  theta, phi = onp.meshgrid(onp.linspace(-onp.pi, onp.pi, 121), onp.linspace(-onp.pi, onp.pi, 61))
  pts, _ = _torus(theta, phi)
  colors = onp.where((theta <= 0)[..., None], onp.array(tint(SHEET_I, 0.45) + (1,)),
                     onp.array(tint(SHEET_II, 0.45) + (1,)))
  _surface_3d(ax, pts, colors)
  full = onp.linspace(-onp.pi, onp.pi, 400)
  half = onp.linspace(-onp.pi, 0, 200)
  for c in re_levels:
    _line_3d(ax, *_torus(onp.pi * c / K_B + 0 * full, full), color=INK_2, lw=0.6)
  for c in onp.concatenate([im_levels, -im_levels]):
    _line_3d(ax, *_torus(half, onp.pi * c / K_A + 0 * half), color=INK_2, lw=0.6)
  for ph in (0.0, onp.pi):
    _line_3d(ax, *_torus(full, ph + 0 * full), color=REAL, lw=2.0)
  for tt in (0.0, onp.pi):
    _line_3d(ax, *_torus(tt + 0 * full, full), color=IMAG, lw=2.6)
  _line_3d(ax, *_torus(onp.array([onp.pi * u_inf / K_B]), onp.array([0.0])), color=INK, marker="o", ms=5)
  ax.set_title("double of the cylinder: a torus (genus 1)")
  for ax3 in (axs[0][2], axs[1][2]):
    ax3.set_box_aspect((1, 1, 1.0 if ax3 is axs[0][2] else 0.45))
    ax3.set_axis_off()
  return fig


def plot_r_transform(a=1.4):
  r"""Inverting G is reading its curve with the axes swapped; then R-transforms simply add.

  Left: G on the real axis off the support (bold) sits on the real locus of L(G, z) = 0 (thin).
  Middle: the same curves reflected in the diagonal are the inverse functions K(w). Right:
  R(w) = K(w) - 1/w on the branch through w = 0 (z -> +-oo); the free sum's R-transform, computed
  from the resultant curve, is the pointwise sum R_semicircle + R_Bernoulli (dots).
  """
  L1, L2 = semicircle_poly(), bernoulli_poly(a)
  L3 = free_sum(L1, L2)
  curves = [(L1, "semicircle", BLUE), (L2, rf"Bernoulli $\pm{a:g}$", ORANGE), (L3, "free sum", AQUA)]
  fig, axs = plt.subplots(1, 3, figsize=(14, 4.4), layout="constrained")
  x = onp.linspace(-5, 5, 2001)
  gg, zg = onp.linspace(-2.2, 2.2, 700), onp.linspace(-5, 5, 900)
  branches = []
  for L, name, color in curves:
    vals = sp.lambdify((G, Z), L, "numpy")(gg[:, None], zg[None, :])
    g = real_branch(L, x)
    branches.append(g)
    axs[0].contour(zg, gg, vals, levels=[0], colors=[tint(color, 0.45)], linewidths=0.9)
    axs[1].contour(gg, zg, vals.T, levels=[0], colors=[tint(color, 0.45)], linewidths=0.9)
    axs[0].plot(x, g, color=color, lw=2.4, label=name)
    axs[1].plot(g, x, color=color, lw=2.4, label=name)
  axs[1].plot([-2.2, 2.2], [-2.2, 2.2], color=MUTED, lw=0.8)
  axs[0].set(xlabel="$z$ (real, off the support)", ylabel="$G(z)$", xlim=(-5, 5), ylim=(-2.2, 2.2),
             title="$G$ on the real axis: part of the curve $L(G, z) = 0$")
  axs[1].set(xlabel="$w$", ylabel="$K(w) = G^{-1}(w)$", xlim=(-2.2, 2.2), ylim=(-5, 5),
             title="swap the axes: the inverse $K$")
  for L, name, color in curves:
    for side in (1, -1):   # follow G in from z = +-oo until it stops being real (the support)
      w_b, r_b = r_transform_branch(L, side)
      axs[2].plot(w_b, r_b, color=color, lw=2.4, label=name if side == 1 else None)
  w = onp.linspace(-0.9, 0.9, 26)
  axs[2].plot(w, w + bernoulli_r(a, w), "o", ms=4.5, mfc=SURFACE, mec=INK, mew=1.0, label=r"$R_{\rm sc}(w) + R_{\rm B}(w)$")
  axs[2].set(xlabel="$w$", ylabel="$R(w) = K(w) - 1/w$", xlim=(-0.9, 0.9), ylim=(-2.2, 2.2),
             title="$R$-transforms add under free convolution")
  for ax in axs:
    ax.axhline(0, color=AXIS, lw=0.6)
    ax.axvline(0, color=AXIS, lw=0.6)
  axs[0].legend(loc="upper left", fontsize=8.5)
  axs[2].legend(loc="upper left", fontsize=8.5)
  return fig, L3


def plot_free_sums(a_values=(0.6, 1.0, 1.4), n=2000, *, rng=None):
  r"""semicircle [+] Bernoulli(+-a) from the resultant curve against eigenvalues of W + diag(+-a).

  Free convolution is what adding "randomly rotated" matrices does at large n: W (GOE) is
  rotation invariant, so W + A is in free position with A. As the atoms separate the support
  splits in two at a = 1: the free sum can have higher genus than either summand.
  """
  rng = onp.random.default_rng(rng)
  fig, axs = plt.subplots(1, len(a_values), figsize=(4.2 * len(a_values), 3.4), layout="constrained",
                          sharey=True)
  x = onp.linspace(-3.8, 3.8, 1601)
  Wm = goe(n, rng)
  for ax, a in zip(axs, a_values):
    L = free_sum(semicircle_poly(), bernoulli_poly(a))
    eigs = goe_plus_atoms(Wm, a)
    ax.hist(eigs, bins=90, range=(-3.8, 3.8), density=True, color=tint(INK, 0.8),
            label=f"eig($W + A$), $n = {n}$")
    ax.plot(x, density_from_curve(L, x), color=IMAG, lw=1.8, label="from the resultant curve")
    bp = branch_points(L)
    bp = onp.sort(bp.real[onp.abs(bp.imag) < 1e-5])
    ax.plot(bp, 0 * bp, "|", color=INK, ms=12, mew=1.5, label="real branch points")
    rho0 = density_from_curve(L, onp.array([0.0]))[0]
    kind = "one interval" if rho0 > 1e-3 else "two intervals" if a > 1 else r"critical: $\rho(0) = 0$"
    ax.set(xlabel="$x$", title=rf"$a = {a:g}$: {kind}")
  axs[0].set_ylabel("density")
  axs[0].legend(loc="upper left", fontsize=8)
  return fig


def _chord(ax, p, q, color):
  r"""Quadratic Bezier from p to q bowed toward the center."""
  t = onp.linspace(0, 1, 60)[:, None]
  c = 0.15 * (p + q)
  pts = (1 - t) ** 2 * p + 2 * (1 - t) * t * c + t**2 * q
  ax.plot(pts[:, 0], pts[:, 1], color=color, lw=1.6, solid_capstyle="round")


def plot_chord_diagrams(k=3, *, ncols=5):
  r"""All gluings of a 2k-gon, sorted by the genus of the resulting surface.

  A chord joins two sides that are glued. Non-crossing diagrams give spheres; there are
  Catalan(k) of them, the semicircle's moments. Crossings force handles.
  """
  sigmas = sorted(pairings(k), key=gluing_genus)
  nrows = -(-len(sigmas) // ncols)
  fig, axs = plt.subplots(nrows, ncols, figsize=(1.75 * ncols, 1.85 * nrows), layout="constrained")
  corners = onp.exp(1j * (onp.pi / 2 + onp.pi / (2 * k) + onp.pi * onp.arange(2 * k + 1) / k))
  mids = 0.5 * (corners[:-1] + corners[1:])
  colors = [BLUE, ORANGE, AQUA, YELLOW]
  for ax in axs.flat:
    ax.axis("off")
  for ax, sigma in zip(axs.flat, sigmas):
    g = gluing_genus(sigma)
    ax.plot(corners.real, corners.imag, color=AXIS, lw=1.0)
    for i, j in enumerate(sigma):
      if i < j:
        _chord(ax, onp.array([mids[i].real, mids[i].imag]), onp.array([mids[j].real, mids[j].imag]), colors[g])
    ax.plot(mids.real, mids.imag, "o", ms=2.5, color=INK_2)
    ax.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), aspect="equal")
    ax.set_title(f"genus {g}", fontsize=8.5, color=INK_2, pad=1)
  counts = collections.Counter(gluing_genus(s) for s in sigmas)
  fig.suptitle(f"{len(sigmas)} gluings of a {2 * k}-gon: "
               + ", ".join(rf"$\epsilon_{g}({k}) = {counts[g]}$" for g in sorted(counts)), fontsize=11)
  return fig


def plot_genus_expansion(ks=(3, 4), ns=(1, 2, 3, 5, 10), *, size=100_000, rng=None):
  r"""E tr M^{2k} / N for GUE is exactly sum_g epsilon_g(k) N^{-2g}.

  Bars: the genus-g gluings weighted by N^{-2g}; dots: Monte Carlo. At N = 1 every gluing counts
  once, giving the Gaussian moment (2k - 1)!!; as N -> oo only the planar ones survive (Catalan).
  """
  rng = onp.random.default_rng(rng)
  mc = {n: gue_moments(n, max(ks), size, rng=rng) for n in ns}
  labels = [str(n) for n in ns] + [r"$\infty$"]
  fig, axs = plt.subplots(1, len(ks), figsize=(5.6 * len(ks), 3.8), layout="constrained")
  colors = [BLUE, ORANGE, AQUA, YELLOW]
  for ax, k in zip(onp.atleast_1d(axs), ks):
    xs = onp.arange(len(labels))
    bottom = onp.zeros(len(labels))
    for g in range(k // 2 + 1):
      h = onp.array([harer_zagier(g, k) * float(n) ** (-2 * g) for n in ns] + [float(g == 0) * harer_zagier(0, k)])
      ax.bar(xs, h, bottom=bottom, width=0.62, color=colors[g], edgecolor=SURFACE, linewidth=2,
             label=rf"genus {g}: $\epsilon_{g}({k}) = {harer_zagier(g, k)}$")
      bottom += h
    mean = [mc[n][0][k - 1] for n in ns]
    err = [2 * mc[n][1][k - 1] for n in ns]
    ax.errorbar(xs[:-1], mean, yerr=err, fmt="o", ms=5, color=INK, mfc=SURFACE, mew=1.2, capsize=3,
                label=f"Monte Carlo ($\\pm 2$ s.e., {size:,} samples)")
    ax.set_xticks(xs, labels)
    ax.set(xlabel="$N$", ylabel=rf"$E\,\frac{{1}}{{N}}\mathrm{{tr}}\,M^{{{2 * k}}}$",
           title=rf"$k = {k}$: $(2k-1)!! = {math.prod(range(1, 2 * k, 2))}$ gluings in all")
    ax.legend(fontsize=8)
  return fig


def plot_genus_near_edge(W, ns=(10**2, 10**4)):
  r"""The genus expansion breaks down in the Airy window x - 2 ~ N^{-2/3}.

  N^{-2g} W_g(2 + delta) ~ N^{-2g} delta^{-(6g - 1)/2}: every term is a power of N^{-2} delta^{-3}
  times delta^{1/2}. Far from the edge higher genera are negligible; at delta ~ N^{-2/3} they are
  all comparable and must be resummed, which is what the Airy kernel does.
  ``W`` lists callables W_0 (singular part), W_1, W_2, ...
  """
  fig, axs = plt.subplots(1, len(ns), figsize=(5.4 * len(ns), 3.8), layout="constrained", sharey=True)
  delta = onp.geomspace(1e-5, 1.0, 400)
  colors = [IMAG, BLUE, AQUA, YELLOW]
  for ax, n in zip(onp.atleast_1d(axs), ns):
    for g, Wg in enumerate(W):
      ax.loglog(delta, onp.abs(Wg(2 + delta)) * float(n) ** (-2 * g), color=colors[g], lw=1.8,
                label=(r"$|\sqrt{x^2 - 4}|/2$ (singular part of $W_0$)" if g == 0
                       else rf"$N^{{-{2 * g}}}\,W_{g}(x)$"))
    ax.axvline(n ** (-2 / 3), color=INK, lw=1.0)
    ax.text(n ** (-2 / 3) * 1.25, 1e-9, r"$N^{-2/3}$", fontsize=9)
    ax.set(xlabel=r"$\delta = x - 2$", title=rf"$N = 10^{{{int(onp.log10(n))}}}$", ylim=(1e-10, 1e6))
  onp.atleast_1d(axs)[0].set_ylabel("size of the genus-$g$ term")
  onp.atleast_1d(axs)[0].legend(fontsize=8, loc="upper right")
  return fig


def plot_electrostatics(eigs=(-1.7, -1.05, -0.35, 0.3, 1.0, 1.75)):
  r"""S(z) = -(1/n) p'/p: charges at the eigenvalues, equilibrium points at the roots of p'.

  Left: p on the real line; its critical points interlace its roots (Rolle). Right: the 2D
  electric field of unit charges at the eigenvalues, E = sum 1 / conj(z - lambda_i) = -n conj S(z).
  It vanishes exactly at the zeros of p'. As n grows, -2 W(x) balancing V'(x) on the support is
  the continuum version of this force balance.
  """
  eigs = onp.sort(onp.asarray(eigs, float))
  cp = critical_points(eigs)
  fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 3.9), layout="constrained",
                                 gridspec_kw={"width_ratios": [1, 1.3]})
  x = onp.linspace(eigs[0] - 0.4, eigs[-1] + 0.4, 1601)
  p = onp.prod(x[:, None] - eigs[None, :], axis=1)
  pc = onp.prod(cp[:, None] - eigs[None, :], axis=1)
  scale = onp.max(onp.abs(pc))
  ax0.axhline(0, color=AXIS, lw=0.8)
  ax0.plot(x, p / scale, color=INK, lw=1.6, label=r"$p(x) = \det(x - A)$")
  ax0.plot(eigs, 0 * eigs, "o", ms=7, color=IMAG, label="roots = eigenvalues = poles of $S$")
  ax0.plot(cp, pc / scale, "x", ms=7, mew=2, color=INK, label="roots of $p'$ = zeros of $S$")
  ax0.set(xlabel="$x$", ylim=(-1.6, 1.6), yticks=[], title="Critical points interlace the roots")
  ax0.legend(loc="lower left", fontsize=8)
  zz = complex_grid((x[0], x[-1]), (-1.5, 1.5), 500)
  E = onp.conj(-len(eigs) * stieltjes(eigs, zz))
  ax1.streamplot(zz.real[0], zz.imag[:, 0], E.real, E.imag, color=MUTED, density=1.5, linewidth=0.7,
                 arrowsize=0.7)
  ax1.plot(eigs, 0 * eigs, "o", ms=8, color=IMAG, mec=SURFACE, mew=1)
  ax1.plot(cp, 0 * cp, "x", ms=8, mew=2.2, color=INK)
  ax1.set(xlabel=r"$\mathrm{Re}\,z$", ylabel=r"$\mathrm{Im}\,z$", aspect="equal",
          xlim=(x[0], x[-1]), ylim=(-1.5, 1.5), title=r"Field $\overline{p'/p}$ of unit charges")
  return fig


def plot_finite_free(ns=(4, 8, 16, 32), *, n_samples=40, rng=None):
  r"""Expected characteristic polynomials: p [+]_n q = E_Q det(x - A - Q B Q^T).

  Left (n = 4, A = B = diag(1, 1, -1, -1)): characteristic polynomials of random rotations
  scatter, but their average is the MSS polynomial, which is real-rooted. Right: roots of
  (x^2 - 1)^{n/2} [+]_n (x^2 - 1)^{n/2} fill out the arcsine law, the free convolution of two
  Bernoulli laws; a single random A + Q B Q^T (bottom row) is noisier.
  """
  rng = onp.random.default_rng(rng)
  fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.5, 4.2), layout="constrained")
  x = onp.linspace(-2.4, 2.4, 600)
  polys = rotated_char_polys([1.0, 1.0, -1.0, -1.0], max(n_samples, 2000), rng)
  for c in polys[:n_samples]:
    ax0.plot(x, onp.polyval(c, x), color=tint(INK, 0.72), lw=0.7)
  e = elementary_symmetric([1, 1, -1, -1])
  mss = finite_free_sum(e, e)
  mss_coeffs = [float((-1) ** k * c) for k, c in enumerate(mss)]
  ax0.plot(x, onp.polyval(mss_coeffs, x), color=BLUE, lw=2.4, label=r"$p \boxplus_4 p$ (MSS formula)")
  ax0.plot(x[::25], onp.polyval(polys.mean(0), x[::25]), "o", ms=4.5, mfc=SURFACE, mec=INK, mew=1.0,
           label=f"average of {len(polys):,} random rotations")
  rts = onp.sort(onp.roots(mss_coeffs).real)
  ax0.plot(rts, 0 * rts, "o", ms=7, color=BLUE, mec=SURFACE, mew=1.2)
  ax0.plot([], [], color=tint(INK, 0.72), lw=0.7, label=r"$\det(x - A - QAQ^T)$ for random $Q$")
  ax0.axhline(0, color=AXIS, lw=0.8)
  ax0.set(xlabel="$x$", ylim=(-3, 5), title="$n = 4$: the expected characteristic polynomial")
  ax0.legend(loc="upper center", fontsize=8)

  u = onp.linspace(-2, 2, 400)[1:-1]
  ax1.fill_between(u, 0, 1 / (onp.pi * onp.sqrt(4 - u * u)), color=tint(IMAG, 0.8), lw=0)
  ax1.plot(u, 1 / (onp.pi * onp.sqrt(4 - u * u)), color=IMAG, lw=1.4, label=r"arcsine law $= $ Bern $\boxplus$ Bern")
  rows = []
  for j, m in enumerate(ns):
    r = bernoulli_square_roots(m)
    rows.append((rf"$n = {m}$", onp.sort(r.real), onp.max(onp.abs(r.imag))))
  m = ns[-1]
  rows.append((f"one sample\n$n = {m}$", random_rotation_spectrum(m, rng), 0.0))
  for j, (label, r, _) in enumerate(rows):
    yy = -0.35 - 0.3 * j
    color = BLUE if j < len(ns) else INK_2
    ax1.plot(r, onp.full_like(r, yy), "o", ms=4, color=color)
    ax1.text(-2.55, yy, label, ha="right", va="center", fontsize=8.5)
  q = 2 * onp.sin(onp.pi * ((onp.arange(m) + 0.5) / m - 0.5))
  ax1.plot(q, onp.full_like(q, -0.35 - 0.3 * (len(ns) - 1) + 0.12), "|", ms=6, color=IMAG,
           label=f"arcsine quantiles ($n = {m}$)")
  ax1.set(xlabel="$x$", xlim=(-3.4, 2.4), ylim=(-0.35 - 0.3 * len(rows), 1.1), yticks=[],
          title=r"Roots of $(x^2-1)^{n/2} \boxplus_n (x^2-1)^{n/2}$, all real")
  ax1.legend(loc="upper right", fontsize=8)
  return fig, rows


def plot_interlacing(eigs=(-1.6, -0.6, 0.4, 1.3), *, m=6, rng=None):
  r"""An interlacing family: rank-one updates A + v v^T all interlace det(x - A).

  Every det(x - A - v_i v_i^T) has one root in each gap [lambda_j, lambda_{j+1}] and one above
  lambda_n, so any average of them is real-rooted with roots in the same gaps, and some member
  has its largest root at most the average's. Iterating this over a tree of choices is the
  Marcus–Spielman–Srivastava route to Kadison–Singer.
  """
  eigs = onp.sort(onp.asarray(eigs, float))
  n = len(eigs)
  roots = rank_one_update_roots(eigs, m, rng=rng)
  avg = onp.mean([onp.poly(r) for r in roots], axis=0)
  avg_roots = onp.sort(onp.roots(avg).real)
  fig, ax = plt.subplots(figsize=(8.5, 3.6), layout="constrained")
  for lam in eigs:
    ax.axvline(lam, color=AXIS, lw=1.0)
  for i, r in enumerate(roots):
    ax.plot(r, onp.full(n, i), "o", ms=6, color=INK_2)
    ax.plot(r[-1], i, "o", ms=9, mfc="none", mec=INK_2, mew=1.2)
  ax.plot(avg_roots, onp.full(n, m + 0.6), "D", ms=7, color=BLUE)
  best = min(r[-1] for r in roots)
  ax.axvline(avg_roots[-1], color=BLUE, lw=1.0)
  ax.text(avg_roots[-1] + 0.04, m + 0.25, f"largest root of the average: {avg_roots[-1]:.3f}\n"
          f"smallest largest root in the family: {best:.3f}", fontsize=8.5, color=INK_2, va="top")
  ax.set_yticks(list(range(m)) + [m + 0.6],
                [f"$A + v_{i + 1}v_{i + 1}^T$" for i in range(m)] + ["average polynomial"])
  ax.set(xlabel="$x$", title="Roots of rank-one updates (circled: largest); grey lines: eigenvalues of $A$")
  return fig
