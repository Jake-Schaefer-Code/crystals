"""Visualize the nested-neighborhood step in the open mapping theorem.

Dependencies: numpy, matplotlib. Optional notebook UI: ipywidgets.

Run:
  python open_mapping_visualization.py
  python open_mapping_visualization.py --steps 10 --html demo.html

The command writes an offline HTML explorer. In a notebook, call:
  demo = make_demo()
  notebook_app(demo)

Each chart is a separate Matplotlib figure. No plotting colors or styles
are explicitly set. This is a constructed R^2 example, not an inversion
algorithm: a known limit is used to generate an admissible correction path.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


# ---------- 1. Mathematical construction ----------

def _default_A():
  return [[1.4, 0.6], [-0.25, 0.9]]

def make_demo(
  steps: int = 8, 
  radius: float = 0.9, 
  angle_degrees: float = 110.0, 
  matrix=None,
  *,
  max_steps: int = 24
) -> dict:
  """Construct a valid infinite spiral, sampled through `steps + 1`.

  U_k = closed_ball(0, rho_k), rho_k = radius * 2^(-k).
  e_k = 0.6 * rho_(k+1) * v_k, with rotating unit vectors v_k.
  x_star = e_0; s_k = x_star - e_k; y = A @ x_star.

  Thus s_0 = 0 and the correction d_(k+1) = s_(k+1) - s_k obeys
      ||d_(k+1)|| <= ||e_k|| + ||e_(k+1)||
                      = 0.9 * rho_(k+1) < rho_(k+1).
  Also r_k = y - A @ s_k = A @ e_k lies in F_(k+1) = A(U_(k+1)).
  These estimates hold for every k, not just the displayed samples.
  """
  if not isinstance(steps, int) or not 1 <= steps <= max_steps:
    raise ValueError("steps must be an integer between 1 and 24.")
  if not np.isfinite(radius) or radius <= 0:
    raise ValueError("radius must be positive and finite.")
  if not np.isfinite(angle_degrees):
    raise ValueError("angle_degrees must be finite.")
  A = np.array(_default_A() if matrix is None else matrix, dtype=float)
  if A.shape != (2, 2) or not np.all(np.isfinite(A)):
    raise ValueError("matrix must be a finite 2-by-2 array.")
  
  singular_values = np.linalg.svd(A, compute_uv=False)
  if singular_values[-1] <= 1e-10 * singular_values[0]:
    raise ValueError("Use a numerically nonsingular matrix.")

  j = np.arange(steps + 3)
  rho_k = radius * 2.0 ** (-j)
  angle = np.deg2rad(30.0 + angle_degrees * j[:-1])
  e_k = 0.6 * rho_k[1:, None] * np.column_stack((np.cos(angle), np.sin(angle)))
  x_star = e_k[0].copy()
  s = x_star - e_k
  y = A @ x_star
  Ts = s @ A.T
  residuals = e_k @ A.T  # More stable than subtracting nearly equal vectors.
  corrections = np.diff(s, axis=0)
  tol = 100 * np.finfo(float).eps * max(1.0, radius)

  # Finite-sample checks supplement the analytic estimates in the docstring.
  assert np.allclose(s[0], 0.0)
  assert np.allclose(Ts + residuals, y)
  assert np.all(np.linalg.norm(corrections, axis=1) <= rho_k[1:-1] + tol)
  assert np.all(np.linalg.norm(e_k, axis=1) <= rho_k[1:] + tol)
  assert np.all(np.linalg.norm(corrections, axis=1) + rho_k[1:-1] <= rho_k[:-2] + tol)
  for n in range(steps + 1):
      assert np.all(np.linalg.norm(s[n:] - s[n], axis=1) <= rho_k[n] + tol)

  return dict(
    steps=steps, A=A, rho=rho_k, e=e_k, x_star=x_star,
    s=s, y=y, Ts=Ts, residuals=residuals,
    corrections=corrections, opnorm=singular_values[0]
  )


# ---------- 2. Drawing helpers and the four proof views ----------
VIEWS = {
  "basis": "1. Neighborhood basis at zero",
  "nested": "2. Nested neighborhoods around partial sums",
  "images": "3. Nested images retaining the target",
  "residuals": "4. Residuals in shrinking image neighborhoods",
}


def _boundary(center, radius, A=None):
  theta = np.linspace(0, 2 * np.pi, 361)
  p = radius * np.column_stack((np.cos(theta), np.sin(theta)))
  if A is not None:
    p = p @ np.asarray(A).T
  return np.asarray(center) + p


def _curve(ax: Axes, points, **kwargs):
  return ax.plot(points[:, 0], points[:, 1], **kwargs)[0]

def _point(ax: Axes, point, **kwargs):
  return ax.plot(*point, linestyle="None", **kwargs)[0]

def _new_figure(title):
  fig = plt.figure(figsize=(8.2, 7.4))
  ax = fig.add_axes((0.13, 0.31, 0.76, 0.57), axes_class=Axes)
  ax.set_aspect("equal", adjustable="box")
  ax.set(xlabel="First coordinate", ylabel="Second coordinate")
  ax.grid(True, alpha=0.18)
  fig.suptitle(title, fontsize=14, y=0.96)
  return fig, ax


def _limits(ax: Axes, center, half_width):
  center = np.asarray(center)
  ax.set(
    xlim=(center[0] - half_width, center[0] + half_width),
    ylim=(center[1] - half_width, center[1] + half_width),
  )

def _finish(fig, ax: Axes, text):
  handles, labels = ax.get_legend_handles_labels()
  fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.245),
              ncol=3, frameon=False, fontsize=9)
  fig.text(0.06, 0.025, text, fontsize=10, va="bottom", linespacing=1.7)
  return fig

def _w_radius(eps, theta):
  return eps * (1.45 + 0.25 * np.cos(3 * theta) + 0.15 * np.sin(5 * theta))

def plot_view(
  demo: dict, 
  view: str = "nested", 
  k: int = 2,
  epsilon: float = 0.25, 
  zoom: bool = False
):
  """Return one Matplotlib figure; epsilon affects the basis view only.

  Dashed outlines denote either the next ball/image or the test
  neighborhood, as labeled. The unfilled outlines represent whole
  closed balls or ellipses, not just their boundaries.
  """
  nsteps = demo['steps']
  if view not in VIEWS:
    raise ValueError(f"view must be one of {tuple(VIEWS)}")
  if not isinstance(k, (int, np.integer)) or not 0 <= k <= nsteps:
    raise ValueError(f"k must be between 0 and {nsteps}.")
  if not np.isfinite(epsilon) or epsilon <= 0:
    raise ValueError("epsilon must be positive and finite.")
  A, rho, s = demo["A"], demo["rho"], demo["s"]
  x_star, y, Ts = demo["x_star"], demo["y"], demo["Ts"]
  residuals, opnorm = demo["residuals"], demo["opnorm"]
  fig, ax = _new_figure(VIEWS[view] + f"  |  k = {k}")
  zero = np.zeros(2)

  cmap = plt.get_cmap('Greens')
  cmap2 = plt.get_cmap('Reds')

  def _plot_boundary(ax, points, j, label):
    kwargs = dict(
      linewidth = 2.5 if j == k else 1.2,
      linestyle = "--" if j == k + 1 else "-",
      c = (cmap(j/nsteps) if j>= k else cmap2(j/nsteps)),
      alpha = 1.0 if j >= k else 0.3,
    )
    return _curve(ax, points, label=label, **kwargs)

  if view == "basis":
    for j in range(k + 2):
      label = (rf"$U_{j}$" if j in {k, k + 1} else "_nolegend_")
      _plot_boundary(ax, _boundary(zero, rho[j]), j, label)

    theta = np.linspace(0, 2 * np.pi, 601)
    # W is the OPEN interior of this radial boundary. Its radius is
    # >= 1.05 * epsilon, so B(0, epsilon) is certainly contained in W.
    w = _w_radius(epsilon, theta)[:, None] * np.column_stack((np.cos(theta), np.sin(theta)))
    _curve(ax, w, linestyle="--", linewidth=1.8, c='slategray', label=r"Boundary of $W$")
    _curve(ax, _boundary(zero, epsilon), linestyle=":", linewidth=1.5, c='mediumpurple',
            label=r"Boundary of $B(0,\epsilon)$")
    _point(ax, zero, marker="+", markersize=10)
    # Some finite tails illustrate the uniform-in-m Cauchy estimate.
    tails = s[k + 1:] - s[k]
    _curve(ax, tails, marker="o", markersize=3, linewidth=0.9, c='black',
            label=r"Tails $s_m-s_k$, $m>k$")
    _limits(ax, zero, 1.22 * rho[k] if zoom
            else 1.12 * max(rho[0], 1.85 * epsilon))
    status = (
      rf"$\rho_{k}<\epsilon$: $U_{k}\subset B(0,\epsilon)\subset W$."
      if rho[k] < epsilon else
      r"The chosen $\epsilon$ does not yet certify $U_k\subset W$."
    )
    text = (
      rf"$\rho_k={rho[k]:.5g}$; $\epsilon={epsilon:.5g}$.  " + status
      + "\n" + r"$U_{k+1}+U_{k+1}=U_k$; consequently "
      + r"$s_m-s_k\in U_k$ for every $m>k$."
      + "\n" + r"Shrinking $U_k$ eventually fit inside any prescribed neighborhood of $0$."
    )

  elif view == "nested":
    for j in range(k + 2):
      label = rf"$C_{j}=s_{j}+U_{j}$" if j in {k, k + 1} else "_nolegend_"
      _plot_boundary(ax, _boundary(s[j], rho[j]), j, label)

    _curve(ax, s[:k + 2], marker="o", markersize=3.5, linewidth=1.0, label="Partial sums")
    _point(ax, s[k], marker="s", markersize=7, label=rf"$s_{k}$")
    _point(ax, x_star, marker="*", markersize=13, label=r"Limit $x$")
    ax.annotate("", xy=s[k + 1], xytext=s[k], arrowprops=dict(arrowstyle="->", linewidth=1.3))
    _limits(ax, s[k] if zoom else zero, 1.22 * (rho[k] if zoom else rho[0]))
    move = np.linalg.norm(s[k + 1] - s[k])
    text = (
      rf"$\|s_{{k+1}}-s_k\|+\rho_{{k+1}}="
      rf"{move + rho[k + 1]:.5g}\leq {rho[k]:.5g}=\rho_k$." + "\n" 
      + r"$C_{k+1}=s_k+x_{k+1}+U_{k+1}\subseteq s_k+U_k=C_k$." + "\n" 
      + r"Closed, nested sets; $\mathrm{diam}(C_k)=2\rho_k\to0$; "
      + r"$\bigcap_{k\geq0}C_k=\{x\}$."
    )

  elif view == "images":
    for j in range(k + 2):
      label = rf"$T(C_{j})$" if j in {k, k + 1} else "_nolegend_"
      _plot_boundary(ax, _boundary(Ts[j], rho[j], A), j, label)

    _curve(ax, Ts[:k + 2], marker="o", markersize=3.5, linewidth=1.0, label=r"Images $Ts_j$")
    _point(ax, Ts[k], marker="s", markersize=7, label=rf"$Ts_{k}$")
    _point(ax, y, marker="*", markersize=13, label=r"Target $y$")
    ax.annotate("", xy=y, xytext=Ts[k],
                arrowprops=dict(arrowstyle="->", linewidth=1.3))

    _limits(ax, Ts[k] if zoom else zero, 1.22 * opnorm * (rho[k] if zoom else rho[0]))
    text = (
      r"The arrow from $Ts_k$ to $y$ is the residual $r_k=y-Ts_k$."
      + "\n" + r"$y\in T(C_{k+1})\subseteq T(C_k)$ in this example; "
      + r"$\|r_k\|=" + f"{np.linalg.norm(residuals[k]):.5g}" + r"$."
      + "\n" + r"As $C_k$ shrink to $x$, their images shrink to $Tx=y$."
    )

  else:  # residuals
    for j in range(1, k + 2):
      label = rf"$F_{j}=T(U_{j})$" if j in {max(1, k), k + 1} else "_nolegend_",
      _plot_boundary(ax, _boundary(zero, rho[j], A), j, label)

    _curve(ax, residuals[:k + 1], marker="o", markersize=3.5,
            linewidth=1.0, label=r"Residuals $r_0,\ldots,r_k$")
    _point(ax, residuals[k], marker="s", markersize=7, label=rf"$r_{k}$")
    _point(ax, zero, marker="+", markersize=11, label=r"Limit $0$")
    _limits(ax, zero, 1.22 * opnorm * (rho[k + 1] if zoom else rho[1]))
    text = (
      r"$r_k=T(e_k)$ and $\|e_k\|=0.6\rho_{k+1}<\rho_{k+1}$; hence $r_k\in F_{k+1}$."
      + "\n" + r"$\|r_k\|\leq\|T\|\rho_{k+1}="
      + f"{opnorm * rho[k + 1]:.5g}" + r"\longrightarrow0$."
      + "\n" + r"Continuity and $s_k\to x$ give $Ts_k\to Tx$; "
      + r"also $Ts_k=y-r_k\to y$."
    )

  ax.ticklabel_format(style="sci", axis="both", scilimits=(-3, 3), useOffset=False)
  return _finish(fig, ax, text)




# ---------- 3. Optional notebook controls ----------
def notebook_app(demo=None):
  """Display live controls. Uses inline Matplotlib, so no ipympl is needed."""
  try:
      import ipywidgets as widgets
      from IPython.display import display
  except ImportError as exc:
      raise ImportError("Install ipywidgets, or use the exported HTML explorer.") from exc
  if demo is None:
      demo = make_demo()

  step = widgets.IntSlider(value=min(2, demo["steps"]), min=0, max=demo["steps"],
                            description="Step k", continuous_update=False)
  eps = widgets.FloatLogSlider(value=0.25, base=10, min=-3, max=0.2,
                                step=0.05, description="epsilon",
                                readout_format=".3g", continuous_update=False)
  zoom = widgets.Checkbox(value=False, description="Local zoom")
  view = widgets.Dropdown(options=[(title, key) for key, title in VIEWS.items()]
                          + [("Show all four separately", "all")],
                          value="nested", description="View")

  def update(k, epsilon, zoom, view):
      for name in (VIEWS if view == "all" else [view]):
          fig = plot_view(demo, name, k, epsilon, zoom)
          display(fig)
          plt.close(fig)

  output = widgets.interactive_output(update, dict(k=step, epsilon=eps,
                                                    zoom=zoom, view=view))
  ui = widgets.VBox([view, step, eps, zoom, output])
  display(ui)
  return ui


# ---------- 4. Offline, browser-only explorer ----------
def export_html(demo: dict, path="open_mapping_explorer.html",
              epsilon_values=(0.04, 0.10, 0.25, 0.60)) -> Path:
  """Pre-render Matplotlib SVGs and embed them in an offline HTML page.

  No JavaScript libraries, server, internet connection, or Python runtime
  are needed to view the resulting file. Only the basis view depends on
  epsilon, so the other frames are shared across epsilon choices.
  """
  eps_values = [float(e) for e in epsilon_values]
  if not eps_values or not all(np.isfinite(e) and e > 0 for e in eps_values):
      raise ValueError("epsilon_values must be nonempty, positive and finite.")
  path = Path(path).expanduser().resolve()
  path.parent.mkdir(parents=True, exist_ok=True)
  frames = {}
  for view in VIEWS:
      indices = range(len(eps_values)) if view == "basis" else [0]
      for ei in indices:
          for zoom in (False, True):
              for k in range(demo["steps"] + 1):
                  fig = plot_view(demo, view, k, eps_values[ei], zoom)
                  buffer = io.StringIO()
                  fig.savefig(buffer, format="svg")
                  plt.close(fig)
                  svg = buffer.getvalue()
                  svg = svg[svg.index("<svg"):]
                  frames[f"{view}|{ei}|{int(zoom)}|{k}"] = svg
      print(f"Rendered: {VIEWS[view]}", flush=True)

  meta = {
      "steps": demo["steps"], "eps": eps_values,
      "views": VIEWS, "radius": float(demo["rho"][0]),
      "matrix": demo["A"].tolist(),
  }
  html = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Open mapping theorem — nested neighborhoods</title>
<style>
body {font: 16px/1.6 system-ui, sans-serif; max-width: 940px; margin: 28px auto; padding: 0 18px;}
h1 {line-height: 1.2; font-size: 30px; margin-bottom: 10px;}
h2 {font-size: 19px; margin: 22px 0 8px;}
p {margin: 8px 0 14px;}
.controls {display: flex; flex-wrap: wrap; align-items: center; gap: 12px 20px;
border: 1px solid; border-radius: 8px; padding: 14px;}
label {display: inline-flex; align-items: center; gap: 8px;}
select, button, input {font: inherit;}
select {max-width: 100%; padding: 5px;}
input[type=range] {width: 200px;}
button {padding: 5px 12px; cursor: pointer;}
#plot {margin: 8px 0; min-height: 360px;}
#plot svg {width: 100%; height: auto; display: block;}
.note {border-left: 3px solid; padding-left: 14px;}
code {font-size: 0.95em;}
</style></head><body>
<h1>Open mapping theorem: nested neighborhoods</h1>
<p>Track the allowed corrections, the moving sets containing the limit, and their
images. Each view is a separate Matplotlib figure of the same construction.</p>
<div class="controls">
<label>View <select id="view" aria-label="Proof view"></select></label>
<label>Step k <input id="step" aria-label="Step k" type="range" min="0" step="1" value="2">
<output id="stepvalue">2</output></label>
<label><input id="zoom" type="checkbox"> Local zoom</label>
<label>Test &epsilon; <select id="eps" aria-label="Test neighborhood radius"></select></label>
<button id="play" type="button">Play</button>
</div>
<p id="hint"></p>
<div id="plot" role="img" aria-label="Interactive mathematical visualization"></div>
<h2>Read the four views together</h2>
<p><strong>1.</strong> U<sub>k</sub> shrink around zero. The irregular open neighborhood W
contains B(0,&epsilon;); once &rho;<sub>k</sub>&lt;&epsilon;, U<sub>k</sub> fits inside W.
The small marked vectors are finite tails s<sub>m</sub>&minus;s<sub>k</sub>, not the partial sums themselves.</p>
<p><strong>2.</strong> C<sub>k</sub>=s<sub>k</sub>+U<sub>k</sub> shrink around the eventual
limit. The arrow shows the next correction. Its length plus the next radius is at
most the current radius, so the entire next ball stays inside.</p>
<p><strong>3.</strong> The corresponding image ellipses retain the same target y.
The arrow from Ts<sub>k</sub> to y is the current residual.</p>
<p><strong>4.</strong> Translating that residual back to the origin places it in
F<sub>k+1</sub>. These neighborhoods shrink to zero, giving
Ts<sub>k</sub>=y&minus;r<sub>k</sub>&rarr;y.</p>
<h2>What this example does — and does not — show</h2>
<p class="note">This is a deliberately constructed example in X=Y=&Ropf;<sup>2</sup>,
with an invertible matrix T and closed balls. Here T(U<sub>k</sub>) is a closed ellipse,
so F<sub>k</sub>=closure(T(U<sub>k</sub>))=T(U<sub>k</sub>).
The illustration therefore has actual image membership, a stronger property than the
closure membership available in the general proof. It visualizes the geometry;
it does not prove the general closure-removal step.</p>
<p>The target x is chosen in advance to generate an admissible spiraling path.
This is not a general-purpose algorithm for finding unknown preimages.
The analytic formulas define all k&ge;0; the slider displays a finite initial segment.</p>
<p>Local zoom changes the coordinate window at each step; turn it off to compare
absolute sizes. Only view 1 uses the test-&epsilon; control. Circles and ellipses
outline whole sets; the sets are not merely their boundaries.</p>
<details><summary>Construction and uniform bounds</summary>
<p>Let &rho;<sub>k</sub>=&rho;<sub>0</sub>2<sup>&minus;k</sup> and let v<sub>k</sub>
be unit vectors rotating by 110 degrees per step in the default example.
Set e<sub>k</sub>=0.6&rho;<sub>k+1</sub>v<sub>k</sub>, x=e<sub>0</sub>,
s<sub>k</sub>=x&minus;e<sub>k</sub>, and y=Tx. Then s<sub>0</sub>=0 and</p>
<p>&Vert;x<sub>k+1</sub>&Vert;=&Vert;e<sub>k</sub>&minus;e<sub>k+1</sub>&Vert;
&le;0.9&rho;<sub>k+1</sub>&lt;&rho;<sub>k+1</sub>.<br>
r<sub>k</sub>=Te<sub>k</sub>&isin;F<sub>k+1</sub>, and
U<sub>k+1</sub>+U<sub>k+1</sub>=U<sub>k</sub>.</p>
<p id="parameters"></p></details>
<noscript>This explorer requires JavaScript to switch among its embedded figures.</noscript>
<script>
const frames = __FRAMES__;
const meta = __META__;
const $ = id => document.getElementById(id);
for (const [key, title] of Object.entries(meta.views)) {
const o = document.createElement('option'); o.value = key; o.textContent = title;
$('view').append(o);
}
meta.eps.forEach((e, i) => {
const o = document.createElement('option'); o.value = i; o.textContent = e;
$('eps').append(o);
});
$('view').value = 'nested';
$('eps').value = Math.min(2, meta.eps.length - 1);
$('step').max = meta.steps;
$('step').value = Math.min(2, meta.steps);
$('parameters').textContent = 'Radius rho_0 = ' + meta.radius + '; T = ' + JSON.stringify(meta.matrix) + '.';
function render() {
const view = $('view').value, k = Number($('step').value);
const ei = view === 'basis' ? Number($('eps').value) : 0;
const zoom = Number($('zoom').checked);
$('stepvalue').textContent = k;
$('eps').disabled = view !== 'basis';
$('plot').innerHTML = frames[`${view}|${ei}|${zoom}|${k}`];
$('plot').setAttribute('aria-label', meta.views[view] + ', step ' + k);
$('hint').textContent = $('zoom').checked
  ? 'Local zoom: the axis scale changes with k. Older outlines may lie outside the window.'
  : 'Global view: the axis scale stays fixed as k changes.';
}
['view','eps','zoom','step'].forEach(id => $(id).addEventListener('input', render));
let timer = null;
$('play').addEventListener('click', () => {
if (timer !== null) { clearInterval(timer); timer = null; $('play').textContent = 'Play'; return; }
$('play').textContent = 'Pause';
timer = setInterval(() => {
  $('step').value = (Number($('step').value) + 1) % (meta.steps + 1); render();
}, 1000);
});
render();
</script></body></html>'''
  html = html.replace("__FRAMES__", json.dumps(frames).replace("</", "<\\/"))
  html = html.replace("__META__", json.dumps(meta))
  path.write_text(html, encoding="utf-8")
  return path


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--steps", type=int, default=8)
  parser.add_argument("--html", default="open_mapping_explorer.html")
  args = parser.parse_args()
  print(export_html(make_demo(steps=args.steps), args.html))
