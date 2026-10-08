"""Open mapping theorem: linked X and Y visualizations.

Adapted from the user's updated visualization. The mathematical construction
in make_demo is unchanged. Requires numpy and matplotlib; ipywidgets is optional.

Notebook:
  demo = make_demo(steps=10)
  notebook_app(demo)

Static pair (two independent Matplotlib figures, arranged together in HTML):
  display_spaces(demo, k=2, zoom=True, focus="both")

Offline explorer:
  python open_mapping_xy.py --steps 8 --html open_mapping_xy.html

Dotted outlines: origin neighborhoods U_j and F_j.
Solid / dashed outlines: translated sets C_j and T(C_j), current / next.
Previous scales use the user's Reds palette; active scales use Greens.
The two spaces use identical coordinate units and equal axis aspect ratios.
"""
from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path
from uuid import uuid4

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# ---------- 1. Mathematical construction, retained from the supplied code ----------

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
    raise ValueError(f"steps must be an integer between 1 and {max_steps}.")
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

  return dict(steps=steps, A=A, rho=rho_k, e=e_k, x_star=x_star,
              s=s, y=y, Ts=Ts, residuals=residuals,
              corrections=corrections, opnorm=singular_values[0])


# ---------- 2. Two combined panels ----------
SPACES = {"X": "X  |  Neighborhoods and partial sums",
          "Y": "Y  |  Image neighborhoods and residuals"}
FOCI = ("both", "origin", "translated")


def _boundary(center, radius, A=None):
  theta = np.linspace(0, 2 * np.pi, 361)
  points = radius * np.column_stack((np.cos(theta), np.sin(theta)))
  if A is not None:
      points = points @ np.asarray(A).T
  return np.asarray(center) + points


def _curve(ax: Axes, points, **kwargs):
  return ax.plot(points[:, 0], points[:, 1], **kwargs)[0]


def _w_radius(epsilon, theta):
  return epsilon * (1.45 + 0.25 * np.cos(3 * theta) + 0.15 * np.sin(5 * theta))


def _validate(demo, k, epsilon, focus):
  if not isinstance(k, (int, np.integer)) or not 0 <= k <= demo["steps"]:
    raise ValueError(f"k must be an integer between 0 and {demo['steps']}.")
  if not np.isfinite(epsilon) or epsilon <= 0:
    raise ValueError("epsilon must be positive and finite.")
  if focus not in FOCI:
    raise ValueError(f"focus must be one of {FOCI}.")


def _windows(demo, k, epsilon, zoom, focus):
  """Separate centers but a COMMON half-width: T's scaling stays visible."""
  centers = {"X": demo["s"][k], "Y": demo["Ts"][k]}
  scale = max(1.0, demo["opnorm"])
  if not zoom:
    half = 1.12 * max(scale * demo["rho"][0], 1.85 * epsilon)
    return {space: (np.zeros(2), half) for space in SPACES}
  radius = scale * demo["rho"][k]
  if focus == "origin":
    return {space: (np.zeros(2), 1.22 * radius) for space in SPACES}
  if focus == "translated":
    return {space: (center, 1.22 * radius) for space, center in centers.items()}
  # Fit both the origin-centered and translated families. As k grows,
  # their separated centers prevent this window shrinking to zero.
  half = 1.12 * (radius + 0.5 * max(np.max(np.abs(c)) for c in centers.values()))
  return {space: (center / 2, half) for space, center in centers.items()}


def _scale_style(demo, j, k, family):
  """Preserve the supplied Reds/Greens scheme, avoiding near-white levels."""
  level = 0.50 + 0.45 * min(j / (demo["steps"] + 1), 1.0)
  color = plt.get_cmap("Greens" if j >= k else "Reds")(level)
  return dict(
      color=color,
      alpha=1.0 if j >= k else 0.24,
      linewidth=2.5 if j == k else (1.65 if j == k + 1 else 1.0),
      linestyle=":" if family == "origin" else ("--" if j == k + 1 else "-"),
      zorder=3 if j >= k else 1,
  )


def _mark(ax: Axes, point, label, marker, offset=(7, 7), color="black", size=7):
  ax.plot(*point, marker=marker, markersize=size, linestyle="None",
          color=color, zorder=9)
  ax.annotate(label, point, xytext=offset, textcoords="offset points",
              color=color, fontsize=10, zorder=10, annotation_clip=True)


def _arrow(ax: Axes, start, end, color="black", linewidth=1.3):
  ax.annotate("", xy=end, xytext=start, annotation_clip=True,
              arrowprops=dict(arrowstyle="->", color=color, linewidth=linewidth, shrinkA=2, shrinkB=2),
              zorder=7)


def plot_space(
  demo: dict,
  space: str = "X",
  k: int = 2,
  epsilon: float = 0.25,
  zoom: bool = False,
  focus: str = "both",
  show_history: bool = True,
  show_test: bool = True,
) -> Figure:
  """Draw one combined panel. X merges old views 1/2; Y merges views 3/4.

  Both families remain plotted at their actual coordinates. `focus` controls
  the window only when zoom=True. The test neighborhood W occurs only in X.
  An outline represents the whole closed ball/ellipse, not just its boundary.
  """
  _validate(demo, k, epsilon, focus)
  space = space.upper()
  if space not in SPACES:
      raise ValueError("space must be 'X' or 'Y'.")
  rho, A = demo["rho"], demo["A"]
  is_x = space == "X"
  transform = None if is_x else A
  partials = demo["s"] if is_x else demo["Ts"]
  target = demo["x_star"] if is_x else demo["y"]
  errors = demo["e"] if is_x else demo["residuals"]
  current, zero = partials[k], np.zeros(2)

  fig = plt.figure(figsize=(7.0, 8.0))
  ax = fig.add_axes((0.115, 0.27, 0.84, 0.66), axes_class=Axes)
  ax.set_aspect("equal", adjustable="box")
  ax.set(xlabel="First coordinate", ylabel="Second coordinate")
  ax.grid(True, alpha=0.18)
  fig.suptitle(SPACES[space], fontsize=15, y=0.986)
  fig.text(0.5, 0.945, f"k = {k}   |   dotted: at 0   |   solid / dashed: translated",
            ha="center", fontsize=10)

  # These are the two former views overlaid on the SAME coordinate axes.
  # A -> None yields circles in X; A yields their image ellipses in Y.
  for j in range(k + 2):
      if j < k and not show_history:
          continue
      origin_label = rf"$U_{{{j}}}$" if is_x else rf"$F_{{{j}}}$"
      translated_label = (rf"$C_{{{j}}}$" if is_x else rf"$T(C_{{{j}}})$")
      origin = _curve(ax, _boundary(zero, rho[j], transform),
                      label=origin_label if j >= k else "_nolegend_",
                      **_scale_style(demo, j, k, "origin"))
      shifted = _curve(ax, _boundary(partials[j], rho[j], transform),
                        label=translated_label if j >= k else "_nolegend_",
                        **_scale_style(demo, j, k, "translated"))
      # SVG tags allow the offline explorer to hide history without
      # pre-rendering every combination of checkbox states.
      if j < k:
          origin.set_gid(f"history-origin-{j}")
          shifted.set_gid(f"history-translated-{j}")

  if is_x and show_test:
      theta = np.linspace(0, 2 * np.pi, 601)
      w = _w_radius(epsilon, theta)[:, None] * np.column_stack((np.cos(theta), np.sin(theta)))
      for name, points, style, color, label in (
          ("W", w, "--", "slategray", r"Boundary of $W$"),
          ("epsilon", _boundary(zero, epsilon), ":", "mediumpurple",
            r"Boundary of $B(0,\epsilon)$"),
      ):
          line = _curve(ax, points, linestyle=style, linewidth=1.3,
                        color=color, alpha=0.8, label=label, zorder=2)
          line.set_gid(f"test-{name}")

  # Partial sums/images are actual positions; tails/errors are vectors based
  # at 0. Distinct markers prevent conflating these two roles.
  _curve(ax, partials[:k + 2], color="black", marker="o", markersize=2.7,
          linewidth=0.8, alpha=0.55, zorder=4)
  if is_x:
      tails = demo["s"][k + 1:] - demo["s"][k]
      _curve(ax, tails, color="black", marker="o", markerfacecolor="none",
              markersize=3.0, linewidth=0.7, alpha=0.65, zorder=5)
  else:
      _curve(ax, errors[:k + 1], color="mediumpurple", marker=".",
              markersize=3.0, linewidth=0.8, alpha=0.65, zorder=5)

  # The same error vector is drawn once at 0 and once at the moving center.
  # X: 0 -> e_k and s_k -> x.    Y: 0 -> r_k and Ts_k -> y.
  _arrow(ax, zero, errors[k], color="mediumpurple")
  _arrow(ax, current, target, color="mediumpurple")
  _arrow(ax, current, partials[k + 1])
  _mark(ax, zero, "$0$", "+", offset=(-13, -17), size=9)
  _mark(ax, current, rf"$s_{{{k}}}$" if is_x else rf"$Ts_{{{k}}}$", "s", offset=(7, 7))
  _mark(ax, target, "$x$" if is_x else "$y$", "*", offset=(7, -15), size=12)
  _mark(ax, errors[k], rf"$e_{{{k}}}$" if is_x else rf"$r_{{{k}}}$", "D",
        offset=(7, -15), color="mediumpurple", size=5)

  center, half = _windows(demo, k, epsilon, zoom, focus)[space]
  ax.set(xlim=(center[0] - half, center[0] + half),
          ylim=(center[1] - half, center[1] + half))
  ax.ticklabel_format(style="sci", axis="both", scilimits=(-3, 3), useOffset=False)
  handles, labels = ax.get_legend_handles_labels()
  legend = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 0.21),
                      ncol=3 if is_x else 2, frameon=False, fontsize=10)
  for i, (line, text_item) in enumerate(zip(legend.get_lines(), legend.get_texts())):
      if text_item.get_text().startswith("Boundary of"):
          line.set_gid(f"test-legend-line-{i}")
          text_item.set_gid(f"test-legend-text-{i}")

  if is_x:
      text = (r"$C_k=s_k+U_k,\quad C_{k+1}\subseteq C_k$" + "\n"
              + r"Open circles: tails $s_m-s_k\in U_k$ for $m>k$." + "\n"
              + r"Purple arrows: $e_k=x-s_k$, drawn at $0$ and at $s_k$.")
  else:
      text = (r"$T(C_k)=Ts_k+F_k,\quad y\in T(C_{k+1})\subseteq T(C_k)$" + "\n"
              + r"$r_k=y-Ts_k\in F_{k+1},\qquad r_k\longrightarrow0$." + "\n"
              + r"Purple arrows: the same $r_k$, drawn at $0$ and at $Ts_k$.")
  fig.text(0.075, 0.025, text, fontsize=10, va="bottom", linespacing=1.6)
  return fig


def plot_spaces(demo: dict, k: int = 2, epsilon: float = 0.25,
              zoom: bool = False, focus: str = "both",
              show_history: bool = True, show_test: bool = True) -> tuple[Figure, Figure]:
  """Return (figure_X, figure_Y), two separate but synchronized figures."""
  options = dict(k=k, epsilon=epsilon, zoom=zoom, focus=focus,
                  show_history=show_history, show_test=show_test)
  fig_x = plot_space(demo, "X", **options)
  try:
    fig_y = plot_space(demo, "Y", **options)
  except Exception:
    plt.close(fig_x)
    raise
  return fig_x, fig_y


def _svg(fig: Figure, prefix: str) -> str:
  """Unique SVG ids prevent clip-path/marker collisions across the two panels."""
  buffer = io.StringIO()
  fig.savefig(buffer, format="svg")
  svg = buffer.getvalue()
  svg = svg[svg.index("<svg"):]
  ids = re.findall(r'\bid="([^"]+)"', svg)
  # Replace references BEFORE replacing declarations. Keep history/test
  # words in the ids so the browser's layer toggles can identify them.
  for old in sorted(set(ids), key=len, reverse=True):
    new = f"{prefix}-{old}"
    svg = svg.replace(f'url(#{old})', f'url(#{new})')
    svg = svg.replace(f'href="#{old}"', f'href="#{new}"')
    svg = svg.replace(f'id="{old}"', f'id="{new}"')
  return svg


def _pair_html(figures: tuple[Figure, Figure]) -> str:
  prefix = "xy" + uuid4().hex
  panes = [_svg(fig, f"{prefix}-{space}") for space, fig in zip(SPACES, figures)]
  return (f'<div id="{prefix}" style="overflow-x:auto;">'
          '<div style="display:flex;align-items:flex-start;min-width:760px;">'
          + "".join('<div style="width:50%;min-width:0;">' + svg + '</div>' for svg in panes)
          + '</div></div>'
          + f'<style>#{prefix} svg {{width:100%;height:auto;display:block;}}</style>')


def display_spaces(demo: dict, **kwargs):
  """Show the two figures SIDE BY SIDE in a notebook, without a widget backend.

  Returns an IPython HTML object; figures are closed after serialization,
  preventing automatic duplicate displays at the end of a notebook cell.
  """
  from IPython.display import HTML, display
  figures = plot_spaces(demo, **kwargs)
  try:
    result = HTML(_pair_html(figures))
    display(result)
  finally:
    for fig in figures:
      plt.close(fig)
  return result


# ---------- 3. Synchronized notebook controls ----------
def notebook_app(demo=None):
  """Live X/Y panels. Install ipywidgets, or use the offline HTML explorer."""
  try:
    import ipywidgets as widgets
    from IPython.display import HTML, display
  except ImportError as exc:
    raise ImportError("Install ipywidgets or use display_spaces()/the offline HTML.") from exc
  if demo is None:
      demo = make_demo()
  step = widgets.IntSlider(value=min(2, demo["steps"]), min=0, max=demo["steps"],
                            description="Step k", continuous_update=False)
  eps = widgets.FloatLogSlider(value=0.25, base=10, min=-4, max=0.3, step=0.05,
                                description="epsilon", readout_format=".3g",
                                continuous_update=False)
  mode = widgets.Dropdown(options=[("Global: fixed scale", "global"),
                                    ("Zoom: both families", "both"),
                                    ("Zoom: origin neighborhoods", "origin"),
                                    ("Zoom: translated neighborhoods", "translated")],
                          value="both", description="Window",
                          layout=widgets.Layout(width="340px"))
  history = widgets.Checkbox(value=True, description="Earlier outlines")
  test = widgets.Checkbox(value=True, description="Show W and epsilon-ball")
  note = widgets.HTML()

  def update(k, epsilon, mode, show_history, show_test):
    radius = demo["rho"][k]
    statement = (f"rho_{k} = {radius:.5g} < epsilon = {epsilon:.5g}: "
                  "U_k lies inside B(0, epsilon), hence inside W."
                  if radius < epsilon else
                  f"rho_{k} = {radius:.5g} >= epsilon = {epsilon:.5g}: "
                  "this radius test does not yet certify U_k is inside W.")
    note.value = ("<b>Neighborhood test:</b> " + statement
                  + "<br>Matched axis scales. Zoomed windows may clip W or earlier outlines.")
    figures = plot_spaces(demo, k=k, epsilon=epsilon, zoom=mode != "global",
                          focus=mode if mode != "global" else "both",
                          show_history=show_history, show_test=show_test)
    try:
      display(HTML(_pair_html(figures)))
    finally:
      for fig in figures:
        plt.close(fig)

  output = widgets.interactive_output(update, dict(k=step, epsilon=eps, mode=mode,
                                                    show_history=history, show_test=test))
  ui = widgets.VBox([widgets.HBox([step, eps]),
                      widgets.HBox([mode, history, test]), note, output])
  display(ui)
  return ui


# ---------- 4. Offline explorer: two figures, arranged in a browser grid ----------
def export_html(demo: dict, path="open_mapping_xy.html",
              epsilon_values=(0.04, 0.10, 0.25, 0.60)) -> Path:
  """Embed all plot frames. No internet, server, or Python needed to view.

  The windows have matched X/Y scales. Only X includes W. A global frame in
  Y also depends on epsilon, because the shared window must accommodate W.
  SVG groups allow history/test layers to be toggled without more frames.
  """
  eps_values = [float(e) for e in epsilon_values]
  if not eps_values or not all(np.isfinite(e) and e > 0 for e in eps_values):
      raise ValueError("epsilon_values must be nonempty, positive, and finite.")
  path = Path(path).expanduser().resolve()
  path.parent.mkdir(parents=True, exist_ok=True)
  modes = ("global", "both", "origin", "translated")
  frames = {}
  for space in SPACES:
    for mode in modes:
      # For zoomed Y, epsilon affects neither the window nor the content.
      indices = range(len(eps_values)) if space == "X" or mode == "global" else [0]
      for ei in indices:
        for k in range(demo["steps"] + 1):
          fig = plot_space(demo, space, k=k, epsilon=eps_values[ei],
                            zoom=mode != "global",
                            focus=mode if mode != "global" else "both")
          try:
            frames[f"{space}|{mode}|{ei}|{k}"] = _svg(fig, f"{space}-{mode}-{ei}-{k}")
          finally:
            plt.close(fig)
      print(f"Rendered {space}: {mode}", flush=True)
  meta = dict(steps=demo["steps"], eps=eps_values, radius=float(demo["rho"][0]),
              matrix=demo["A"].tolist(), opnorm=float(demo["opnorm"]))
  html = _HTML.replace("__FRAMES__", json.dumps(frames).replace("</", "<\\/"))
  html = html.replace("__META__", json.dumps(meta).replace("</", "<\\/"))
  path.write_text(html, encoding="utf-8")
  return path


_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Open mapping theorem | X and Y</title>
<style>
body{font:16px/1.5 system-ui,sans-serif;max-width:1450px;margin:24px auto;padding:0 22px;}
h1{font-size:30px;line-height:1.2;margin:0 0 10px} h2{font-size:20px}
p{margin:10px 0}.controls{display:flex;flex-wrap:wrap;align-items:center;gap:10px 22px;
border:1px solid;border-radius:8px;padding:13px;margin:18px 0 12px;}
label{display:inline-flex;align-items:center;gap:8px}input,select,button{font:inherit}
input[type=range]{width:175px}button,select{padding:5px 9px}button{cursor:pointer}
.pair-scroll{overflow-x:auto}.pair{display:grid;grid-template-columns:1fr 1fr;min-width:780px;}
.panel{min-width:0}.panel svg{display:block;width:100%;height:auto}
#hint{font-size:14px}#test-status{font-size:15px;margin-bottom:2px}
.hide-history [id*="history-"]{display:none}.hide-test [id*="test-"]{display:none}
.key{border-top:1px solid;padding-top:14px}.key p{margin:8px 0}
code{font-size:.95em}.math{font-family:Georgia,serif;font-size:1.08em}
</style></head><body>
<h1>Open mapping theorem: the two spaces together</h1>
<p><span class="math">X &xrarr;<sup>T</sup> Y</span> &nbsp;|&nbsp;
Left: neighborhoods and their translations. Right: their images and the residuals.
All controls update both spaces; coordinate scales are matched.</p>
<div class="controls">
<label>Step k <input id="step" type="range" min="0" step="1" value="2" aria-label="Step k">
<output id="step-value">2</output></label>
<label>Window <select id="mode" aria-label="Coordinate window">
<option value="global">Global: fixed scale</option><option value="both" selected>Zoom: both families</option>
<option value="origin">Zoom: origin neighborhoods</option>
<option value="translated">Zoom: translated neighborhoods</option></select></label>
<label>Test &epsilon; <select id="eps" aria-label="Test epsilon"></select></label>
<label><input type="checkbox" id="history" checked>Earlier outlines</label>
<label><input type="checkbox" id="test" checked>Show W and epsilon-ball</label>
<button id="play" type="button">Play</button>
</div>
<p id="test-status" aria-live="polite"></p><p id="hint"></p>
<div id="plots" class="pair-scroll"><div class="pair">
<div id="plot-X" class="panel" role="img" aria-label="Domain X"></div>
<div id="plot-Y" class="panel" role="img" aria-label="Codomain Y"></div>
</div></div>
<div class="key">
<p><b>Read across:</b> U<sub>k</sub> maps to F<sub>k</sub>; C<sub>k</sub>=s<sub>k</sub>+U<sub>k</sub>
maps to T(C<sub>k</sub>)=Ts<sub>k</sub>+F<sub>k</sub>. Squares mark the partial sum and its image;
stars mark x and y; diamonds mark e<sub>k</sub> and r<sub>k</sub>=Te<sub>k</sub>.</p>
<p><b>Read within each space:</b> dotted neighborhoods are centered at 0; solid translated sets are
centered at s<sub>k</sub> or Ts<sub>k</sub>. The next translated set is dashed. Earlier outlines use
red and the current/next outlines use green. Purple arrows draw the same error twice: at 0 and at
the moving center. A black arrow shows the next correction.</p>
<p><b>Neighborhood-basis argument:</b> the hollow circles in X are sample tails
s<sub>m</sub>&minus;s<sub>k</sub>, m&gt;k, inside U<sub>k</sub>. Once &rho;<sub>k</sub>&lt;&epsilon;,
they lie inside B(0,&epsilon;)&sub;W. W is open: the irregular gray curve is its boundary.</p>
<p><b>Zoom:</b> “both families” keeps both centers visible; the centers remain separated as k grows.
Use “origin” or “translated” to follow a single shrinking family. The window changes with k in all
zoom modes. In “global” it remains fixed for a fixed epsilon. Both panels always use the same units.</p>
<details><summary>Construction and limitation of this illustration</summary>
<p>The supplied mathematical construction is retained: &rho;<sub>k</sub>=&rho;<sub>0</sub>2<sup>&minus;k</sup>,
e<sub>k</sub>=0.6&rho;<sub>k+1</sub>v<sub>k</sub>, x=e<sub>0</sub>, s<sub>k</sub>=x&minus;e<sub>k</sub>.
The vectors v<sub>k</sub> rotate, producing the spiral; x is known in advance.</p>
<p>The correction has norm at most 0.9&rho;<sub>k+1</sub>, so
C<sub>k+1</sub>&sube;C<sub>k</sub>. Also e<sub>k</sub>&isin;U<sub>k+1</sub>, hence
r<sub>k</sub>&isin;F<sub>k+1</sub>.</p>
<p>This is an invertible-matrix example in X=Y=R<sup>2</sup>. Images of closed balls are closed ellipses,
so F<sub>k</sub>=closure(T(U<sub>k</sub>))=T(U<sub>k</sub>). Actual image membership is stronger than
the closure membership in the general proof. Only finitely many terms of the infinite construction are drawn.</p>
<p id="parameters"></p></details>
</div>
<noscript>JavaScript is required to switch between the embedded figures.</noscript>
<script>
const frames=__FRAMES__, meta=__META__;
const $=id=>document.getElementById(id);
meta.eps.forEach((e,i)=>{let o=document.createElement('option');o.value=i;o.textContent=e;$('eps').append(o);});
$('eps').value=Math.min(2,meta.eps.length-1);$('step').max=meta.steps;$('step').value=Math.min(2,meta.steps);
$('parameters').textContent='rho_0 = '+meta.radius+'; T = '+JSON.stringify(meta.matrix)+'.';
function layers(){
$('plots').classList.toggle('hide-history',!$('history').checked);
$('plots').classList.toggle('hide-test',!$('test').checked);
}
function render(){
const k=Number($('step').value), mode=$('mode').value, ei=Number($('eps').value);
$('step-value').textContent=k;
for(const space of ['X','Y']){
  const index=(space==='X'||mode==='global')?ei:0;
  const key=`${space}|${mode}|${index}|${k}`;
  $('plot-'+space).innerHTML=frames[key];
  $('plot-'+space).setAttribute('aria-label',`${space}: step ${k}, ${mode} window`);
}
const rho=meta.radius*2**(-k),eps=meta.eps[ei];
$('test-status').textContent='Neighborhood test: rho_'+k+' = '+rho.toPrecision(4)
+(rho<eps?' < ':' >= ')+'epsilon = '+eps+'. '
+(rho<eps?'U_k lies inside B(0, epsilon), hence inside W.':'This radius test does not yet certify U_k is inside W.');
$('hint').textContent=mode==='global'
?'Global view: the window is fixed as k changes. X and Y use the same scale.'
:'Zoomed view: the window changes with k. W and earlier outlines may be clipped; X and Y use the same scale.';
layers();
}
['step','mode','eps'].forEach(id=>$(id).addEventListener('input',render));
['history','test'].forEach(id=>$(id).addEventListener('change',layers));
let timer=null;$('play').addEventListener('click',()=>{
if(timer!==null){clearInterval(timer);timer=null;$('play').textContent='Play';return;}
$('play').textContent='Pause';timer=setInterval(()=>{
  $('step').value=(Number($('step').value)+1)%(meta.steps+1);render();
},1000);
});
render();
</script></body></html>'''


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--steps", type=int, default=8)
  parser.add_argument("--html", default="open_mapping_xy.html")
  args = parser.parse_args()
  print(export_html(make_demo(steps=args.steps), args.html))
