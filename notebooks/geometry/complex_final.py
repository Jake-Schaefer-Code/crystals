# 
import jax.numpy as jnp
import numpy as onp
import plotly.graph_objects as go
import io

try:
  import imageio.v3 as iio
except Exception:
  iio = None


def f(z):
	return jnp.pi / (jnp.tan(jnp.pi * z) * z ** 2) #  * z ** 2

def G(z):
  return jnp.cos(jnp.pi * z) * jnp.pi / (jnp.sin(jnp.pi * z))

num = 1000
N0 = 3  # initial N for the contour
M = 6   # plotting domain size; keep fixed for animation
x = (M + 0.5) * (2 * jnp.linspace(0, 1, num) - 1)
y = M * (2 * jnp.linspace(0, 1, num) - 1)
X, Y = jnp.meshgrid(x, y)
Z = X + 1j * Y

mod_fz = jnp.abs(f(Z))


mod_np = onp.nan_to_num(onp.array(mod_fz), nan=0.0, posinf=0.0)
cap = onp.nanpercentile(mod_np[onp.isfinite(mod_np) & (mod_np > 0)], 99) if onp.any(onp.isfinite(mod_np)) else 1.0
mod_capped = onp.minimum(mod_np, cap)

max_z = 8
mod_capped = onp.minimum(mod_capped, max_z)

X_np, Y_np = onp.array(X), onp.array(Y)

levels = 20
xmin, xmax = float(X_np.min()), float(X_np.max())
ymin, ymax = float(Y_np.min()), float(Y_np.max())
zmin, zmax = float(mod_capped.min()), float(mod_capped.max())


n_slice = 12
x_step = (xmax - xmin) / (n_slice - 1)
y_step = (ymax - ymin) / (n_slice - 1)
z_levels = 20
z_step = (zmax - zmin) / (z_levels - 1)


fig = go.Figure(
  data=[
    go.Surface(
      x=X_np,
      y=Y_np,
      z=mod_capped,
      colorscale=[[0.0, "lightgray"], [1.0, "lightgray"]],
      showscale=False,
      contours={
        "x": {
          "show": True,
          "start": xmin,
          "end": xmax,
          "size": x_step,
          "color": "black",
        },
        "y": {
          "show": True,
          "start": ymin,
          "end": ymax,
          "size": y_step,
          "color": "black",
        },
        "z": {
          "show": True,
          "start": zmin,
          "end": zmax,
          "size": z_step,
          "color": "black",
        },
      }
    )
  ]
)

def contour_on_surface(N, cap, max_z, n_edge=400):
  rect_a = float(N + 0.5)  # corners at ±(N+1/2) ± iN
  rect_b = float(N)

  t = jnp.linspace(0.0, 1.0, n_edge, endpoint=True)

  # Edges: (a,b)->(-a,b)->(-a,-b)->(a,-b)->(a,b)
  x1, y1 = rect_a + (-2.0 * rect_a) * t, rect_b + 0.0 * t
  x2, y2 = -rect_a + 0.0 * t, rect_b + (-2.0 * rect_b) * t
  x3, y3 = -rect_a + (2.0 * rect_a) * t, -rect_b + 0.0 * t
  x4, y4 = rect_a + 0.0 * t, -rect_b + (2.0 * rect_b) * t

  # Concatenate while avoiding duplicated corner points
  rect_x = jnp.concatenate([x1, x2[1:], x3[1:], x4[1:]])
  rect_y = jnp.concatenate([y1, y2[1:], y3[1:], y4[1:]])
  rect_z_raw = jnp.abs(f(rect_x + 1j * rect_y))

  rect_z = onp.nan_to_num(onp.array(rect_z_raw), nan=0.0, posinf=0.0)
  rect_z = onp.minimum(rect_z, cap)
  rect_z = onp.minimum(rect_z, max_z)
  rect_x = onp.array(rect_x)
  rect_y = onp.array(rect_y)
  return rect_x, rect_y, rect_z


rect_x, rect_y, rect_z = contour_on_surface(N0, cap, max_z)

fig.add_trace(
  go.Scatter3d(
    x=rect_x,
    y=rect_y,
    z=rect_z,
    mode="lines",
    line=dict(color="red", width=8),
    name="Contour",
    showlegend=False,
  )
)

N_values = list(range(1, 11))
frames = []
for N in N_values:
  fx, fy, fz = contour_on_surface(N, cap, max_z)
  frames.append(
    go.Frame(
      name=str(N),
      data=[go.Scatter3d(x=fx, y=fy, z=fz, mode="lines", line=dict(color="red", width=8), showlegend=False)],
      traces=[1],
    )
  )

fig.frames = frames

fig.update_layout(
  scene=dict(
    xaxis=dict(
      title="Re(z)",
      showbackground=False,
      showgrid=False,
      showline=True,
      ticks="outside",
      showticklabels=True,
    ),
    yaxis=dict(
      title="Im(z)",
      showbackground=False,
      showgrid=False,
      showline=True,
      ticks="outside",
      showticklabels=True,
    ),
    zaxis=dict(
      title="|f(z)|",
      range=[0, max_z],
      showbackground=False,
      showgrid=False,
      showline=True,
      ticks="outside",
      showticklabels=True,
    ),
  )
)

fig.update_layout(
  updatemenus=[
    dict(
      type="buttons",
      showactive=False,
      x=0.02,
      y=0.98,
      xanchor="left",
      yanchor="top",
      buttons=[
        dict(
          label="Play",
          method="animate",
          args=[None, {"frame": {"duration": 150, "redraw": True}, "fromcurrent": True}],
        ),
        dict(
          label="Pause",
          method="animate",
          args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
        ),
      ],
    )
  ],
  sliders=[
    dict(
      active=N_values.index(N0) if N0 in N_values else 0,
      currentvalue={"prefix": "N = "},
      pad={"t": 30},
      steps=[
        dict(method="animate", args=[[str(N)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}], label=str(N))
        for N in N_values
      ],
    )
  ],
)

def save_contour_gif(fig, N_values, out_path, cap, max_z, width=1100, height=800, scale=2, frame_duration_s=5.0):
  if iio is None:
    raise RuntimeError("Missing dependency: imageio. Install with `python3 -m pip install -U imageio kaleido`.")

  images = []
  for N in N_values:
    fx, fy, fz = contour_on_surface(N, cap, max_z)
    fig.data[1].x = fx
    fig.data[1].y = fy
    fig.data[1].z = fz
    fig.layout.sliders[0].active = N_values.index(N) if fig.layout.sliders else 0
    png_bytes = fig.to_image(format="png", width=width, height=height, scale=scale)
    images.append(iio.imread(io.BytesIO(png_bytes)))

  iio.imwrite(out_path, images, duration=int(frame_duration_s * 100), loop=0)


SAVE_GIF = True
if SAVE_GIF:
  save_contour_gif(fig, list(range(1, 7)), "cot.gif", cap=cap, max_z=max_z)

fig.show()


