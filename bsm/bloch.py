import dataclasses as dcls
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D


σx = np.array([[0, 1], [1, 0]], dtype=complex)
σy = np.array([[0, -1j], [1j, 0]], dtype=complex)
σz = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)



def normalize(v):
  v = np.asarray(v, dtype=complex)
  return v / np.linalg.norm(v)

def state_from_angles(theta, phi):
  # |psi> = cos(theta/2)|0> + e^{i phi} sin(theta/2)|1>
  return np.array([
    np.cos(theta / 2),
    np.exp(1j * phi) * np.sin(theta / 2)
  ], dtype=complex)

def complex_state_to_3d_real_state(psi: np.ndarray):
  return np.array([
    psi[1].real, 
    psi[1].imag, 
    psi[0].real,
  ], dtype=float)

def bloch_vector(psi):
  psi = normalize(psi)
  return np.array([
    np.real(np.vdot(psi, σx @ psi)),
    np.real(np.vdot(psi, σy @ psi)),
    np.real(np.vdot(psi, σz @ psi)),
  ], dtype=float)


def effect_to_bloch(E):
  # Write E = (beta/2)(I + m·σ)
  beta = float(np.real(np.trace(E)))
  if beta < 1e-14:
    return beta, np.zeros(3)

  m = np.array([
    np.real(np.trace(E @ σx)),
    np.real(np.trace(E @ σy)),
    np.real(np.trace(E @ σz)),
  ], dtype=float) / beta

  return beta, m


@dcls.dataclass
class Effect:
  u: np.ndarray
  alpha: float

  def __post_init__(self):
    self.u = normalize(self.u)
    self.alpha = float(self.alpha)

  @property
  def E(self):
    return self.alpha * np.outer(self.u, self.u.conj())

  @property
  def n(self):
    return bloch_vector(self.u)

  def probability(self, psi):
    psi = normalize(psi)
    return float(np.real(np.vdot(psi, self.E @ psi)))
  
  def __matmul__(self, other: np.ndarray) -> np.ndarray:
    return self.E @ other
  
  def __add__(self, other: np.ndarray) -> np.ndarray:
    return self.E + other

  def __neg__(self) -> np.ndarray:
    return -self.E



def plot_ball3(ax: Axes3D):
  u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
  xs = np.sin(v) * np.cos(u)
  ys = np.sin(v) * np.sin(u)
  zs = np.cos(v)
  ax.plot_surface(xs, ys, zs, color='lightblue', alpha=0.3)
  return ax

def plot_vec_3d_from_angles(ax: Axes3D, th: np.ndarray, ph: np.ndarray):
  vector, = ax.plot(
    [0, np.sin(th) * np.cos(ph)],
    [0, np.sin(th) * np.sin(ph)],
    [0, np.cos(th)],
    color='r', linewidth=2
  )
  return vector


def make_effects(alpha):
  ket1 = np.array([0, 1], dtype=complex)
  ket_minus = normalize(np.array([1, -1], dtype=complex))

  E1 = Effect(ket1, alpha)
  E2 = Effect(ket_minus, alpha)
  E3 = I2 - E1.E - E2.E
  return E1, E2, E3

def set_axes_equal_3d(ax: Axes3D):
  xlim = ax.get_xlim3d()
  ylim = ax.get_ylim3d()
  zlim = ax.get_zlim3d()

  xmid = 0.5 * (xlim[0] + xlim[1])
  ymid = 0.5 * (ylim[0] + ylim[1])
  zmid = 0.5 * (zlim[0] + zlim[1])

  radius = 0.5 * max(
    xlim[1] - xlim[0],
    ylim[1] - ylim[0],
    zlim[1] - zlim[0],
  )
  ax.set(
    xlim3d=[xmid - radius, xmid + radius],
    ylim3d=[ymid - radius, ymid + radius],
    zlim3d=[zmid - radius, zmid + radius],
  )

def draw_bloch_sphere(ax: Axes3D):
  u, v = np.mgrid[0:2*np.pi:120j, 0:np.pi:60j]
  x = np.sin(v) * np.cos(u)
  y = np.sin(v) * np.sin(u)
  z = np.cos(v)

  ax.plot_surface(x, y, z, alpha=0.12)
  ax.plot([-1, 1], [0, 0], [0, 0], lw=1)
  ax.plot([0, 0], [-1, 1], [0, 0], lw=1)
  ax.plot([0, 0], [0, 0], [-1, 1], lw=1)

  ax.set(
    xlabel=r'$x$', ylabel=r'$y$', zlabel=r'$z$',
    xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), zlim=(-1.1, 1.1)
  )
  set_axes_equal_3d(ax)


def plot_arrow(ax: Axes3D, vec, **kwargs):
  # 3D quiver ignores linestyle on the shaft (and update() was dropping kwargs anyway).
  # ax.plot draws a real dashed/solid line from the origin.
  linestyle = kwargs.get('linestyle', '-')
  if linestyle not in (None, '-', 'solid'):
    plot_kw = {k: v for k, v in kwargs.items() if k != 'arrow_length_ratio'}
    (line,) = ax.plot(
      [0, vec[0]], [0, vec[1]], [0, vec[2]],
      **plot_kw,
    )
    return line
  kwargs = {k: v for k, v in kwargs.items() if k != 'linestyle'}
  return ax.quiver(
    0, 0, 0,
    vec[0], vec[1], vec[2],
    arrow_length_ratio=kwargs.pop('arrow_length_ratio', 0.12),
    **kwargs,
  )


def interactive_povm_3d(theta0=np.pi/3, phi0=np.pi/4, alpha0=0.4):
  fig = plt.figure(figsize=(12, 6))
  ax_ball = fig.add_subplot(121, projection='3d')
  assert isinstance(ax_ball, Axes3D)
  ax_bar = fig.add_subplot(122)
  assert isinstance(ax_bar, Axes)

  plt.subplots_adjust(bottom=0.28, wspace=0.35)

  draw_bloch_sphere(ax_ball)
  ax_ball.set(title='Bloch sphere')

  # static labels for the effect states

  ket1_vec = bloch_vector(np.array([0, 1])) # |1>
  ketp_vec = bloch_vector(np.sqrt(2) * np.array([1, 1])) # |+>


  ax_ball.scatter(*ket1_vec, s=60)
  ax_ball.text(*ket1_vec, r'  $|1\rangle$')


  ax_ball.scatter(*ketp_vec, s=60)
  ax_ball.text(*ketp_vec, r'  $|+\rangle$')

  psi_arrow = plot_arrow(ax_ball, np.array([0, 0, 1]), linewidth=2)
  e1_arrow  = plot_arrow(ax_ball, ket1_vec, linewidth=2, linestyle='--', color='grey')
  e2_arrow  = plot_arrow(ax_ball, ketp_vec, linewidth=2, linestyle='--', color='grey')
  e3_arrow  = plot_arrow(ax_ball, np.array([0, 0, 0]), linewidth=2, linestyle='--', color='black')

  bars = ax_bar.bar([r'$p_1$', r'$p_2$', r'$p_3$'], [0, 0, 0])
  ax_bar.set(ylim=(0,1), ylabel='probability',title='Outcome probabilities')

  txt = ax_bar.text(
    0.02, 0.98, '',
    transform=ax_bar.transAxes,
    ha='left', va='top'
  )

  ax_theta = plt.axes([0.20, 0.14, 0.65, 0.03])
  ax_phi = plt.axes([0.20, 0.09, 0.65, 0.03])
  ax_alpha = plt.axes([0.20, 0.04, 0.65, 0.03])

  s_theta = Slider(ax_theta, r'$\theta$', 0, np.pi, valinit=theta0)
  s_phi = Slider(ax_phi, r'$\phi$', 0, 2*np.pi, valinit=phi0)
  s_alpha = Slider(ax_alpha, r'$\alpha$', 0, 0.8, valinit=alpha0)

  def redraw_quiver(old_quiver, vec, **kwargs):
    old_quiver.remove()
    return plot_arrow(ax_ball, vec, **kwargs)

  def update(_=None):
    nonlocal psi_arrow, e1_arrow, e2_arrow, e3_arrow

    theta = s_theta.val
    phi = s_phi.val
    alpha = s_alpha.val

    psi = state_from_angles(theta, phi)
    r = bloch_vector(psi)

    E1, E2, E3 = make_effects(alpha)
    beta3, m3 = effect_to_bloch(E3)
    evals3 = np.linalg.eigvalsh(E3)

    p1 = E1.probability(psi)
    p2 = E2.probability(psi)
    p3 = float(np.real(np.vdot(psi, E3 @ psi)))

    psi_arrow = redraw_quiver(psi_arrow, r, linewidth=2)
    e1_arrow = redraw_quiver(
      e1_arrow, E1.n, linewidth=2, linestyle='--', color='grey',
    )
    e2_arrow = redraw_quiver(
      e2_arrow, E2.n, linewidth=2, linestyle='--', color='grey',
    )
    e3_arrow = redraw_quiver(e3_arrow, m3, linewidth=2, linestyle='--', color='black')

    for bar, p in zip(bars, [p1, p2, p3]):
      bar.set_height(max(0.0, p))

    valid = np.min(evals3) >= -1e-10

    txt.set_text(
      f"r = ({r[0]:.3f}, {r[1]:.3f}, {r[2]:.3f})\n"
      f"p1 = {p1:.3f}\n"
      f"p2 = {p2:.3f}\n"
      f"p3 = {p3:.3f}\n"
      f"eig(E3) = [{evals3[0]:.3f}, {evals3[1]:.3f}]\n"
      f"valid POVM: {valid}"
    )

    ax_ball.set_title(
      'Bloch sphere' if valid else 'Bloch sphere (E3 not positive)'
    )

    fig.canvas.draw_idle()

  s_theta.on_changed(update)
  s_phi.on_changed(update)
  s_alpha.on_changed(update)
  update()
  plt.show()



interactive_povm_3d()
