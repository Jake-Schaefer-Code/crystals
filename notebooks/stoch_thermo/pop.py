import jax
from jax import lax
from jax import Array
import jax.numpy as jnp
from jax.scipy.special import xlogy
jax.config.update('jax_enable_x64', True)
import numpy as np
from numpy.linalg import matrix_power
from scipy.stats import entropy
from itertools import product as iproduct
from scipy.linalg import expm
from time import perf_counter
from functools import partial
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

# ---- entropy and KL divergence use convention that 0 log 0 = 0
def H(p, axis=None):
  return - jnp.sum(jnp.where(p != 0., p * jnp.log(p), jnp.zeros_like(p)), axis=axis)

def KL(p, q, axis=None):
  return jnp.sum(jnp.where(p != 0., xlogy(p,p) - xlogy(p,q), jnp.zeros_like(p)), axis=axis)

def KL2(p, q, axis=None):
  p_ok = p != 0.
  return jnp.sum(jnp.where(p_ok, lax.mul(p, lax.log(p)) - lax.mul(p, lax.log(q)), jnp.zeros_like(p)), axis=axis)

# ── Periodic boundaries ──────────────────────────────────
# def neighbors(i, L):
#     r, c = divmod(i, L)
#     return [((r-1)%L)*L + c, ((r+1)%L)*L + c,
#             r*L + (c-1)%L,   r*L + (c+1)%L]

# ── Empty boundaries ──────────────────────────────────
def neighbors(i: int, L: int) -> list[int]:
  r, c = divmod(i, L)
  nbrs = []
  if r > 0:       nbrs.append((r-1)*L + c)   # up
  if r < L-1:     nbrs.append((r+1)*L + c)   # down
  if c > 0:       nbrs.append(r*L + (c-1))   # left
  if c < L-1:     nbrs.append(r*L + (c+1))   # right
  return nbrs

def compute_site_neighbors(L: int):
  return [neighbors(i, L) for i in range(L * L)]



# ── Build K^ν — every spin coupled to every reservoir ──
def build_K(nu, L, gamma, beta, J, h):
  N = L * L
  n_states = 1 << N
  # ── States ───────────────────────────────────
  states    = [np.array(s, dtype=np.int8) for s in iproduct([-1,1], repeat=N)]
  state_idx = {tuple(s): k for k, s in enumerate(states)}

  def dE_flip(sigma, i):
    return 2.0 * sigma[i] * (J * sum(sigma[j] for j in neighbors(i, L)) + h)

  def rate(dE, beta, gamma):
    return gamma * np.exp(-0.5 * beta * dE)

  
  K = np.zeros((n_states, n_states))

  # (2^N, N) operations
  for col, sigma in enumerate(states):
    for i in range(N):                          # all spins

      eta    = sigma.copy()
      eta[i] *= -1
      row = state_idx[tuple(eta)]

      dE = dE_flip(sigma, i)
      r = rate(dE, beta[nu], gamma[nu])
      K[row, col] += r

  # print(np.allclose(np.diag(K), 0))
  np.fill_diagonal(K, -K.sum(axis=0))
  return K


# ── Propagator and time evolution ─────────────
def make_propagator(K):
  def propagator(t):
    return expm(K * t)
  return propagator

GraphT = tuple[Array, Array, Array]

def make_interaction_graph(shape: tuple[int, int]) -> GraphT:
  r""" only dependent on topology, no params """
  # build on host, output jax arrays
  Lx, Ly = shape
  N = Lx * Ly
  nstates = 1 << N


  # - each amsk has exactly one bit set, corresponding to the lattice site
  # - last element of the array is 1<<0, so it is flipping the first bit
  masks = np.array([1 << (N - 1 - site) for site in range(N)], dtype=np.int64)

  # ^ XOR --> σ^mask flips the bit at the site
  # rows[site, σ] = σ ^ masks[site] --> states flipped at each site, which is the index of the flipped state
  # shape (N, 2^N)
  rows = [np.arange(nstates, dtype=np.int64) ^ mask for mask in masks]

  # get the neighbors for each site - list of lists
  site_neighbors = compute_site_neighbors(Lx)

  # enumerate all 2^N states
  states = np.arange(nstates, dtype=np.int64)
  # convert binary bits to spins - shape (N, 2^N)
  # states[None, :] & masks[:, None] -> gives a bitstring with 0 offsite and either 1 or 0 on site
  # --> each element here is an integer of the form 2^site or 0. If the element is a zero, it thus
  # means that this boolean operation converted the whole bitstring to zeros since it was zero at
  # the site bit. Otherwise, if !=0, then it is of form 2^site and had a 1 at that site
  #
  # spins thus gives the spin value at each site of N sites for all 2^N states, resulting in an
  # array of size (N, 2^N)
  spins = np.where((states[None, :] & masks[:, None]) != 0, 1.0, -1.0)
  # alternatively,
  # spins = np.where((np.array(rows) & masks[:, None]) == 0, 1.0, -1.0)

  # neighbor sum
  nb_sum = np.zeros((N, nstates), dtype=np.float32)
  for site in range(N):
    for j in site_neighbors[site]:
      nb_sum[site] += spins[j]

  # should have no numerical issues here since spins are +1, -1 and |nb_sum| <= #neighbors
  # ferromagnetic interaction part
  dE_J = 2.0 * spins * nb_sum
  # external magnetic field interaction part
  dE_h_sign = 2.0 * spins
  # convert to JAX
  return (
    jnp.array(rows, dtype=jnp.int64), 
    jnp.array(dE_J, dtype=jnp.float32), 
    jnp.array(dE_h_sign, dtype=jnp.float32)
  )


@jax.jit
def apply_glauber_old(p: Array, graph: GraphT, beta, h, J=1.0):
  rows, dE_J, dE_h_sign = graph
  N = rows.shape[0]
  dE = J * dE_J + h * dE_h_sign

  # flip probability
  # off = jnp.exp(-beta * jnp.maximum(dE, 0.0)) / N
  flip_prob = jax.nn.sigmoid(-beta * dE) / N

  # prob of remaining in original state
  diag = 1.0 - flip_prob.sum(axis=0)

  p_next = diag * p
  
  def body(site, res):
    # rows[site, sigma] is sigma with site flipped. 
    # => rows[site] is all possible states with flipped at `site`
    #     - this is also the index of the state with this flipped site, which is used to get energy
    incoming = (flip_prob[site] * p)[rows[site]]
    return res + incoming


  # row -> index of state with site i flipped
  # col -> index of sigma
  # G[row, col] = G[sigma^i, sigma] = prob of going from sigma to sigma^i
  # go through flipping each of N sites
  return lax.fori_loop(0, N, body, p_next)

@jax.jit
def apply_glauber_old_v2(p: Array, graph: GraphT, beta, h, J=1.0):
  rows, dE_J, dE_h_sign = graph
  N = rows.shape[0]
  dE = J * dE_J + h * dE_h_sign

  # flip probability
  # off = jnp.exp(-beta * jnp.maximum(dE, 0.0)) / N
  flip_prob = jax.nn.sigmoid(-beta * dE) / N

  # prob of remaining in original state
  diag = 1.0 - flip_prob.sum(axis=0)
  p_next = diag * p
  # this is equivalent to the for loop
  idk = flip_prob * p[None, :]
  incoming = jnp.take_along_axis(idk, rows, axis=1).sum(axis=0)
  return incoming + p_next

def get_rates(graph: GraphT, J: float, h: float, betas: Array, gammas: Array):
  r""" 
  betas,   # shape (n_baths,)
  gammas,  # shape (n_baths,)
  """
  rows, dE_J, dE_h_sign = graph

  # shape: (N, nstates)
  dE = J * dE_J + h * dE_h_sign

  # Bath-resolved rates:
  #     rates_nu[nu, i, sigma] = rate for sigma -> sigma^i through bath nu
  # Total rate after summing physical channels
  # shape: (N, nstates)
  # rates = (
  #   gammas[:, None, None]
  #   * jnp.exp(-0.5 * betas[:, None, None] * dE[None, :, :])
  # ).sum(axis=0)
  rates = (gammas[:, None, None] * jax.nn.sigmoid(-betas[:, None, None] * dE[None, :, :])).sum(axis=0)
  return rates, dE

def apply_K_from_rates(p: Array, rows: Array, rates: Array):
  """
  Compute K @ p without constructing K.

  rows[i, sigma] = configuration sigma with spin i flipped
  rates[i, sigma] = rate sigma -> sigma^i
  """
  # Outgoing flux from each source configuration:
  #     flux[i, sigma] = rate(sigma -> sigma^i) p[sigma]
  flux = rates * p[None, :]

  # diagonal / escape contribution
  dp = -flux.sum(axis=0)

  # add flux to destination states
  incoming = jnp.zeros_like(p).at[rows.reshape(-1)].add(
    flux.reshape(-1)
  )

  return dp + incoming


def apply_P_from_rates(p, rows, rates, lam):
  # P[sigma^i, sigma] = rate_i(sigma) / lam
  flip_prob = rates / lam

  # diagonal of P
  diag = 1.0 - flip_prob.sum(axis=0)

  # outgoing flux associated with each possible flip
  flux = flip_prob * p[None, :]

  # Because spin flipping is an involution:
  # rows[i, sigma] gives the source for incoming
  # probability into sigma through site i.
  incoming = jnp.take_along_axis(flux, rows, axis=1).sum(axis=0)

  return diag * p + incoming

@jax.jit
def apply_K(p, graph, J, h, betas, gammas):
  rates, _ = get_rates(graph, J, h, betas, gammas)
  return apply_K_from_rates(p, graph[0], rates)


def apply_K_glauber(p: Array, graph: GraphT, beta, h, J=1.0):
  rows, dE_J, dE_h_sign = graph
  dE = J * dE_J + h * dE_h_sign

  # CTMC RATE, not probability
  rates = jax.nn.sigmoid(-beta * dE)

  flux = rates * p[None, :]

  # outgoing contribution
  dp = -flux.sum(axis=0)

  # incoming contribution
  incoming = jnp.zeros_like(p).at[rows.reshape(-1)].add(
    flux.reshape(-1)
  )
  return dp + incoming


def make_uniformization_coeffs(lam, dt, Nsteps, n_terms=128, dtype=np.float64):
  # Means corresponding to p_0, ..., p_{M-1}
  mus = lam * dt * np.arange(Nsteps, dtype=dtype)

  # Mean corresponding to p_M
  muN = lam * dt * Nsteps

  # m = 0 Poisson weights
  w_avg = np.exp(-mus)
  w_N = np.exp(-muN)

  a = np.empty(n_terms, dtype=dtype)
  b = np.empty(n_terms, dtype=dtype)

  a[0] = w_avg.mean()
  b[0] = w_N

  for m in range(1, n_terms):
    w_avg *= mus / m
    w_N *= muN / m

    a[m] = w_avg.mean()
    b[m] = w_N

  avg_tail = 1.0 - a.sum()
  final_tail = 1.0 - b.sum()

  return (jnp.asarray(a), jnp.asarray(b), avg_tail, final_tail)


@partial(jax.jit, static_argnames=("Nsteps",))
def pmmc_kernel_fast(
    p0,
    graph,
    J,
    h,
    betas,
    gammas,
    Nsteps,
    lam,
    a_coeff,
    b_coeff,
):
  rows = graph[0]

  # Crucially: compute rates ONCE per (beta, h) point
  rates, _ = get_rates(graph, J, h, betas, gammas)

  def apply_P(p):
    return apply_P_from_rates(p, rows, rates, lam)

  # q_m = P^m p0
  q = p0

  # m = 0 contributions
  p_avg = a_coeff[0] * q
  p_N   = b_coeff[0] * q

  def body(m, carry):
    q, p_avg, p_N = carry

    q = apply_P(q)

    p_avg = p_avg + a_coeff[m] * q
    p_N   = p_N   + b_coeff[m] * q

    return q, p_avg, p_N

  _, p_avg, p_N = lax.fori_loop(1, a_coeff.shape[0], body, (q, p_avg, p_N))

  #
  # F p_avg = p_avg + (p_N - p0) / Nsteps
  #
  # where F = exp(K dt)
  Gp_avg = p_avg + (p_N - p0) / Nsteps

  return H(p_N) - H(p0) + Nsteps * (H(p_avg) - H(Gp_avg))



@partial(jax.jit, static_argnames=("n_terms", "use_glauber"))
def _propagate_ctmc(p0: Array, t, graph: GraphT, J, h, betas, gammas, n_terms=64, use_glauber: bool=False):
  rows, dE_J, dE_h_sign = graph
  if use_glauber:
    N = rows.shape[0]
    # Uniformization rate
    lam = N

    if len(betas) > 0:
      beta = betas[0]
    else:
      beta = betas

    def apply_P(p):
      return apply_glauber_old(p, graph, beta, h, J)
  else:
    # Compute rates ONCE.
    # Do not recompute them on every application of K.
    rates, _ = get_rates(graph, J, h, betas, gammas)

    # sum over sites
    escape = rates.sum(axis=0)

    # Uniformization rate
    lam = jnp.max(escape)

    # Avoid division by zero in pathological K=0 case
    lam_safe = jnp.where(lam > 0, lam, 1.0)

    # P = I + K / λ
    def apply_P(p):
      Kp = apply_K_from_rates(p, rows, rates)
      return p + Kp / lam_safe

  mu = lam * t

  # n = 0 term
  weight0 = jnp.exp(-mu)

  def body(n, carry):
    q, w, res, weight_sum = carry
    # q_n = P q_{n-1}
    q = apply_P(q)
    # Poisson recurrence:
    #   w_n = w_{n-1} * mu/n
    w = w * mu / n

    res = res + w * q
    return (q, w, res, weight_sum + w)

  init = (p0, weight0, weight0 * p0, weight0)
  _, _, result, weight_sum = lax.fori_loop(1, n_terms, body, init)

  # This is the omitted Poisson-tail mass.
  truncation_mass = 1.0 - weight_sum

  return result, {'mass': truncation_mass, 'lam': lam}

def propagate_ctmc(p0, t, graph: GraphT, J, h, betas, gammas, n_terms=64, use_glauber: bool=False) -> tuple[Array, Array]:
  return _propagate_ctmc(p0, t, graph, J, h, betas, gammas, n_terms, use_glauber)



# apply = lambda p: propagate_ctmc(
#     p, dt, graph, J, h, betas, gammas, n_terms=32
#   )[0]
def make_pmmc_kernel(apply):
  def body(_, carry):
    p, p_avg = carry
    return (apply(p), p_avg + p)

  def pmmc_kernel(p0: Array, Nsteps: int):
    p0 = jnp.asarray(p0)
    init = (p0, jnp.zeros_like(p0))
    pN, p_avg = lax.fori_loop(0, Nsteps, body, init)
    p_avg = p_avg / Nsteps
    Gp_avg = p_avg + (pN - p0) / Nsteps
    return H(pN) - H(p0) + Nsteps * (H(p_avg) - H(Gp_avg))

  return pmmc_kernel


def pmmc_kernel_old(p0, graph, beta, h, J, Nsteps):
  p0 = jnp.asarray(p0)
  apply = lambda p: apply_glauber_old(p, graph, beta, h, J)
  
  def body(_, carry):
    p, p_avg = carry
    return (apply(p), p_avg + p)

  pN, p_avg = lax.fori_loop(
    0,
    Nsteps,
    body,
    (p0, jnp.zeros_like(p0)),
  )

  p_avg = p_avg / Nsteps
  Gp_avg = p_avg + (pN - p0) / Nsteps
  # Gp_avg = apply(p_avg)
  return H(pN) - H(p0) + Nsteps * (H(p_avg) - H(Gp_avg))

# ── Parameters ──────────────────────────────
L      = 3
shape  = (L, L)
N      = L * L
n_states = 1 << N
p0 = np.ones(1 << N) / (1 << N)

J      = 1.0
h      = 0.5
T1, T2 = 4.0, 1.0
gamma = {1: 0.5, 2: 0.5}
beta   = {1: 1/T1, 2: 1/T2}
betas_jax = jnp.array([beta[1], beta[2]])
gammas_jax = jnp.array([gamma[1], gamma[2]])

# critical temperature
T_c = 2 * J / np.log(1 + np.sqrt(2))
print(T_c)

graph = make_interaction_graph(shape)

# Parameter ranges
h_vals = jnp.linspace(0, 4, 20)
beta_vals = jnp.linspace(0, 4, 20)
BB, HH = jnp.meshgrid(beta_vals, h_vals, indexing="ij")
params = jnp.stack([BB.ravel(), HH.ravel()], axis=1)

def plot_heatmap(ax: Axes, heatmap):
  return ax.imshow(
    heatmap,
    origin='lower',
    extent=[h_vals[0], h_vals[-1], beta_vals[0], beta_vals[-1]],
    aspect='auto',
    cmap='turbo'
  )

def compare_heatmaps(h1, h2):
  # fig, ax = plt.subplots(1, 1, figsize=(8,6), squeeze=True)
  fig, _axs = plt.subplots(1, 3, figsize=(20,6), squeeze=False)
  axs: list[Axes] = _axs.flatten().tolist()
  delta = np.asarray(h1 - h2)
  delta_absmax = float(np.max(np.abs(delta)))
  delta_norm = mcolors.TwoSlopeNorm(
    vmin=-(delta_absmax if delta_absmax > 0.0 else 1.0),
    vcenter=0.0,
    vmax=(delta_absmax if delta_absmax > 0.0 else 1.0),
  )

  im = plot_heatmap(axs[0], h1)
  axs[0].set(xlabel=r"$h$", ylabel=r"$\beta$", title=r"MMC for $K$")
  fig.colorbar(im, ax=axs[0], label=r"$\mathcal{MC}_N$")

  im = plot_heatmap(axs[1], h2)
  axs[1].set(xlabel=r"$h$", ylabel=r"$\beta$", title=r"MMC for $G$")
  fig.colorbar(im, ax=axs[1], label=r"$\mathcal{MC}_N$")

  im = axs[2].imshow(
    delta,
    origin='lower',
    extent=[h_vals[0], h_vals[-1], beta_vals[0], beta_vals[-1]],
    aspect='auto',
    cmap='seismic',
    norm=delta_norm,
  )
  axs[2].set(xlabel=r"$h$", ylabel=r"$\beta$", title=r"$\Delta$MMC")
  fig.colorbar(im, ax=axs[2], label=r"$\Delta\mathcal{MC}_N$")
  fig.suptitle(fr"Minimal Periodic MMC for $N={Nsteps}$, $|\Lambda|={N}$, $dt={dt}$, $h={h}$, $J={J}$")
  plt.tight_layout()
  plt.show()

  fig.savefig(f"discrete_vs_continuous_N{Nsteps}_Lam{N}.pdf")

# Number of steps to include in the cumulative MMC
Nsteps = 16

p0 = jnp.ones((1 << N)) / (1 << N)

dt = 1 / N

lam = N * float(sum(gamma.values()))




a_coeff, b_coeff, avg_tail, final_tail = make_uniformization_coeffs(lam, dt, Nsteps, n_terms=32)

print(avg_tail, final_tail)

def pmmc_eval(betas, h):
  return pmmc_kernel_fast(
      p0, graph, J, h, betas, gammas_jax, Nsteps, lam, a_coeff, b_coeff,
  )

@jax.jit
def pmmc_row(betas, h_vals):
  return lax.map(lambda h: pmmc_eval(betas, h), h_vals)


@jax.jit
def eval_heatmap(params):
  def one(x):
    beta, h = x
    return pmmc_eval(jnp.array([beta, beta]), h)
  
  return lax.map(one, params, batch_size=4)

# _ = pmmc_eval(jnp.array([beta_vals[0], beta_vals[0]]), h_vals[0]).block_until_ready()  # one compile

heatmap = np.empty((len(beta_vals), len(h_vals)), dtype=np.float32)
for i, beta in enumerate(beta_vals):
  heatmap[i] = np.asarray(pmmc_row(
    jnp.array([beta, beta]), h_vals,
  ))

heatmap = np.asarray(eval_heatmap(params)).reshape(len(beta_vals), len(h_vals))


# ── Parameters ──────────────────────────────
dt = 0.1
N_max = 10000


plt.figure()

# unjitted, and constructing this function in the T loop, it is much slower than just a 
# for loop with a numpy array. However, jitted beforehand, it is much faster -> 2-3 orders of magnitude faster
@partial(jax.jit, static_argnames=("N_max",))
def get_traj(G, p0, N_max):
  # p <- G @ p
  def traj(p, x):
    return G @ p, p # one mat-vec per step, not mat-mat
  pN, pts = lax.scan(traj, init=p0, length=N_max)
  return pN, jnp.concatenate((pts, pN[None]))

for T in range(1, 12):

  beta   = {1: 1/T1, 2: 1/T}
  K1 = build_K(1, L, gamma, beta, J, h)
  K2 = build_K(2, L, gamma, beta, J, h)
  K  = K1 + K2

  # ── Propagator and time evolution ─────────────
  propagator = make_propagator(K)


  G  = jnp.array(propagator(dt))

  # p <- G @ p
  # precompute all p_t = G^t @ p0 as a matrix — one pass
  pN, pts = get_traj(G, p0, N_max)

  
  # cumulative average: p_avg(N) = (1/N) Σ_{t=0}^{N-1} p_t
  cumsum = jnp.cumsum(pts[:-1], axis=0)           # shape (N_max, n_states)
  Ns     = jnp.arange(1, N_max + 1)
  p_avgs = cumsum / Ns[:, None]                  # broadcasting


  Gp_avgs = p_avgs @ G.T                         # G @ p_avg for each N

  # H(G^N p0) - H(p0) + N*(H(p_avg) - H(G p_avg))
  # vectorized over rows
  pmmcs = (
    H(pts[1:], axis=1) - H(p0) + Ns * (H(p_avgs, axis=1) - H(Gp_avgs, axis=1))
  )


  # stationary state — null eigenvector of K
  vals, vecs = np.linalg.eig(G)
  idx = np.argmin(np.abs(vals - 1.0))    # eigenvalue closest to 1, not 0
  pi  = np.real(vecs[:, idx])
  pi  = np.clip(pi / pi.sum(), 0, None)

  kl  = KL(p0, pi)     # KL(p0 || pi) = Σ p log(p/q)
  plt.hlines(y=kl, xmin = 0, xmax = Ns[-1], color = 'black', linestyles='dashed')
  plt.semilogx(Ns, pmmcs, label=f"T2 = {T}")


plt.xlabel("N iterations")
plt.ylabel("PMMC")
# plt.yscale("log")
plt.title(f"PMMC vs iterations  (dt={dt}, T1={T1}, T2 varied)")
plt.legend(title="Temperature")
plt.tight_layout()


def energy(sigma, L):
  E = -h * np.sum(sigma)

  for r in range(L):
    for c in range(L):
      i = r * L + c

      # Count each bond once
      if r < L - 1:
        j = (r + 1) * L + c
        E -= J * sigma[i] * sigma[j]

      if c < L - 1:
        j = r * L + (c + 1)
        E -= J * sigma[i] * sigma[j]
  return E

def heat_rate_vector(Knu, energies):
  # dE[x, y] = E_x - E_y
  dE = energies[:, None] - energies[None, :]

  # g[y] = sum_x Knu[x,y] (E_x - E_y)
  # Diagonal contributes zero automatically.
  return np.sum(Knu * dE, axis=0)


import diffrax as dfx

def make_rhs(K, g):
  n = K.shape[0]
  # State vector:
  # z = [p_1,...,p_n, Q1, Q2]
  def rhs(t, z, *args):
    p = z[:n]
    dp = K @ p
    dQ = g @ p
    return jnp.concatenate([dp, dQ])

  return dfx.ODETerm(jax.jit(rhs))

def total_ep(p0, term, tau):
  n = len(p0)
  solver = dfx.Dopri5()

  z0 = np.concatenate([p0, [0.0, 0.0]])
  sol = dfx.diffeqsolve(
    term,
    solver,
    t0=0,
    t1=tau,
    dt0=0.01,
    y0=z0,
  )

  p_tau = sol.ys[-1, :n]
  Q1 = sol.ys[-1, n]
  Q2 = sol.ys[-1, n+1]
  # total EP
  Sigma = H(p_tau) - H(p0) - beta[1] * Q1 - beta[2] * Q2
  return Sigma, p_tau, Q1, Q2


def make_precomp_rhs(K: Array, tau, g):
  n = K.shape[0]
  G = jax.scipy.linalg.expm(K * tau)

  # a_nu = integral_0^tau exp(K^T t) g_nu dt
  #
  # Store a1,a2 as ROWS of A.
  
  def rhs(t, A_flat, *args):
    A = A_flat.reshape(2, n)
    # row version of da/dt = K^T a + g
    dA = A @ K + g
    return dA.ravel()
  
  return G, dfx.ODETerm(jax.jit(rhs))

def precomp_ep_cost(p0: Array, term, tau):
  n = p0.shape[0]
  solver = dfx.Dopri5()

  sol = dfx.diffeqsolve(
    term,
    solver,
    t0=0,
    t1=tau,
    dt0=0.01,
    y0=jnp.zeros((2 * n,)),
  )
  A = sol.ys[-1].reshape(2, n)
  a1 = A[0]
  a2 = A[1]
  return a1, a2


states    = [np.array(s, dtype=np.int8) for s in iproduct([-1,1], repeat=N)]
energies = np.array([energy(s, L) for s in states])
g1 = heat_rate_vector(K1, energies)
g2 = heat_rate_vector(K2, energies)
g = np.stack((g1, g2), axis=0)

tau = 1.0

p0 = np.ones(n_states) / n_states

rhs = make_rhs(K, g)
Sigma, p_tau, Q1, Q2 = total_ep(
    p0, rhs, tau
)

print("Q1 =", Q1)
print("Q2 =", Q2)
print("Total EP =", Sigma)



def EP_cost(p, G, c):
  p_tau = G @ p
  return H(p_tau) - H(p) - c @ p

G, rhs = make_precomp_rhs(K, tau, g)
a1, a2 = precomp_ep_cost(p0, rhs, tau)
c = beta[1] * a1 + beta[2] * a2
Sigma = EP_cost(p0, G, c)
print(Sigma)



from scipy.optimize import minimize
from scipy.special import softmax
from scipy.special import logsumexp

def find_optimal_prior(G, c, tol=1e-13, maxiter=10000, damping=0.5, q0=None):
  n = G.shape[0]

  if q0 is None:
    q = np.ones(n) / n
  else:
    q = np.asarray(q0, dtype=float)
    q = q / q.sum()

  for it in range(maxiter):
    Gq = G @ q

    if np.any(Gq <= 0):
      raise ValueError("G @ q contains nonpositive entries.")

    # KKT fixed-point equation:
    # q* ∝ exp(c + G^T log(G q*))
    log_q_new = c + G.T @ np.log(Gq)

    # stable softmax
    log_q_new -= logsumexp(log_q_new)
    q_new = np.exp(log_q_new)

    # damping improves stability
    q_next = (1.0 - damping) * q + damping * q_new
    q_next /= q_next.sum()

    if np.linalg.norm(q_next - q, ord=1) < tol:
      q = q_next
      break

    q = q_next

  # Check the actual KKT condition
  grad = np.log(q) - G.T @ np.log(G @ q) - c

  # At optimum grad must be constant across states
  lam = np.dot(q, grad)
  kkt_residual = np.max(np.abs(grad - lam))

  residual_EP = EP_cost(q, G, c)

  return q, residual_EP, {
    "iterations": it + 1,
    "kkt_residual": kkt_residual,
    "converged": it + 1 < maxiter,
  }

q_star, residual_EP, info = find_optimal_prior(G, c)

print("residual EP =", residual_EP)
print("KKT residual =", info["kkt_residual"])
print("iterations =", info["iterations"])

from scipy.special import softmax
pi = softmax(-beta[1] * energies)

gibbs_EP = EP_cost(pi, G, c)

print("stationarity:", np.linalg.norm(K @ pi))
print("EP at Gibbs:", gibbs_EP)

p_tau = G @ p0
q_tau = G @ q_star
pi_tau = G @ pi


total_EP = EP_cost(p0, G, c)

MC_from_KL = KL(p0, q_star) - KL(p_tau, q_tau)

MC_from_EP = total_EP - residual_EP

print("Total EP =", total_EP)
print("Residual EP =", residual_EP)
print("MC from EP difference =", MC_from_EP)
print("MC from KL contraction =", MC_from_KL)
print("difference =", MC_from_EP - MC_from_KL)



MC_from_EP = total_EP - gibbs_EP

MC_from_KL = KL(p0, pi) - KL(p_tau, pi_tau)

print("MC from EP difference =", MC_from_EP)
print("MC from KL contraction =", MC_from_KL)
print("difference =", MC_from_EP - MC_from_KL)
