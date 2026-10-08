# physics/glauber.py
r"""Glauber dynamics of Ising-type spin systems on the full ``2**N`` state space, and the stochastic
thermodynamics built on it: heat, entropy production, optimal priors and mismatch cost.

Conventions
-----------
- States are bit strings ``sigma`` in ``0 .. 2**N - 1``, spin ``+1`` where the bit is set.
- A *graph* is ``(rows, dE_J, dE_h_sign)``: ``rows[i, sigma]`` is the index of ``sigma`` with spin
  ``i`` flipped, and the energy cost of that flip is ``J * dE_J[i, sigma] + h * dE_h_sign[i, sigma]``.
- Rates come per bath ``nu`` with inverse temperature ``betas[nu]`` and attempt rate ``gammas[nu]``
  (``get_rates``); the generator is column-oriented, ``dp/dt = K @ p``, so columns of ``K`` sum to 0.
- ``heat_rate_vectors`` gives ``g[nu, sigma]`` with ``Qdot_nu = g[nu] @ p``, heat entering the system.

Families
--------
Graphs (``make_interaction_graph`` square lattice with open edges, ``make_mean_field_graph``,
``make_asymmetric_sk_graph``), dense and matrix-free generators, uniformized propagation, the
exact mean-field reduction to magnetization sectors, and prior optimization
(``find_optimal_prior*``).

Known quirks, kept as they were when this moved out of ``notebooks/stoch_thermo/ising.py``:
``make_interaction_graph`` sets the field term to ``0.0`` (no external field) and builds neighbors
for a square lattice from ``shape[0]`` only, so use square shapes. ``diffrax`` is imported inside
the ODE helpers so importing this module does not require it.
"""
from __future__ import annotations
import jax
from jax import Array, lax
import jax.numpy as jnp
from jax.scipy.linalg import expm
from jax.scipy.special import gammaln
import numpy as np
from functools import partial
from physics.markov import H
from scipy.stats import entropy, poisson
from scipy.optimize import minimize
from scipy.special import softmax, logsumexp
GraphT = tuple[Array, Array, Array]

# ── Lattice helpers ───────────────────────────────────────────────────────────

# ---- Periodic boundaries ----
# def neighbors(i, L):
#     r, c = divmod(i, L)
#     return [((r-1)%L)*L + c, ((r+1)%L)*L + c,
#             r*L + (c-1)%L,   r*L + (c+1)%L]

# ---- Empty boundaries ----
def neighbors(i: int, L: int) -> list[int]:
  r, c = divmod(i, L)
  nbrs = []
  if r > 0:       nbrs.append((r-1)*L + c)   # up
  if r < L-1:     nbrs.append((r+1)*L + c)   # down
  if c > 0:       nbrs.append(r*L + (c-1))   # left
  if c < L-1:     nbrs.append(r*L + (c+1))   # right
  return nbrs

# ---- get all neighbors ----
def compute_site_neighbors(L: int):
  return [neighbors(i, L) for i in range(L * L)]




def dense_generator_from_rates(rows: Array, rates: Array):
  N, n_states = rates.shape
  cols = jnp.arange(n_states)
  K = jnp.zeros((n_states, n_states), dtype=rates.dtype)
  K = K.at[
    rows.reshape(-1), 
    jnp.tile(cols, N),
  ].add(rates.reshape(-1))
  # on diagonal, so cols sum to zero
  return K.at[cols, cols].add(-rates.sum(axis=0))

# ---- for Glauber using G ----
@jax.jit
def apply_glauber_transition(p: Array, graph: GraphT, beta, h, J=1.0):
  rows, dE_J, dE_h_sign = graph
  N = rows.shape[0]
  dE = J * dE_J + h * dE_h_sign
  # flip probability
  flip_prob = jnp.exp(-0.5 * beta * dE) / N
  # flip_prob = jax.nn.sigmoid(-beta * dE) / N

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


### Graph Topology
def make_state_space_graph(N: int):
  nstates = 1 << N
  # - each amsk has exactly one bit set, corresponding to the lattice site
  # - last element of the array is 1<<0, so it is flipping the first bit
  masks = np.array([1 << (N - 1 - i) for i in range(N)], dtype=np.uint64,)

  # enumerate all 2^N states
  states = np.arange(nstates, dtype=np.uint64,)

  # ^ XOR --> σ^mask flips the bit at the site
  # rows[site, σ] = σ ^ masks[site] --> states flipped at each site, which is the index of the flipped state
  # shape (N, 2^N)
  rows = np.stack([states ^ mask for mask in masks]).astype(np.uint64)
  # convert binary bits to spins - shape (N, 2^N)
  # states[None, :] & masks[:, None] -> gives a bitstring with 0 offsite and either 1 or 0 on site
  # --> each element here is an integer of the form 2^site or 0. If the element is a zero, it thus
  # means that this boolean operation converted the whole bitstring to zeros since it was zero at
  # the site bit. Otherwise, if !=0, then it is of form 2^site and had a 1 at that site
  #
  # spins thus gives the spin value at each site of N sites for all 2^N states, resulting in an
  # array of size (N, 2^N)
  spins = np.where((states[None, :] & masks[:, None]) != 0, 1.0, -1.0).astype(np.float32)
  # alternatively,
  # spins = np.where((np.array(rows) & masks[:, None]) == 0, 1.0, -1.0)
  return rows, spins

def make_interaction_graph(shape: tuple[int, int]) -> GraphT:
  r""" only dependent on topology, no params. Square lattice """
  # build on host, output jax arrays
  Lx, Ly = shape
  N = Lx * Ly
  rows, spins = make_state_space_graph(N)

  # get the neighbors for each site - list of lists
  site_neighbors = compute_site_neighbors(Lx)

  # neighbor sum
  nb_sum = np.zeros((N, 1 << N), dtype=np.float32)
  for site in range(N):
    for j in site_neighbors[site]:
      nb_sum[site] += spins[j]

  # should have no numerical issues here since spins are +1, -1 and |nb_sum| <= #neighbors
  # ferromagnetic interaction part
  dE_J = 2.0 * spins * nb_sum
  # external magnetic field interaction part
  # dE_h_sign = 2.0 * spins
  # no external field
  dE_h_sign = 0.0
  # convert to JAX
  return (
    jnp.array(rows, dtype=jnp.uint32), 
    jnp.array(dE_J, dtype=jnp.float32), 
    jnp.array(dE_h_sign, dtype=jnp.float32)
  )

def make_mean_field_graph(N: int) -> GraphT:
  rows, spins = make_state_space_graph(N)
  # magnetic field magnitude?
  # can break this into sectors of identical magnetization
  # if k is the number of up spins, then M = 2k - N
  # there are g_k = (N k) -> N choose k microstates with the same k
  # --> N choose k configurations with k spins up
  # Ex: ++-, +-+, -++ all belong to k=2
  # ---> a permutation of spin labels maps any one 
  #      of these configs to any other, without 
  #      changing the dynamics
  M = spins.sum(axis=0, keepdims=True)              # (1, nstates)
  # 
  # ΔE_i = J (2s_i (M - s_i)) / N + 2hs_i
  # in sector k, i.e., sectors with identical M, 
  # the transition changes in energy (either for flipping an up spin down or vice versa)
  # are clearly identical
  ΔE_J = 2.0 * spins * (M - spins) / N             # (N, nstates)
  dE_h_sign = 2.0 * spins                           # field term
  return (
    jnp.array(rows, dtype=jnp.uint32),
    jnp.array(ΔE_J, dtype=jnp.float32),
    jnp.array(dE_h_sign, dtype=jnp.float32),
  )

def build_mean_field_reduced_system(N, J, h, betas, gammas):
  """Exact mean-field dynamics on sectors k = number of up spins."""
  dtype = jnp.result_type(J, h, betas, gammas)

  k = jnp.arange(N + 1, dtype=dtype)
  # M = \sum_i s_i = k(+1) + (N-k)(-1) = 2k - N
  M = 2.0 * k - N

  # k -> k - 1: flip one of the k up spins down.
  dE_m = 2.0 * J * (M - 1.0) / N + 2.0 * h

  # k -> k + 1: flip one of the N-k down spins up.
  dE_p = -2.0 * J * (M + 1.0) / N - 2.0 * h

  # Bath-resolved aggregate sector rates, shape (n_baths, N+1).
  rates_minus_nu = (
    k[None, :] * gammas[:, None] * jnp.exp(-0.5 * betas[:, None] * dE_m[None, :])
  )
  rates_plus_nu = (
    (N - k)[None, :] * gammas[:, None] * jnp.exp(-0.5 * betas[:, None] * dE_p[None, :])
  )

  rates_minus = rates_minus_nu.sum(axis=0)
  rates_plus = rates_plus_nu.sum(axis=0)
  escape = rates_minus + rates_plus

  # Column-generator convention: dp/dt = K @ p.
  idx = jnp.arange(N + 1)
  K = jnp.zeros((N + 1, N + 1), dtype=dtype)

  # Source k -> destination k-1.
  K = K.at[idx[:-1], idx[1:]].set(rates_minus[1:])

  # Source k -> destination k+1.
  K = K.at[idx[1:], idx[:-1]].set(rates_plus[:-1])

  K = K.at[idx, idx].set(-escape)

  # Heat entering the system through each bath.
  # Shape: (n_baths, N+1).
  heat_rate = (
    rates_minus_nu * dE_m[None, :]
    + rates_plus_nu * dE_p[None, :]
  )

  log_degeneracy = (
    gammaln(jnp.asarray(N + 1, dtype=dtype))
    - gammaln(k + 1.0)
    - gammaln(N - k + 1.0)
  )

  return K, heat_rate, escape, log_degeneracy

def mean_field_micro_entropy(P, log_degeneracy):
  """Full 2**N-state entropy from an (N+1)-sector distribution."""
  # H(P) alone is the entropy of the sector-label, not the original spin dist
  return H(P, axis=-1) + jnp.sum(P * log_degeneracy, axis=-1)

def reduced_step_operators(K, heat_rate, dt):
  n_states = K.shape[0]
  n_baths = heat_rate.shape[0]

  # d/dt [p] = [K  0] [p]
  #      [Q]   [g  0] [Q]
  generator = jnp.zeros(
    (n_states + n_baths, n_states + n_baths),
    dtype=K.dtype,
  )
  generator = generator.at[:n_states, :n_states].set(K)
  generator = generator.at[n_states:, :n_states].set(heat_rate)

  step = expm(generator * dt)

  G = step[:n_states, :n_states]
  heat_operator = step[n_states:, :n_states]

  return G, heat_operator

@partial(jax.jit, static_argnames=("N", "N_max"))
def mean_field_reduced_curves(
  P0, N, J, h, betas, gammas, dt, N_max
):
  K, heat_rate, _, log_deg = build_mean_field_reduced_system(
    N, J, h, betas, gammas
  )
  G, heat_operator = reduced_step_operators(K, heat_rate, dt)

  def step(P, _):
    heat = heat_operator @ P
    P_next = G @ P
    return P_next, (P_next, heat)

  _, (tail, heat_steps) = lax.scan(step, P0, xs=None, length=N_max)

  points = jnp.concatenate([P0[None, :], tail], axis=0)

  Ns = jnp.arange(1, N_max + 1, dtype=P0.dtype)
  P_avgs = jnp.cumsum(points[:-1], axis=0) / Ns[:, None]

  # G @ P_avg, using the telescoping identity.
  GP_avgs = P_avgs + (points[1:] - P0[None, :]) / Ns[:, None]

  entropy_diff = (
    mean_field_micro_entropy(points[1:], log_deg)
    - mean_field_micro_entropy(P0, log_deg)
  )

  pmmc = entropy_diff + Ns * (
    mean_field_micro_entropy(P_avgs, log_deg)
    - mean_field_micro_entropy(GP_avgs, log_deg)
  )

  medium_ep = jnp.cumsum(-(heat_steps @ betas))
  total_ep = entropy_diff + medium_ep

  return pmmc, total_ep

def make_asymmetric_sk_graph(N, Jij, Theta=None):
  rows, spins = make_state_space_graph(N)

  if Theta is None:
    Theta = np.zeros(N)

  # fields[i, sigma] =
  # Theta_i + sum_j Jij[i,j] sigma_j
  fields = Theta[:, None] + Jij @ spins
  return jnp.asarray(rows, dtype=jnp.uint32), jnp.asarray(spins), jnp.asarray(fields)

### Rates

@jax.jit
def sk_fields(Jij, spins, Theta):
  return Theta[:, None] + Jij @ spins

@jax.jit
def get_sk_rates(spins, fields, beta, gamma):
  return gamma * jax.nn.sigmoid(-2.0 * beta * spins * fields)


def sk_ep_rate(p: Array, rows: Array, rates: Array):
  # forward one-way probability flux
  flux_fwd = rates * p[None, :]

  # reverse flux on same edge
  flux_rev = jnp.take_along_axis(flux_fwd, rows, axis=1)

  mask = (flux_fwd > 0) & (flux_rev > 0)

  return 0.5 * jnp.sum((flux_fwd[mask] - flux_rev[mask]) * jnp.log(flux_fwd[mask] / flux_rev[mask]))



def get_rates(graph: GraphT, J: float, h: float, betas: Array, gammas: Array):
  r""" 
  betas,   # shape (n_baths,)
  gammas,  # shape (n_baths,)
  """
  _, dE_J, dE_h_sign = graph
  # N, nstates = dE_J.shape
  # shape: (N, nstates)
  dE = J * dE_J + h * dE_h_sign

  # Bath-resolved rates:
  #     rates_nu[nu, i, sigma] = rate for sigma -> sigma^i through bath nu
  # Total rate after summing physical channels
  # shape: (N, nstates)
  # return gammas[:, None, None] * jax.nn.sigmoid(-betas[:, None, None] * dE[None, :, :])
  return gammas[:, None, None] * jnp.exp(-0.5 * betas[:, None, None] * dE[None, :, :])

@jax.jit
def medium_entropy_rate_vector(rows: Array, rates: Array):
  reverse_rates = jnp.take_along_axis(rates, rows, axis=1)
  return jnp.sum(rates * (jnp.log(rates) - jnp.log(reverse_rates)), axis=0)

def apply_K_from_rates(p: Array, rows: Array, rates: Array):
  r"""
  Compute K @ p without constructing K.

  rows[i, sigma] = configuration sigma with spin i flipped
  rates[i, sigma] = rate sigma -> sigma^i

  d/dt p_σ =Σ_i[k_i(σ^i)p_{σ^i} - k_i(σ)p_σ].
  """
  # flux is probability/time, but, as opposed to current, it is a one-way, directed flow rate

  # Outgoing flux from each source configuration:
  #     flux[i, sigma] = rate(sigma -> sigma^i) p[sigma]
  flux = rates * p[None, :]

  # diagonal / escape contribution
  outgoing = flux.sum(axis=0)

  # add flux to destination states -> incoming contribution
  incoming = jnp.take_along_axis(flux, rows, axis=1).sum(axis=0)
  # incoming = jnp.zeros_like(p).at[rows.reshape(-1)].add(flux.reshape(-1))

  return incoming - outgoing

@partial(jax.jit, static_argnames="transpose")
def apply_from_rates(v, rows, rates, lam, *, transpose: bool=False):
  # P[sigma^i, sigma] = rate_i(sigma) / lam
  flip_prob = rates / lam

  # diagonal of P
  diag = 1.0 - flip_prob.sum(axis=0)

  if transpose:
    # (P^T v)[sigma]
    incoming_value = jnp.sum(flip_prob * v[rows], axis=0)
    return diag * v + incoming_value
  
  # v = p in this case

  # outgoing flux associated with each possible flip
  flux = flip_prob * v[None, :]

  # Because spin flipping is an involution:
  # rows[i, sigma] gives the source for incoming
  # probability into sigma through site i.
  incoming = jnp.take_along_axis(flux, rows, axis=1).sum(axis=0)
  return diag * v + incoming

apply_P_from_rates = apply_from_rates


@jax.jit
def apply_K(p, graph, J, h, betas, gammas):
  rates = get_rates(graph, J, h, betas, gammas).sum(axis=0)
  return apply_K_from_rates(p, graph[0], rates)

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
      return apply_glauber_transition(p, graph, beta, h, J)
  else:
    # Compute rates ONCE, Do not recompute on every application of K.
    # sum over ν
    rates = get_rates(graph, J, h, betas, gammas).sum(axis=0)

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
  w0 = jnp.exp(-mu)

  def body(n, carry):
    q, w, res, weight_sum = carry
    # q_n = P q_{n-1}
    q = apply_P(q)
    # Poisson recurrence:
    #   w_n = w_{n-1} * mu/n
    w = w * mu / n

    res = res + w * q
    return (q, w, res, weight_sum + w)

  init = (p0, w0, w0 * p0, w0)
  _, _, result, weight_sum = lax.fori_loop(1, n_terms, body, init)

  # This is the omitted Poisson-tail mass.
  truncation_mass = 1.0 - weight_sum

  return result, {'mass': truncation_mass, 'lam': lam}

def propagate_ctmc(p0, t, graph: GraphT, J, h, betas, gammas, n_terms=64, use_glauber: bool=False) -> tuple[Array, Array]:
  return _propagate_ctmc(p0, t, graph, J, h, betas, gammas, n_terms, use_glauber)



### Heat 

def heat_rate_vectors(rates_nu, dE):
  r"""
  Returns g[ν, σ], where
      Qdot_nu = g[ν] @ p
  and positive Qdot means heat entering the system.

  For a source configuration σ,
    d/dt Q_ν(σ) = Σ_i k_i^ν(σ) ΔE_i(σ)
  """
  # sum over sites
  return jnp.sum(rates_nu * dE[None, :, :], axis=1)

def build_system_from_graph(graph: GraphT, J: float, h: float, betas: Array, gammas: Array):
  r""" gives dense generator K and heat rates g """
  assert betas.shape[0] == gammas.shape[0]
  rows, dE_J, dE_h_sign = graph
  rates_nu = get_rates(graph, J, h, betas, gammas)
  rates = rates_nu.sum(0)
  dE = J * dE_J + h * dE_h_sign
  g = heat_rate_vectors(rates_nu, dE)
  K = dense_generator_from_rates(rows, rates)
  return K, g

def EP_cost(p, G, c):
  p_tau = G @ p
  return H(p_tau) - H(p) - c @ p


def precompute_ep(K: Array, g: Array, dt):
  if g.ndim == 1:
    g = g.reshape(1, -1)
  
  # (num_baths, num_states)
  n_baths, ns = g.shape
  n_tot = ns + n_baths
  M = jnp.zeros((n_tot, n_tot), dtype=K.dtype)
  M = M.at[:ns, :ns].set(K.T)
  M = M.at[:ns, ns:].set(g.T)
  EM_dt = jnp.asarray(expm(M * dt))
  # A[nu, :] = a_nu
  # c for a single dt step — needed by find_optimal_prior
  A_1step = EM_dt[:ns, ns:].T
  # c = betas @ a
  return EM_dt, A_1step


def make_uniformization_coeffs(lam, dt, N_values, tol=1e-12):
  r""" 
  lam -> rate of the system of superposition of all clocks
  """
  # ensure this works if given a single N
  N_values = np.atleast_1d(np.asarray(N_values, dtype=int))
  Nmax = int(N_values.max())

  # number of uniformization-clock rings accumulated 
  # between time 0 and time t is
  #     R(t) ~ Poisson(μ(t)),     μ(t) = λt
  # expected count grows linearly with time: E[R(t)] = λt
  #   -> λt is the expected number of rings over finite time t
  #   -> Discrete -> t_n = n dt, E[R_n] = λ(n dt).
  # 
  # Mean # of global-clock rings accumulated by time t_n for n = 0,...,Nmax
  # -> saved distributions occur at t_n = n dt, so p_n = p(t_n) = p(n dt)
  mus = lam * (dt * np.arange(Nmax + 1))
  mu_max = mus[-1]

  # TODO idk
  # Need m = 0,...,n_terms-1
  n_terms = int(poisson.isf(tol, mu_max)) + 1
  ms = np.arange(n_terms)


  # W[t, m] = P(M_t = m), 
  # where M_t ~ Poisson(mu_t)
  W = poisson.pmf(ms[None, :], mus[:, None])

  a, b, s = [], [], []
  for Nsteps in N_values:
    # pbar_N =
    # 1/N sum_{n=0}^{N-1} p_n
    a.append(W[:Nsteps].mean(axis=0))

    # p_N
    b.append(W[Nsteps])
    # integral coefficient -> remaining mass / final tail after m for m in ms
    # survival function S(m) = P(M_t > m) = 1 - P(M_t \leq m), 
    # where P(M_t=m) = e^{-λt} (λt)^m / m!
    s.append(poisson.sf(ms, lam * dt * Nsteps) / lam)
    poisson.cdf
  
  # { # Stable tails -- don't use 1 - sum(...)
  #   'avg_tail': np.mean(poisson.sf(n_terms - 1, mus)), 
  #   'final_tail': poisson.sf(n_terms - 1, mus[-1])
  # }
  return jnp.asarray(a), jnp.asarray(b), jnp.asarray(s)

# @partial(jax.jit, static_argnames=("tol", "safety"))
def setup_uniformization(graph, J, h, betas, gammas, dt, tol=1e-12, safety=1.01):
  r""" tol -> smallest probability error """
  rows, dE_J, dE_h_sign = graph

  rates_nu = get_rates(graph, J, h, betas, gammas)
  rates = rates_nu.sum(axis=0)

  escape = rates.sum(axis=0)

  # λ >= max escape rate.
  # Slightly larger makes P have a self-loop everywhere.
  lam = safety * float(jax.device_get(jnp.max(escape)))

  mu = lam * dt

  # Truncate Poisson expansion for ONE dt interval.
  # smallest value m where the survival prob is <= a given probability tol
  mmax = int(poisson.isf(tol, mu))
  ms = np.arange(mmax + 1)

  # exp(K dt) coefficients
  w = poisson.pmf(ms, mu)

  # integral_0^dt exp(K s) ds coefficients:
  #
  # ∫ Pois(m; λs) ds = P[Pois(λdt) > m] / λ
  d = poisson.sf(ms, mu) / lam

  dE = J * dE_J + h * dE_h_sign
  g = heat_rate_vectors(rates_nu, dE)

  return (
    rows,
    rates,
    rates_nu,
    escape,
    g,
    lam,
    jnp.asarray(w, dtype=rates_nu.dtype),
    jnp.asarray(d, dtype=rates_nu.dtype),
  )

@partial(jax.jit, static_argnames=("N_max",))
def get_uniformized_traj(p0, rows, rates: Array, g, lam, w, d, N_max):
  escape = rates.sum(axis=0)
  apply_P = lambda p: apply_from_rates(p, rows, rates, lam)

  def uniformized_step(p):
    # q_m = P^m p
    q = p
    # p(t + dt)
    p_next = w[0] * q
    # integral_0^dt p(t+s) ds
    # occupancy
    occu = d[0] * q

    def body(m, carry):
      q, p_next, occu = carry
      q = apply_P(q)
      p_next = p_next + w[m] * q
      occu = occu + d[m] * q
      return q, p_next, occu

    _, p_next, occu = lax.fori_loop(1, w.shape[0], body, (q, p_next, occu))

    # Q_nu over this dt interval
    heat_step = g @ occu

    # expected number of spin flips over this interval
    activity_step = escape @ occu

    return p_next, heat_step, activity_step
    
  def body(p, _):
    p_next, heat, activity = uniformized_step(p)
    return p_next, (p_next, heat, activity)

  pN, (tail, heat_steps, activity_steps) = lax.scan(body, p0, xs=None, length=N_max)

  pts = jnp.concatenate([p0[None, :], tail], axis=0)

  return pN, pts, heat_steps, activity_steps

@partial(jax.jit, static_argnames=("N_max",))
def get_uniformized_traj2(p0, rows, rates: Array, g, lam, w, d, N_max: int):
  escape = rates.sum(axis=0)
  # P[sigma^i, sigma] = rate_i(sigma) / lam
  flip_prob = rates / lam
  # diagonal of P
  diag = 1.0 - flip_prob.sum(axis=0)
  # independent of p, so gather it once.
  incoming_prob = jnp.take_along_axis(
    flip_prob, rows, axis=1
  )

  def apply_P(p):
    # p changes, so this gather cannot be recomputed
    incoming = jnp.sum(incoming_prob * p[rows], axis=0)
    return diag * p + incoming

  def uniformized_step(p):
    # q_m = P^m p
    q = p
    # p(t + dt)
    p_next = w[0] * q
    # integral_0^dt p(t+s) ds
    occu = d[0] * q

    def body(m, carry):
      q, p_next, occu = carry
      q = apply_P(q)
      p_next = p_next + w[m] * q
      occu = occu + d[m] * q
      return q, p_next, occu

    _, p_next, occu = lax.fori_loop(1, w.shape[0], body, (q, p_next, occu))

    # Q_nu over this dt interval
    heat_step = g @ occu

    # expected number of spin flips over this interval
    activity_step = escape @ occu

    return p_next, heat_step, activity_step
    
  def body(p, _):
    p_next, heat, activity = uniformized_step(p)
    return p_next, (p_next, heat, activity)

  pN, (tail, heat_steps, activity_steps) = lax.scan(body, p0, xs=None, length=N_max)

  pts = jnp.concatenate([p0[None, :], tail], axis=0)

  return pN, pts, heat_steps, activity_steps




@partial(jax.jit, static_argnames=("Nsteps",))
def pmmc_kernel_fast(p0, graph, J, h, betas, gammas, Nsteps, lam, a_coeff, b_coeff):
  rows = graph[0]
  n_terms = a_coeff.shape[0]

  # Crucially: compute rates ONCE per (beta, h) point
  rates = get_rates(graph, J, h, betas, gammas).sum(axis=0)

  def apply_P(p):
    return apply_P_from_rates(p, rows, rates, lam)

  # q_m = P^m p0
  # m = 0 contributions
  p_avg = a_coeff[0] * p0
  p_N   = b_coeff[0] * p0

  def body(m, carry):
    q, p_avg, p_N = carry

    q = apply_P(q)

    p_avg = p_avg + a_coeff[m] * q
    p_N   = p_N   + b_coeff[m] * q

    return q, p_avg, p_N

  _, p_avg, p_N = lax.fori_loop(1, n_terms, body, (p0, p_avg, p_N))
  # F p_avg = p_avg + (p_N - p0) / Nsteps
  #
  # where F = exp(K dt)
  Gp_avg = p_avg + (p_N - p0) / Nsteps

  return H(p_N) - H(p0) + Nsteps * (H(p_avg) - H(Gp_avg))



@partial(jax.jit, static_argnames=("Nsteps",))
def sk_pmmc_fast(p0, rows, spins, fields, beta, gamma, Nsteps, lam, a_coeff, b_coeff):
  rates = get_sk_rates(spins, fields, beta, gamma)

  def apply_P(p):
    return apply_P_from_rates(p, rows, rates, lam)

  n_terms = a_coeff.shape[0]

  # q_m = P^m p0
  q = p0

  p_avg = a_coeff[0] * p0
  p_N   = b_coeff[0] * p0

  def body(m, carry):
    q, p_avg, p_N = carry
    q = apply_P(q)
    p_avg = p_avg + a_coeff[m] * q
    p_N = p_N + b_coeff[m] * q
    return q, p_avg, p_N

  _, p_avg, p_N = lax.fori_loop(1, n_terms, body, (q, p_avg, p_N))

  Gp_avg = p_avg + (p_N - p0) / Nsteps
  return H(p_N) - H(p0) + Nsteps * (H(p_avg) - H(Gp_avg))

@jax.jit
def sk_pmmc_multiN_fast(p0, rows, spins, fields, beta, gamma, N_values, lam, a_coeff, b_coeff, d_coeff):
  rates = get_sk_rates(spins, fields, beta, gamma)
  entropy_rate_vec = medium_entropy_rate_vector(rows, rates)

  apply_P = lambda p: apply_P_from_rates(p, rows, rates, lam)

  n_horizons, n_terms = a_coeff.shape

  q = p0
  p_avgs = a_coeff[:, 0, None] * p0[None, :]
  p_Ns = b_coeff[:, 0, None] * p0[None, :]
  medium = d_coeff[:, 0] * (entropy_rate_vec @ p0)

  def body(m, carry):
    q, p_avgs, medium, p_Ns = carry
    q = apply_P(q)
    p_avgs = p_avgs + a_coeff[:, m, None] * q[None, :]
    medium = medium + d_coeff[:, m] * (entropy_rate_vec @ q)
    p_Ns = p_Ns + b_coeff[:, m, None] * q[None, :]
    return q, p_avgs, medium, p_Ns,

  _, p_avgs, medium, p_Ns = lax.fori_loop(1, n_terms, body, (q, p_avgs, medium, p_Ns))
  entropy_diff = H(p_Ns, axis=1) - H(p0)
  total_ep = entropy_diff + medium
  Gp_avgs = p_avgs + (p_Ns - p0[None, :]) / N_values[:, None]
  return entropy_diff + N_values * (H(p_avgs, axis=1)- H(Gp_avgs, axis=1)), total_ep



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
  apply = lambda p: apply_glauber_transition(p, graph, beta, h, J)
  
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


def find_optimal_prior(G, c, tol=1e-13, maxiter=10000, damping=0.5):
  G, c = np.asarray(G), np.asarray(c)
  n = G.shape[0]
  q = np.ones(n) / n

  for it in range(maxiter):
    Gq = G @ q
    # KKT fixed-point equation:
    # q* ∝ exp(c + G^T log(G q*))
    z = c + G.T @ np.log(Gq)
    q_fp = softmax(z)
    # damping improves stability
    q_new = (1.0 - damping) * q + damping * q_fp
    if np.linalg.norm(q_new - q, ord=1) < tol:
      q = q_new
      break
    q = q_new

  # Check the actual KKT condition
  grad = np.log(q) - G.T @ np.log(G @ q) - c
  # At optimum grad must be constant across states
  lagrange = q @ grad
  kkt = np.max(np.abs(grad - lagrange))
  residual_EP = float(EP_cost(q, G, c))
  return q, residual_EP, {
    "iterations": it + 1,
    "kkt_residual": kkt,
    "converged": kkt < 1e-8,
  }



def stationary_distribution(K):
  vals, vecs = np.linalg.eig(K)
  idx = np.argmin(np.abs(vals))
  pi = np.real(vecs[:, idx])

  if pi.sum() < 0:
    pi = -pi

  pi /= pi.sum()

  if pi.min() < -1e-8:
    raise RuntimeError("bad stationary eigenvector")

  pi = np.maximum(pi, 0)
  return pi / pi.sum()

@partial(jax.jit, static_argnames=("maxiter",))
def find_stationary(rows, rates, lam, tol=1e-9, maxiter=100000):
  nstates = rates.shape[1]

  q0 = jnp.ones(nstates, dtype=rates.dtype) / nstates

  apply_P = lambda p: apply_P_from_rates(p, rows, rates, lam)

  def cond(state):
    q, err, it = state
    return (err > tol) & (it < maxiter)

  def body(state):
    q, _, it = state
    q_new = apply_P(q)
    err = jnp.sum(jnp.abs(q_new - q))
    return q_new, err, it + 1

  pi, err, it = lax.while_loop(cond, body, (q0, jnp.inf, 0))
  return pi / pi.sum(), err, it



@partial(jax.jit, static_argnames=("n_iter",))
def stationary_sk_fast(rows, rates, lam, n_iter):
  ns = rates.shape[1]
  p = jnp.ones(ns) / ns

  def body(_, p):
    return apply_P_from_rates(p, rows, rates, lam)

  return lax.fori_loop(0, n_iter, body, p)


def observables_for_sk(beta, Jij, spins, rows, N, Theta, gamma):
  fields = sk_fields(Jij, spins, Theta)
  rates = get_sk_rates(spins, fields, beta, gamma)
  K = dense_generator_from_rates(rows, rates)
  pi = stationary_distribution(K)
  sigma_dot = sk_ep_rate(pi, rows, rates)
  m_state = spins.mean(axis=0)
  abs_m = np.abs(m_state) @ pi
  return abs_m, sigma_dot / N



@partial(jax.jit, static_argnames="transpose")
def apply_G(p, rows, rates, lam, w: Array, *, transpose: bool=False):
  q = p
  res = w[0] * q

  def body(m, carry):
    q, res = carry
    q = apply_from_rates(q, rows, rates, lam, transpose=transpose)
    res = res + w[m] * q
    return q, res

  _, res = lax.fori_loop(1, w.shape[0], body, (q, res))
  return res

def apply_GT(v, rows, rates, lam, w):
  return apply_G(v, rows, rates, lam, w, transpose=True)

@jax.jit
def integrated_GT(v, rows, rates, lam, d: Array,):
  q = v
  res = d[0] * q

  def body(m, carry):
    q, res = carry
    q = apply_from_rates(q, rows, rates, lam, transpose=True)
    res = res + d[m] * q
    return q, res

  _, res = lax.fori_loop(1, d.shape[0], body, (q, res))
  return res



def EP_cost_rows_rates(p, c, rows, rates, lam, w):
  p_tau = apply_G(p, rows, rates, lam, w)
  return H(p_tau) - H(p) - c @ p


def find_optimal_prior_rows_rates(rows, rates, c, lam, w, tol=1e-13, maxiter=10000, damping=0.5):
  ns = rates.shape[1]
  q = jnp.ones(ns) / ns

  for it in range(maxiter):
    Gq = apply_G(q, rows, rates, lam, w)
    # KKT fixed-point equation:
    # q* ∝ exp(c + G^T log(G q*))
    z = c + apply_GT(jnp.log(Gq), rows, rates, lam, w)
    q_fp = jax.nn.softmax(z)
    # damping improves stability
    q_new = (1.0 - damping) * q + damping * q_fp
    if jnp.linalg.norm(q_new - q, ord=1) < tol:
      q = q_new
      break
    q = q_new

  # Check the actual KKT condition
  grad = jnp.log(q) - apply_GT(jnp.log(apply_G(q, rows, rates, lam, w)), rows, rates, lam, w) - c
  # At optimum grad must be constant across states
  lagrange = q @ grad
  kkt = jnp.max(jnp.abs(grad - lagrange))
  residual_EP = float(EP_cost_rows_rates(q, c, rows, rates, lam, w))
  return q, residual_EP, {
    "iterations": it + 1,
    "kkt_residual": kkt,
    "converged": kkt < 1e-8,
  }




def make_rhs(K, g):
  import diffrax as dfx
  n = K.shape[0]
  # State vector:
  # z = [p_1,...,p_n, Q1, Q2]
  def rhs(t, z, *args):
    p = z[:n]
    dp = K @ p
    dQ = g @ p
    return jnp.concatenate([dp, dQ])

  return dfx.ODETerm(jax.jit(rhs))

def total_ep(p0, term, tau, betas):
  import diffrax as dfx
  n = len(p0)
  solver = dfx.Dopri5()
  z0 = np.concatenate([p0, [0.0, 0.0]])
  sol = dfx.diffeqsolve(term, solver, t0=0, t1=tau, dt0=0.01, y0=z0)
  assert sol.ys is not None
  p_tau = sol.ys[-1, :n]
  Q1 = sol.ys[-1, n]
  Q2 = sol.ys[-1, n+1]
  # total EP
  Sigma = H(p_tau) - H(p0) - betas[0] * Q1 - betas[1] * Q2
  return Sigma, p_tau, Q1, Q2


def make_precomp_rhs(K: Array, tau, g):
  import diffrax as dfx
  n = K.shape[0]
  G = jax.scipy.linalg.expm(K * tau)
  # a_nu = integral_0^tau exp(K^T t) g_nu dt
  # Store a1,a2 as ROWS of A.
  def rhs(t, A_flat, *args):
    A = A_flat.reshape(2, n)
    # row version of da/dt = K^T a + g
    dA = A @ K + g
    return dA.ravel()
  return G, dfx.ODETerm(jax.jit(rhs))

def precomp_ep_cost(p0: Array, term, tau):
  import diffrax as dfx
  n = p0.shape[0]
  solver = dfx.Dopri5()
  y0=jnp.zeros((2 * n,))
  sol = dfx.diffeqsolve(term, solver, t0=0, t1=tau, dt0=0.01, y0=y0)
  assert sol.ys is not None
  A = sol.ys[-1].reshape(2, n)
  return A[0], A[1]

