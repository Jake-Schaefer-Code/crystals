
import jax
from jax import Array, lax
import jax.numpy as jnp
from jax.scipy.special import xlogy
from functools import partial

# ---- entropy and KL divergence use convention that 0 log 0 = 0
# TODO make custom vjp since both branches can participate in autodiff tracing, which is unpleasant
# (see jax xlogy vjp)
def H(p, axis=None):
  r""" Entropy """
  return - jnp.sum(jnp.where(p != 0., xlogy(p, p), jnp.zeros_like(p)), axis=axis)
S = H
Hv = lambda p: H(p, axis=1)
r""" Vectorized entropy """

def KL(p, q, axis=None):
  r""" Relative entropy """
  return jnp.sum(jnp.where(p != 0., xlogy(p, p) - xlogy(p,q), jnp.zeros_like(p)), axis=axis)

KLv = lambda p, q: KL(p, q, axis=1)
r""" Vectorized relative entropy """

def KL2(p, q, axis=None):
  r""" Relative entropy V2 TODO """
  p_ok = p != 0.
  return jnp.sum(jnp.where(p_ok, lax.mul(p, lax.log(p)) - lax.mul(p, lax.log(q)), jnp.zeros_like(p)), axis=axis)

def matrix_powers(G: Array, N: int):
  def step(Gk: Array, _):
    return Gk @ G, Gk

  _, Gs = lax.scan(
    step,
    init=jnp.eye(G.shape[0], dtype=G.dtype),
    xs=None,
    length=N,
  )
  return Gs  # [I, G, G^2, ..., G^(N-1)]



def get_q_star(p: Array, G: Array, n_steps: int):
  Gs = matrix_powers(G, n_steps)
  traj = jax.vmap(lambda G_i: G_i @ p)(Gs)
  return jnp.mean(traj, axis=0)

def bound(p: Array, G: Array, n_steps: int):
  Gs = matrix_powers(G, n_steps)
  Gn = G @ Gs[-1]
  traj = jax.vmap(lambda G_i: G_i @ p)(Gs)
  q_star = jnp.mean(traj, axis=0)
  gen_landauer =  S(p) - S(Gn @ p)
  return -gen_landauer + n_steps * (S(q_star) - S(G @ q_star)), gen_landauer


def bound2(p: Array, G: Array, n_steps: int, *, scan_G: bool=False):
  def _step(carry, _):
    p_i, agg = carry
    return (G @ p_i, agg + p_i), None

  def _step_G(carry, _):
    Gi, agg = carry
    return (G @ Gi, agg + Gi @ p), None

  d = p.shape[0]
  if scan_G:
    init = (jnp.eye(d, d), jnp.zeros((d,)))
    step = _step_G
  else:
    init = (p, jnp.zeros((d,)))
    step = _step

  out, _ = lax.scan(step, init=init, xs=None, length=n_steps)
  # p_n = G^np
  if scan_G:
    Gn, traj_f = out
    q_star = traj_f / n_steps
    gen_landauer =  S(p) - S(Gn @ p)
  else:
    p_n, traj_f = out
    q_star = traj_f / n_steps
    gen_landauer =  S(p) - S(p_n)
  return -gen_landauer + n_steps * (S(q_star) - S(G @ q_star)), gen_landauer

  

def mmc(
  q: Array, 
  G: Array,
  p0: Array,
  n_steps: int,
):
  Gq = G @ q
  # Gs = matrix_powers(G, n_steps)
  def mmc1(p_i, _):
    p_next = G @ p_i
    cost_i = (
      KL(p_i, q) - KL(p_next, Gq)
    )
    return p_next, cost_i
  
  _, costs = lax.scan(mmc1, init=p0, xs=None, length = n_steps)
  return jnp.sum(costs)


# unjitted, and constructing this function in the T loop, it is much slower than just a 
# for loop with a numpy array. However, jitted beforehand, it is much faster -> 2-3 orders of magnitude faster
@partial(jax.jit, static_argnames=("N_max",))
def get_traj(G, p0, N_max):
  # p <- G @ p
  def traj(p, x):
    return G @ p, p # one mat-vec per step, not mat-mat
  pN, pts = lax.scan(traj, init=p0, length=N_max)
  return pN, jnp.concatenate((pts, pN[None]))

